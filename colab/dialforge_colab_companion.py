#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from faster_whisper import WhisperModel

TOKEN = os.environ.get("DIALFORGE_COMPANION_TOKEN", "").strip()
MODEL = os.environ.get("DIALFORGE_COMPANION_MODEL", "qwen3:4b-instruct-2507-q8_0").strip()
OLLAMA = os.environ.get("OLLAMA_BASE", "http://127.0.0.1:11434").rstrip("/")
ROOT = Path(os.environ.get("DIALFORGE_COMPANION_ROOT", "/content/dialforge-colab-companion"))
VOICE = ROOT / "reference.wav"
MAX_VOICE_BYTES = 32 * 1024 * 1024

if not TOKEN:
    raise RuntimeError("DIALFORGE_COMPANION_TOKEN is required")

ROOT.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="Dialforge Colab Companion", docs_url=None, redoc_url=None)
_load_lock = threading.RLock()
_tts_lock = threading.RLock()
_stt_lock = threading.RLock()
_stt: WhisperModel | None = None
_tts: ChatterboxTurboTTS | None = None
_voice_revision = 0
_voice_prepared_revision = -1


def _authorized(request: Request) -> bool:
    header = request.headers.get("authorization", "")
    return header == f"Bearer {TOKEN}"


@app.middleware("http")
async def auth(request: Request, call_next):
    if not _authorized(request):
        return JSONResponse({"detail": "unauthorized"}, status_code=401)
    return await call_next(request)


def _gpu() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False, "name": None, "vram_mb": 0}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "vram_mb": int(props.total_memory // (1024**2)),
    }


def _ensure_stt() -> WhisperModel:
    global _stt
    with _load_lock:
        if _stt is None:
            _stt = WhisperModel(
                "small.en",
                device="cuda" if torch.cuda.is_available() else "cpu",
                compute_type="int8",
            )
        return _stt


def _ensure_tts() -> ChatterboxTurboTTS:
    global _tts
    with _load_lock:
        if _tts is None:
            _tts = ChatterboxTurboTTS.from_pretrained(
                device="cuda" if torch.cuda.is_available() else "cpu",
                nano=True,
            )
        return _tts


def _prepare_voice() -> None:
    global _voice_prepared_revision
    tts = _ensure_tts()
    if not VOICE.exists():
        raise RuntimeError("Dialforge voice reference has not been uploaded")
    if _voice_prepared_revision == _voice_revision:
        return
    tts.prepare_conditionals(str(VOICE), exaggeration=0.0, norm_loudness=True)
    _voice_prepared_revision = _voice_revision


def _tensor_wav_bytes(tensor, sample_rate: int) -> bytes:
    audio = tensor.detach().float().cpu()
    if audio.ndim > 1:
        audio = audio[0]
    pcm = (audio.clamp(-1, 1).numpy() * 32767).astype(np.int16)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        path = Path(handle.name)
    try:
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(int(sample_rate))
            wav.writeframes(pcm.tobytes())
        return path.read_bytes()
    finally:
        path.unlink(missing_ok=True)


def _validate_voice_wav(path: Path) -> dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            width = wav.getsampwidth()
            duration = wav.getnframes() / float(sample_rate)
    except Exception as exc:
        raise ValueError("Voice reference must be a valid WAV file") from exc
    if channels != 1 or width != 2 or sample_rate != 24000:
        raise ValueError("Dialforge remote voice must be normalized 24 kHz mono 16-bit WAV")
    if not 6.0 <= duration <= 20.0:
        raise ValueError("Voice reference must be between 6 and 20 seconds")
    return {"duration_seconds": round(duration, 2), "sample_rate": sample_rate, "channels": channels}


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "dialforge-colab-companion",
        "model": MODEL,
        "gpu": _gpu(),
        "voice_ready": VOICE.exists() and _voice_prepared_revision == _voice_revision,
        "stt_loaded": _stt is not None,
        "tts_loaded": _tts is not None,
    }


@app.post("/dialforge/voice-reference")
async def voice_reference(file: UploadFile = File(...)):
    global _voice_revision, _voice_prepared_revision
    total = 0
    with tempfile.NamedTemporaryFile(dir=ROOT, suffix=".wav", delete=False) as output:
        staged = Path(output.name)
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_VOICE_BYTES:
                staged.unlink(missing_ok=True)
                raise HTTPException(413, "Voice reference exceeds 32 MB")
            output.write(chunk)
    await file.close()
    try:
        if total == 0:
            raise ValueError("Voice reference is empty")
        metadata = _validate_voice_wav(staged)
        staged.replace(VOICE)
        _voice_revision += 1
        _voice_prepared_revision = -1
        with _tts_lock:
            _prepare_voice()
        return {"ok": True, "revision": _voice_revision, **metadata}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        staged.unlink(missing_ok=True)


@app.post("/dialforge/warmup")
async def warmup(request: Request):
    started = time.perf_counter()
    requested = MODEL
    try:
        payload = await request.json()
        requested = str(payload.get("model") or MODEL).strip()
    except Exception:
        pass
    if requested != MODEL:
        raise HTTPException(409, f"This Colab runtime is loaded for {MODEL}; reconnect Dialforge using that model")

    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(
            OLLAMA + "/api/generate",
            json={
                "model": MODEL,
                "prompt": "Reply only with OK.",
                "stream": False,
                "think": False,
                "keep_alive": "30m",
                "options": {"num_predict": 4, "temperature": 0},
            },
        )
        response.raise_for_status()
    stt = await asyncio.to_thread(_ensure_stt)
    with _stt_lock:
        pass
    await asyncio.to_thread(_ensure_tts)
    with _tts_lock:
        _prepare_voice()
        tts = _ensure_tts()
        _ = tts.generate(
            "Thanks for taking the call.",
            exaggeration=0.0,
            cfg_weight=0.0,
            temperature=0.8,
            norm_loudness=True,
        )
    return {
        "ok": True,
        "model": MODEL,
        "gpu": _gpu(),
        "voice_ready": True,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
    }


@app.post("/v1/audio/transcriptions")
async def transcription(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: str | None = Form(None),
    response_format: str | None = Form(None),
):
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(dir=ROOT, suffix=suffix, delete=False) as output:
        path = Path(output.name)
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    await file.close()
    try:
        stt = await asyncio.to_thread(_ensure_stt)

        def run_stt() -> str:
            with _stt_lock:
                segments, _info = stt.transcribe(
                    str(path),
                    language="en" if not language else language,
                    vad_filter=True,
                    beam_size=1,
                )
                return " ".join(seg.text.strip() for seg in segments if seg.text.strip()).strip()

        text = await asyncio.to_thread(run_stt)
        return {"text": text}
    finally:
        path.unlink(missing_ok=True)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    payload["model"] = MODEL
    if payload.get("max_completion_tokens") is not None and payload.get("max_tokens") is None:
        payload["max_tokens"] = payload.pop("max_completion_tokens")
    payload.pop("reasoning_effort", None)
    stream = bool(payload.get("stream"))

    if not stream:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(OLLAMA + "/v1/chat/completions", json=payload)
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type=response.headers.get("content-type", "application/json"),
            )

    async def proxy_stream():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", OLLAMA + "/v1/chat/completions", json=payload) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    yield body
                    return
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk

    return StreamingResponse(proxy_stream(), media_type="text/event-stream")


@app.post("/v1/audio/speech")
async def speech(request: Request):
    payload = await request.json()
    text = str(payload.get("input") or "").strip()
    if not text:
        raise HTTPException(400, "TTS input is empty")
    if len(text) > 2000:
        raise HTTPException(400, "TTS input is too long")

    def run_tts() -> bytes:
        with _tts_lock:
            _prepare_voice()
            tts = _ensure_tts()
            wav = tts.generate(
                text,
                exaggeration=0.0,
                cfg_weight=0.0,
                temperature=0.8,
                norm_loudness=True,
            )
            return _tensor_wav_bytes(wav, tts.sr)

    audio = await asyncio.to_thread(run_tts)
    return Response(content=audio, media_type="audio/wav")
