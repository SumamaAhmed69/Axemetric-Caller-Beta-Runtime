#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import math
import os
import re
import secrets
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

TOKEN = (
    os.environ.get("LINEBORN_COMPANION_TOKEN")
    or os.environ.get("DIALFORGE_COMPANION_TOKEN")
    or ""
).strip()
MODEL = (
    os.environ.get("LINEBORN_COMPANION_MODEL")
    or os.environ.get("DIALFORGE_COMPANION_MODEL")
    or "qwen3:4b-instruct"
).strip()
OLLAMA = os.environ.get("OLLAMA_BASE", "http://127.0.0.1:11434").rstrip("/")
ROOT = Path(
    os.environ.get("LINEBORN_COMPANION_ROOT")
    or os.environ.get("DIALFORGE_COMPANION_ROOT")
    or "/content/lineborn-colab-runtime"
)
VOICE = ROOT / "reference.wav"
REPORT = ROOT / "lineborn-prerelease-benchmark.json"
MAX_VOICE_BYTES = 32 * 1024 * 1024
MAX_TRANSCRIPTION_BYTES = 16 * 1024 * 1024
STARTED_AT = time.time()
BENCHMARK_VERSION = 1

if not TOKEN:
    raise RuntimeError("LINEBORN_COMPANION_TOKEN is required")

ROOT.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="Lineborn Colab Runtime", docs_url=None, redoc_url=None)

_load_lock = threading.RLock()
_tts_lock = threading.RLock()
_stt_lock = threading.RLock()
_benchmark_lock = threading.Lock()
_stt: WhisperModel | None = None
_tts: ChatterboxTurboTTS | None = None
_voice_revision = 0
_voice_prepared_revision = -1

STRICT_GATES = {
    "errors_max": 0,
    "llm_ttft_median_max_ms": 1200.0,
    "llm_ttft_p95_max_ms": 1800.0,
    "voice_start_median_max_ms": 3000.0,
    "voice_start_p95_max_ms": 4500.0,
    "stt_wer_median_max": 0.18,
    "tts_rtf_median_max": 0.70,
    "pipeline_p95_max_ms": 5500.0,
}

BENCHMARK_INPUTS = (
    "Hi, I have a minute. What is this call about?",
    "We already have an agency and I am happy with them.",
    "I am busy right now. Can you keep this very short?",
    "Do not promise me rankings or results. What can you actually verify?",
    "Tuesday afternoon might work, but I have not agreed to a meeting yet.",
    "If I ask you not to call again, will you actually stop calling?",
    "Just give me the basic reason you reached out in one sentence.",
)


def _authorized(request: Request) -> bool:
    header = request.headers.get("authorization", "")
    return secrets.compare_digest(header, f"Bearer {TOKEN}")


@app.middleware("http")
async def auth(request: Request, call_next):
    if not _authorized(request):
        return JSONResponse({"detail": "unauthorized"}, status_code=401)
    return await call_next(request)


def _gpu() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "available": False,
            "name": None,
            "vram_mb": 0,
            "allocated_mb": 0,
            "reserved_mb": 0,
        }
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "vram_mb": int(props.total_memory // (1024**2)),
        "allocated_mb": round(torch.cuda.memory_allocated(0) / (1024**2), 1),
        "reserved_mb": round(torch.cuda.memory_reserved(0) / (1024**2), 1),
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
        raise RuntimeError("Lineborn voice reference has not been uploaded")
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


def _wav_duration_bytes(data: bytes) -> float:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        path = Path(handle.name)
        handle.write(data)
    try:
        with wave.open(str(path), "rb") as wav:
            return wav.getnframes() / float(wav.getframerate())
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
        raise ValueError("Lineborn remote voice must be normalized 24 kHz mono 16-bit WAV")
    if not 6.0 <= duration <= 20.0:
        raise ValueError("Voice reference must be between 6 and 20 seconds")
    return {
        "duration_seconds": round(duration, 2),
        "sample_rate": sample_rate,
        "channels": channels,
    }


def _normalize_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", str(text or "").lower())


def _wer(reference: str, hypothesis: str) -> float:
    ref = _normalize_words(reference)
    hyp = _normalize_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, rword in enumerate(ref, 1):
        current = [i]
        for j, hword in enumerate(hyp, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (0 if rword == hword else 1),
                )
            )
        previous = current
    return previous[-1] / len(ref)


def _pct(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return round(xs[0], 2)
    position = (len(xs) - 1) * percentile / 100.0
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return round(xs[lo], 2)
    return round(xs[lo] * (hi - position) + xs[hi] * (position - lo), 2)


async def _llm_stream_measure(prompt: str) -> dict[str, Any]:
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Lineborn in a benchmark. Reply naturally and concisely. "
                    "Use at most two short sentences. Never claim an action happened "
                    "unless the prompt says it happened."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "temperature": 0.05,
        "top_p": 0.72,
        "max_tokens": 72,
    }
    started = time.perf_counter()
    first_token_at: float | None = None
    parts: list[str] = []
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", OLLAMA + "/v1/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw or raw == "[DONE]":
                    continue
                try:
                    item = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                choices = item.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = str(delta.get("content") or "")
                if content:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    parts.append(content)
    finished = time.perf_counter()
    text = "".join(parts).strip()
    if first_token_at is None or not text:
        raise RuntimeError("Qwen streaming benchmark produced no text")
    return {
        "text": text,
        "ttft_ms": round((first_token_at - started) * 1000.0, 2),
        "total_ms": round((finished - started) * 1000.0, 2),
    }


def _tts_generate(text: str) -> dict[str, Any]:
    with _tts_lock:
        _prepare_voice()
        tts = _ensure_tts()
        started = time.perf_counter()
        audio = tts.generate(
            text,
            exaggeration=0.0,
            cfg_weight=0.0,
            temperature=0.8,
            norm_loudness=True,
        )
        elapsed = time.perf_counter() - started
        wav = _tensor_wav_bytes(audio, tts.sr)
    duration = max(_wav_duration_bytes(wav), 0.001)
    return {
        "wav": wav,
        "elapsed_ms": round(elapsed * 1000.0, 2),
        "duration_s": round(duration, 3),
        "rtf": round(elapsed / duration, 4),
    }


def _stt_transcribe_wav_bytes(data: bytes) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(dir=ROOT, suffix=".wav", delete=False) as handle:
        path = Path(handle.name)
        handle.write(data)
    try:
        stt = _ensure_stt()
        started = time.perf_counter()
        with _stt_lock:
            segments, _info = stt.transcribe(
                str(path), language="en", vad_filter=True, beam_size=1
            )
            text = " ".join(
                segment.text.strip()
                for segment in segments
                if segment.text.strip()
            ).strip()
        return {
            "text": text,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
        }
    finally:
        path.unlink(missing_ok=True)


async def _warm_runtime() -> dict[str, Any]:
    started = time.perf_counter()
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
    await asyncio.to_thread(_ensure_stt)
    await asyncio.to_thread(_ensure_tts)
    with _tts_lock:
        _prepare_voice()
        tts = _ensure_tts()
        _ = tts.generate(
            "Lineborn runtime ready.",
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


@app.get("/health")
@app.get("/lineborn/health")
def health():
    return {
        "ok": True,
        "service": "lineborn-colab-runtime",
        "benchmark_version": BENCHMARK_VERSION,
        "model": MODEL,
        "gpu": _gpu(),
        "voice_ready": VOICE.exists() and _voice_prepared_revision == _voice_revision,
        "stt_loaded": _stt is not None,
        "tts_loaded": _tts is not None,
        "uptime_seconds": round(time.time() - STARTED_AT, 1),
    }


@app.post("/lineborn/voice-reference")
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
        await asyncio.to_thread(_prepare_voice)
        return {"ok": True, "revision": _voice_revision, **metadata}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        staged.unlink(missing_ok=True)


@app.post("/lineborn/warmup")
@app.post("/dialforge/warmup")
async def warmup(request: Request):
    requested = MODEL
    try:
        payload = await request.json()
        requested = str(payload.get("model") or MODEL).strip()
    except Exception:
        pass
    if requested != MODEL:
        raise HTTPException(
            409,
            f"This Colab runtime is loaded for {MODEL}; reconnect Lineborn using that model",
        )
    try:
        return await _warm_runtime()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/v1/audio/transcriptions")
async def transcription(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: str | None = Form(None),
    response_format: str | None = Form(None),
):
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    total = 0
    with tempfile.NamedTemporaryFile(dir=ROOT, suffix=suffix, delete=False) as output:
        path = Path(output.name)
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_TRANSCRIPTION_BYTES:
                path.unlink(missing_ok=True)
                raise HTTPException(413, "Audio transcription upload exceeds 16 MB")
            output.write(chunk)
    await file.close()
    try:
        if total == 0:
            raise HTTPException(400, "Audio transcription upload is empty")
        stt = await asyncio.to_thread(_ensure_stt)

        def run_stt() -> str:
            with _stt_lock:
                segments, _info = stt.transcribe(
                    str(path),
                    language="en" if not language else language,
                    vad_filter=True,
                    beam_size=1,
                )
                return " ".join(
                    segment.text.strip()
                    for segment in segments
                    if segment.text.strip()
                ).strip()

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
            async with client.stream(
                "POST", OLLAMA + "/v1/chat/completions", json=payload
            ) as response:
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
    try:
        result = await asyncio.to_thread(_tts_generate, text)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return Response(content=result["wav"], media_type="audio/wav")


@app.post("/lineborn/benchmark")
async def strict_benchmark(request: Request):
    if not _benchmark_lock.acquire(blocking=False):
        raise HTTPException(409, "A Lineborn benchmark is already running")
    try:
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        iterations = max(3, min(int(payload.get("iterations") or 7), 10))
        if not torch.cuda.is_available():
            raise HTTPException(409, "Strict pre-release benchmark requires a Colab NVIDIA GPU")
        if not VOICE.exists():
            raise HTTPException(
                409,
                "Voice reference is not uploaded. Pair Lineborn, choose a voice, and Start AI before benchmarking.",
            )

        await _warm_runtime()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)

        seed_inputs: list[dict[str, Any]] = []
        for phrase in BENCHMARK_INPUTS[:iterations]:
            generated = await asyncio.to_thread(_tts_generate, phrase)
            seed_inputs.append({"reference": phrase, "wav": generated["wav"]})

        trials: list[dict[str, Any]] = []
        errors: list[str] = []
        for index in range(iterations):
            seed = seed_inputs[index % len(seed_inputs)]
            reference = str(seed["reference"])
            started = time.perf_counter()
            try:
                stt_result = await asyncio.to_thread(_stt_transcribe_wav_bytes, seed["wav"])
                llm_result = await _llm_stream_measure(stt_result["text"] or reference)
                tts_result = await asyncio.to_thread(_tts_generate, llm_result["text"])
                pipeline_ms = (time.perf_counter() - started) * 1000.0
                voice_start_ms = (
                    float(stt_result["elapsed_ms"])
                    + float(llm_result["ttft_ms"])
                    + float(tts_result["elapsed_ms"])
                )
                trials.append(
                    {
                        "index": index + 1,
                        "reference": reference,
                        "transcript": stt_result["text"],
                        "response": llm_result["text"],
                        "stt_ms": stt_result["elapsed_ms"],
                        "stt_wer": round(_wer(reference, stt_result["text"]), 4),
                        "llm_ttft_ms": llm_result["ttft_ms"],
                        "llm_total_ms": llm_result["total_ms"],
                        "tts_ms": tts_result["elapsed_ms"],
                        "tts_duration_s": tts_result["duration_s"],
                        "tts_rtf": tts_result["rtf"],
                        "voice_start_ms": round(voice_start_ms, 2),
                        "pipeline_ms": round(pipeline_ms, 2),
                    }
                )
            except Exception as exc:
                errors.append(f"trial {index + 1}: {exc}")

        def vals(key: str) -> list[float]:
            return [float(item[key]) for item in trials if item.get(key) is not None]

        metrics = {
            "iterations_requested": iterations,
            "iterations_completed": len(trials),
            "errors": len(errors),
            "llm_ttft_median_ms": _pct(vals("llm_ttft_ms"), 50),
            "llm_ttft_p95_ms": _pct(vals("llm_ttft_ms"), 95),
            "llm_total_median_ms": _pct(vals("llm_total_ms"), 50),
            "stt_median_ms": _pct(vals("stt_ms"), 50),
            "stt_wer_median": _pct(vals("stt_wer"), 50),
            "tts_median_ms": _pct(vals("tts_ms"), 50),
            "tts_rtf_median": _pct(vals("tts_rtf"), 50),
            "voice_start_median_ms": _pct(vals("voice_start_ms"), 50),
            "voice_start_p95_ms": _pct(vals("voice_start_ms"), 95),
            "pipeline_median_ms": _pct(vals("pipeline_ms"), 50),
            "pipeline_p95_ms": _pct(vals("pipeline_ms"), 95),
        }

        gate_results = {
            "errors": metrics["errors"] <= STRICT_GATES["errors_max"],
            "llm_ttft_median": metrics["llm_ttft_median_ms"] is not None and metrics["llm_ttft_median_ms"] <= STRICT_GATES["llm_ttft_median_max_ms"],
            "llm_ttft_p95": metrics["llm_ttft_p95_ms"] is not None and metrics["llm_ttft_p95_ms"] <= STRICT_GATES["llm_ttft_p95_max_ms"],
            "voice_start_median": metrics["voice_start_median_ms"] is not None and metrics["voice_start_median_ms"] <= STRICT_GATES["voice_start_median_max_ms"],
            "voice_start_p95": metrics["voice_start_p95_ms"] is not None and metrics["voice_start_p95_ms"] <= STRICT_GATES["voice_start_p95_max_ms"],
            "stt_wer": metrics["stt_wer_median"] is not None and metrics["stt_wer_median"] <= STRICT_GATES["stt_wer_median_max"],
            "tts_rtf": metrics["tts_rtf_median"] is not None and metrics["tts_rtf_median"] <= STRICT_GATES["tts_rtf_median_max"],
            "pipeline_p95": metrics["pipeline_p95_ms"] is not None and metrics["pipeline_p95_ms"] <= STRICT_GATES["pipeline_p95_max_ms"],
            "all_iterations_completed": len(trials) == iterations,
        }

        gpu = _gpu()
        gpu["peak_allocated_mb"] = round(torch.cuda.max_memory_allocated(0) / (1024**2), 1)
        gpu["peak_reserved_mb"] = round(torch.cuda.max_memory_reserved(0) / (1024**2), 1)

        report = {
            "ok": all(gate_results.values()),
            "strict_pass": all(gate_results.values()),
            "benchmark": "lineborn-prerelease-compute-v1",
            "model": MODEL,
            "generated_at": time.time(),
            "gpu": gpu,
            "gates": STRICT_GATES,
            "gate_results": gate_results,
            "metrics": metrics,
            "errors": errors,
            "trials": trials,
            "scope": (
                "Remote AI compute path only: Whisper -> Qwen -> Chatterbox. "
                "Direct SIP/PSTN, local deterministic call control, installer, and OS integration require separate Lineborn release checks."
            ),
        }
        temp = REPORT.with_suffix(".tmp")
        temp.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temp.replace(REPORT)
        return report
    finally:
        _benchmark_lock.release()


@app.get("/lineborn/benchmark/latest")
def benchmark_latest():
    if not REPORT.exists():
        raise HTTPException(404, "No strict benchmark report has been generated in this session")
    try:
        return json.loads(REPORT.read_text("utf-8"))
    except Exception as exc:
        raise HTTPException(500, f"Benchmark report could not be read: {exc}") from exc
