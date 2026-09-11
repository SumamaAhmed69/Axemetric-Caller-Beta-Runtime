from __future__ import annotations

import io
import json
import os
from pathlib import Path
from threading import RLock

import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from chatterbox.tts_turbo import ChatterboxTurboTTS

PORT = int(os.getenv("AXEMETRIC_CHATTERBOX_PORT", "8881"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NANO = os.getenv("AXEMETRIC_CHATTERBOX_NANO", "1") != "0"


def data_root() -> Path:
    override = os.getenv("AXEMETRIC_DATA_ROOT")
    if override:
        return Path(override)
    if os.name == "nt":
        return Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "Axemetric Caller"
    return Path.home() / ".local" / "share" / "axemetric-caller"


MARKER = data_root() / "runtime" / "voice-reference.json"
app = FastAPI(title="Axemetric Chatterbox", docs_url=None, redoc_url=None)
_lock = RLock()
_model: ChatterboxTurboTTS | None = None
_loaded_revision = -1


class SpeechRequest(BaseModel):
    model: str = "chatterbox"
    input: str
    voice: str = "default"
    response_format: str = "wav"
    speed: float = 1.0


def model() -> ChatterboxTurboTTS:
    global _model
    with _lock:
        if _model is None:
            _model = ChatterboxTurboTTS.from_pretrained(device=DEVICE, nano=NANO)
        return _model


def marker() -> dict:
    if not MARKER.exists():
        return {"revision": 0, "path": None}
    try:
        return json.loads(MARKER.read_text("utf-8"))
    except Exception:
        return {"revision": 0, "path": None}


def refresh_voice(m: ChatterboxTurboTTS) -> None:
    global _loaded_revision
    state = marker()
    revision = int(state.get("revision", 0))
    if revision == _loaded_revision:
        return
    ref = state.get("path")
    if ref and Path(ref).exists():
        m.prepare_conditionals(str(ref), exaggeration=0.0, norm_loudness=True)
        _loaded_revision = revision
    elif _loaded_revision < 0:
        _loaded_revision = revision


@app.get("/health")
def health():
    state = marker()
    return {"ok": True, "device": DEVICE, "nano": NANO, "loaded": _model is not None, "voice_revision": state.get("revision", 0)}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "chatterbox", "object": "model", "owned_by": "axemetric-local"}]}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    text = req.input.strip()
    if not text:
        raise HTTPException(400, "input is required")
    if len(text) > 1200:
        raise HTTPException(400, "input is too long")
    m = model()
    with _lock:
        refresh_voice(m)
        try:
            wav = m.generate(text, exaggeration=0.0, cfg_weight=0.0, temperature=0.8, norm_loudness=True)
        except AssertionError as exc:
            raise HTTPException(409, "Choose a voice reference before synthesizing speech") from exc
        buffer = io.BytesIO()
        torchaudio.save(buffer, wav.detach().cpu(), m.sr, format="wav")
        return Response(buffer.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
