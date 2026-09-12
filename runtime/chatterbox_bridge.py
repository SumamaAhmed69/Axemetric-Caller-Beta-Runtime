from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import time
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
REFERENCE = data_root() / "voices" / "reference.wav"
app = FastAPI(title="Dialforge Chatterbox", docs_url=None, redoc_url=None)
_lock = RLock()
_model: ChatterboxTurboTTS | None = None
_loaded_revision = -1
_load_ms: float | None = None


class SpeechRequest(BaseModel):
    model: str = "chatterbox"
    input: str
    voice: str = "default"
    response_format: str = "wav"
    speed: float = 1.0


def model() -> ChatterboxTurboTTS:
    global _model, _load_ms
    with _lock:
        if _model is None:
            started = time.perf_counter()
            _model = ChatterboxTurboTTS.from_pretrained(device=DEVICE, nano=NANO)
            _load_ms = round((time.perf_counter() - started) * 1000.0, 1)
        return _model


def marker() -> dict:
    if not MARKER.exists():
        return {"configured": False, "revision": 0, "sha256": None}
    try:
        value = json.loads(MARKER.read_text("utf-8"))
        return value if isinstance(value, dict) else {"configured": False, "revision": 0, "sha256": None}
    except Exception:
        return {"configured": False, "revision": 0, "sha256": None, "error": "invalid marker"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clear_conditionals(m: ChatterboxTurboTTS) -> None:
    try:
        m.conds = None
    except Exception:
        pass


def refresh_voice(m: ChatterboxTurboTTS) -> None:
    global _loaded_revision
    state = marker()
    try:
        revision = int(state.get("revision", 0))
    except Exception:
        revision = 0
    if revision == _loaded_revision:
        return

    if not state.get("configured"):
        _clear_conditionals(m)
        _loaded_revision = revision
        return

    expected = str(state.get("sha256") or "").strip().lower()
    if not expected:
        _clear_conditionals(m)
        raise RuntimeError("Voice reference metadata is incomplete. Upload the reference again.")
    if not REFERENCE.exists() or not REFERENCE.is_file():
        _clear_conditionals(m)
        raise RuntimeError("Voice reference file is missing. Upload the reference again.")
    actual = _sha256(REFERENCE).lower()
    if not hmac.compare_digest(actual, expected):
        _clear_conditionals(m)
        raise RuntimeError("Voice reference integrity check failed. Upload the reference again.")

    try:
        m.prepare_conditionals(str(REFERENCE), exaggeration=0.0, norm_loudness=True)
    except Exception as exc:
        _clear_conditionals(m)
        raise RuntimeError("Voice reference could not be prepared by Chatterbox.") from exc
    _loaded_revision = revision


def _status() -> dict:
    state = marker()
    try:
        current_revision = int(state.get("revision", 0) or 0)
    except Exception:
        current_revision = 0
    return {
        "ok": True,
        "device": DEVICE,
        "nano": NANO,
        "loaded": _model is not None,
        "load_ms": _load_ms,
        "voice_revision": current_revision,
        "voice_configured": bool(state.get("configured")),
        "voice_file_present": REFERENCE.exists(),
        "voice_prepared": _model is not None and _loaded_revision == current_revision,
    }


@app.get("/health")
def health():
    return _status()


@app.post("/warmup")
def warmup():
    """Load Chatterbox and prepare the selected voice before dialing anyone."""
    m = model()
    with _lock:
        try:
            refresh_voice(m)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
    return _status()


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [{"id": "chatterbox", "object": "model", "owned_by": "dialforge-local"}],
    }


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    text = req.input.strip()
    if not text:
        raise HTTPException(400, "input is required")
    if len(text) > 1200:
        raise HTTPException(400, "input is too long")
    m = model()
    with _lock:
        try:
            refresh_voice(m)
            wav = m.generate(
                text,
                exaggeration=0.0,
                cfg_weight=0.0,
                temperature=0.8,
                norm_loudness=True,
            )
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except AssertionError as exc:
            raise HTTPException(409, "Choose a voice reference before synthesizing speech") from exc
        buffer = io.BytesIO()
        torchaudio.save(buffer, wav.detach().cpu(), m.sr, format="wav")
        return Response(buffer.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
