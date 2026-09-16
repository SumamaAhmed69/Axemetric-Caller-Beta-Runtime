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
from chatterbox.tts_turbo import ChatterboxTurboTTS, Conditionals

PORT = int(os.getenv("AXEMETRIC_CHATTERBOX_PORT", "8881"))
REQUESTED_DEVICE = os.getenv("AXEMETRIC_CHATTERBOX_DEVICE", "auto").strip().lower()
if REQUESTED_DEVICE == "auto":
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
elif REQUESTED_DEVICE == "mps":
    DEVICE = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
elif REQUESTED_DEVICE == "cuda":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
else:
    DEVICE = "cpu"
NANO = os.getenv("AXEMETRIC_CHATTERBOX_NANO", "1") != "0"
# Bump this whenever the pinned Chatterbox conditional representation changes.
CONDITIONALS_CACHE_VERSION = "5de7a54-nano-v1" if NANO else "5de7a54-turbo-v1"


def data_root() -> Path:
    override = os.getenv("AXEMETRIC_DATA_ROOT")
    if override:
        return Path(override)
    if os.name == "nt":
        return Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "Lineborn"
    if __import__("sys").platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Lineborn"
    return Path.home() / ".local" / "share" / "lineborn"


RUNTIME_DIR = data_root() / "runtime"
MARKER = RUNTIME_DIR / "voice-reference.json"
REFERENCE = data_root() / "voices" / "reference.wav"
CONDITIONALS_CACHE = RUNTIME_DIR / ("voice-conditionals-nano.pt" if NANO else "voice-conditionals-turbo.pt")
CONDITIONALS_META = RUNTIME_DIR / ("voice-conditionals-nano.json" if NANO else "voice-conditionals-turbo.json")
app = FastAPI(title="Lineborn Chatterbox", docs_url=None, redoc_url=None)
_lock = RLock()
_model: ChatterboxTurboTTS | None = None
_loaded_revision = -1
_load_ms: float | None = None
_voice_prepare_ms: float | None = None
_voice_cache_hit = False


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


def _cache_meta() -> dict:
    if not CONDITIONALS_META.exists():
        return {}
    try:
        value = json.loads(CONDITIONALS_META.read_text("utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _invalidate_conditionals_cache() -> None:
    CONDITIONALS_CACHE.unlink(missing_ok=True)
    CONDITIONALS_META.unlink(missing_ok=True)


def _write_conditionals_cache(m: ChatterboxTurboTTS, *, revision: int, voice_sha256: str) -> None:
    conds = getattr(m, "conds", None)
    if conds is None:
        return
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    staged_cache = CONDITIONALS_CACHE.with_suffix(CONDITIONALS_CACHE.suffix + ".tmp")
    staged_meta = CONDITIONALS_META.with_suffix(CONDITIONALS_META.suffix + ".tmp")
    try:
        conds.save(staged_cache)
        staged_cache.replace(CONDITIONALS_CACHE)
        payload = {
            "schema": 1,
            "cache_version": CONDITIONALS_CACHE_VERSION,
            "voice_sha256": voice_sha256,
            "voice_revision": revision,
            "nano": NANO,
            "created_at": time.time(),
        }
        staged_meta.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        staged_meta.replace(CONDITIONALS_META)
    finally:
        staged_cache.unlink(missing_ok=True)
        staged_meta.unlink(missing_ok=True)


def _load_conditionals_cache(m: ChatterboxTurboTTS, *, revision: int, voice_sha256: str) -> bool:
    meta = _cache_meta()
    valid = bool(
        CONDITIONALS_CACHE.exists()
        and meta.get("schema") == 1
        and meta.get("cache_version") == CONDITIONALS_CACHE_VERSION
        and int(meta.get("voice_revision", -1)) == revision
        and bool(meta.get("nano")) == NANO
        and hmac.compare_digest(str(meta.get("voice_sha256") or "").lower(), voice_sha256.lower())
    )
    if not valid:
        _invalidate_conditionals_cache()
        return False
    try:
        # The pinned Chatterbox Conditionals.load uses torch.load(weights_only=True),
        # then we explicitly move only the derived tensors to the active device.
        m.conds = Conditionals.load(CONDITIONALS_CACHE, map_location="cpu").to(DEVICE)
        return True
    except Exception:
        _clear_conditionals(m)
        _invalidate_conditionals_cache()
        return False


def refresh_voice(m: ChatterboxTurboTTS) -> None:
    global _loaded_revision, _voice_prepare_ms, _voice_cache_hit
    state = marker()
    try:
        revision = int(state.get("revision", 0))
    except Exception:
        revision = 0
    if revision == _loaded_revision:
        return

    if not state.get("configured"):
        _clear_conditionals(m)
        _invalidate_conditionals_cache()
        _voice_prepare_ms = 0.0
        _voice_cache_hit = False
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
        _invalidate_conditionals_cache()
        raise RuntimeError("Voice reference integrity check failed. Upload the reference again.")

    started = time.perf_counter()
    if _load_conditionals_cache(m, revision=revision, voice_sha256=expected):
        _voice_prepare_ms = round((time.perf_counter() - started) * 1000.0, 1)
        _voice_cache_hit = True
        _loaded_revision = revision
        return

    try:
        m.prepare_conditionals(str(REFERENCE), exaggeration=0.0, norm_loudness=True)
        _write_conditionals_cache(m, revision=revision, voice_sha256=expected)
    except Exception as exc:
        _clear_conditionals(m)
        _invalidate_conditionals_cache()
        raise RuntimeError("Voice reference could not be prepared by Chatterbox.") from exc
    _voice_prepare_ms = round((time.perf_counter() - started) * 1000.0, 1)
    _voice_cache_hit = False
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
        "voice_prepare_ms": _voice_prepare_ms,
        "voice_cache_hit": _voice_cache_hit,
        "voice_cache_present": CONDITIONALS_CACHE.exists(),
    }


@app.get("/health")
def health():
    return _status()


@app.post("/warmup")
def warmup():
    """Load Chatterbox and prepare or restore the selected voice before dialing."""
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
        "data": [{"id": "chatterbox", "object": "model", "owned_by": "lineborn-local"}],
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
