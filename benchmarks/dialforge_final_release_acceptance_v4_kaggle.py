#!/usr/bin/env python3
"""Kaggle-safe entry point for Dialforge Final Release Acceptance v4.

Preserves the frozen v4 corpus and scoring. The only override is voice-reference
preparation: Chatterbox Turbo requires a conditioning prompt longer than five
seconds, so the locally generated reference is extended deterministically when
needed instead of relying on an external sample URL.
"""
from __future__ import annotations

import math
import time

import torch
from faster_whisper import WhisperModel

import dialforge_final_release_acceptance_v4 as gate


def prepare_voice(work):
    voice_device = "cuda" if torch.cuda.is_available() else "cpu"
    load_start = time.perf_counter()
    tts = gate.ChatterboxTurboTTS.from_pretrained(device=voice_device, nano=True)
    tts_load = (time.perf_counter() - load_start) * 1000

    prospect_paths = []
    for i, text in enumerate(gate.VOICE_CASES):
        wav = tts.generate(
            text,
            exaggeration=0.0,
            cfg_weight=0.0,
            temperature=0.8,
            norm_loudness=True,
        )
        path = work / f"prospect_{i}.wav"
        gate.save_wav(path, wav, tts.sr)
        prospect_paths.append(path)

    # Generate a local conditioning sample without changing any benchmark case.
    # Chatterbox Turbo asserts that the prompt must be >5.0 s. Extend the audio
    # deterministically if model pacing happens to produce a shorter sample.
    reference_text = (
        "Thanks for taking the call. I am speaking clearly and naturally so this "
        "local voice reference contains enough speech for stable conditioning."
    )
    reference_wav = tts.generate(
        reference_text,
        exaggeration=0.0,
        cfg_weight=0.0,
        temperature=0.8,
        norm_loudness=True,
    )
    duration = reference_wav.shape[-1] / float(tts.sr)
    if duration <= 5.25:
        repeats = max(2, math.ceil(5.5 / max(duration, 0.001)))
        reference_wav = torch.cat([reference_wav] * repeats, dim=-1)

    reference = work / "voice_conditioning_reference.wav"
    gate.save_wav(reference, reference_wav, tts.sr)
    reference_seconds = gate.wav_duration(reference)
    if reference_seconds <= 5.0:
        raise RuntimeError(
            f"Generated voice reference is still too short: {reference_seconds:.3f}s"
        )

    prep = time.perf_counter()
    tts.prepare_conditionals(
        str(reference), exaggeration=0.0, norm_loudness=True
    )
    voice_prep = (time.perf_counter() - prep) * 1000

    _ = tts.generate(
        "Thanks for taking the call.",
        exaggeration=0.0,
        cfg_weight=0.0,
        temperature=0.8,
        norm_loudness=True,
    )

    stt_start = time.perf_counter()
    stt = WhisperModel(
        "small.en",
        device="cuda" if torch.cuda.is_available() else "cpu",
        compute_type="int8",
    )
    stt_load = (time.perf_counter() - stt_start) * 1000
    seg, _ = stt.transcribe(
        str(prospect_paths[0]), language="en", vad_filter=True, beam_size=1
    )
    _ = " ".join(s.text for s in seg)

    return tts, stt, prospect_paths, {
        "tts_load_ms": round(tts_load, 1),
        "voice_prepare_ms": round(voice_prep, 1),
        "stt_load_ms": round(stt_load, 1),
        "device": voice_device,
        "reference_seconds": round(reference_seconds, 3),
    }


gate.prepare_voice = prepare_voice

if __name__ == "__main__":
    raise SystemExit(gate.main())
