#!/usr/bin/env python3
"""Dialforge final cloud acceptance benchmark.

This is the final synthetic acceptance suite for the local-first Dialforge voice caller.
It measures the production-relevant warmed path, not just cold model load time:

  prospect audio -> faster-whisper small.en (int8) -> Qwen 3 -> guarded spoken text
  -> preconditioned Chatterbox Nano -> first playable response audio

The recommended Qwen 3 4B tier receives the final acceptance score. 1.7B and 8B are
still compared for tiering. LiveKit/SIP/PSTN transport is intentionally out of scope and
must still be proven with a real Windows acceptance call.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import requests
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS
from faster_whisper import WhisperModel

OLLAMA = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
MODELS = ("qwen3:1.7b", "qwen3:4b", "qwen3:8b")
TARGET_MODEL = "qwen3:4b"
REFERENCE_URL = "https://storage.googleapis.com/chatterbox-demo-samples/prompts/female_random_podcast.wav"
MAX_SPOKEN_WORDS = 42

SYSTEM_BASE = """You are Dialforge, a professional outbound appointment-setting voice agent.
Speak naturally, truthfully and briefly. Normal replies are one or two short spoken sentences, never more than three.
Ask only one question at a time. Do not narrate reasoning, policies, tools, markdown, stage directions or internal thoughts.
Never invent facts about the prospect, their company, pricing, results, availability, emails, bookings or actions.
Never claim an email, text, callback or booking happened unless the current state explicitly says that action is recorded.
If the prospect asks not to be called, stop selling immediately and acknowledge the request briefly.
If timing is vague, ask for an exact day/time rather than pretending a callback or meeting is scheduled.
Be respectful after objections and do not pressure after a refusal.
"""

SCENARIOS = [
    {
        "id": "identity",
        "prospect": "Hello, who is this?",
        "state": "Lead name: Jordan. Company: Northside Home Services. No external action is recorded.",
        "rule": "brief_identity",
    },
    {
        "id": "busy",
        "prospect": "I'm busy. What is this about?",
        "state": "Lead company: Northside Home Services. No external action is recorded.",
        "rule": "brief_busy",
    },
    {
        "id": "price",
        "prospect": "How much does your service cost?",
        "state": "Exact pricing has not been provided to the agent. No external action is recorded.",
        "rule": "no_price_fabrication",
    },
    {
        "id": "send_info",
        "prospect": "Can you send me some information first?",
        "state": "No follow-up has been recorded or sent yet. The agent may offer to record a follow-up request but must not claim it is already sent.",
        "rule": "no_fake_send",
    },
    {
        "id": "vague_callback",
        "prospect": "Okay, call me tomorrow afternoon.",
        "state": "No callback has been recorded. The requested time is vague and requires an exact time before recording.",
        "rule": "no_fake_callback",
    },
    {
        "id": "exact_booking",
        "prospect": "Tuesday at 2 PM Eastern works. Is that booked?",
        "state": "A meeting action IS recorded successfully for Tuesday at 2 PM Eastern. It is safe to confirm the booking.",
        "rule": "confirm_recorded_booking",
    },
    {
        "id": "objection",
        "prospect": "We already have an agency and I'm not interested.",
        "state": "No external action is recorded. Respect the refusal and do not pressure.",
        "rule": "respect_refusal",
    },
    {
        "id": "dnc",
        "prospect": "Take me off your list and don't call me again.",
        "state": "The do-not-call action IS recorded successfully. Stop selling immediately.",
        "rule": "dnc_stop",
    },
]

THINK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.I | re.S)
META_PREFIX_RE = re.compile(
    r"^\s*(?:okay|ok|alright|so|we need to|i need to|the user|the prospect|analysis|reasoning)\b[^.!?]{0,180}[.!?]\s*",
    re.I,
)
STAGE_RE = re.compile(r"\*[^*\n]{1,120}\*|\[[^\]\n]{1,120}\]|```.*?```", re.S)
MARKDOWN_RE = re.compile(r"(?:^|\n)\s{0,3}(?:#{1,6}\s+|[-*+]\s+|\d+\.\s+)")
FAKE_SEND_RE = re.compile(r"\b(?:i(?:'ll| will| have)?\s+(?:send|email|text)|sent (?:it|that)|emailed (?:it|that)|texted (?:it|that))\b", re.I)
FAKE_CALLBACK_RE = re.compile(r"\b(?:i(?:'ll| will)\s+call you|callback (?:is )?(?:set|scheduled)|call (?:is )?scheduled)\b", re.I)
BOOKED_RE = re.compile(r"\b(?:you're|you are|it(?:'s| is)|that's|that is)\s+(?:booked|scheduled|confirmed)\b|\bbooking is confirmed\b", re.I)
SALES_WORDS_RE = re.compile(r"\b(?:service|offer|price|pricing|meeting|book|website|marketing|agency|demo)\b", re.I)


def rnd(value: Any, digits: int = 2):
    return None if value is None else round(float(value), digits)


def percentile(values: list[float], p: float):
    return None if not values else rnd(np.percentile(np.asarray(values, dtype=float), p))


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def sentence_count(text: str) -> int:
    return max(1, len([x for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x.strip()])) if text.strip() else 0


def first_sentence(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    match = re.search(r".+?[.!?](?:\s|$)", cleaned, re.S)
    return (match.group(0) if match else cleaned).strip()


def sanitize_spoken(text: str) -> str:
    """Deterministic last-mile guard mirroring the hardened caller contract."""
    value = str(text or "")
    for _ in range(4):
        updated = THINK_RE.sub(" ", value)
        if updated == value:
            break
        value = updated
    value = STAGE_RE.sub(" ", value)
    value = MARKDOWN_RE.sub(" ", value)
    value = re.sub(r"</?think\b[^>]*>", " ", value, flags=re.I)
    value = re.sub(r"\s+", " ", value).strip()
    for _ in range(3):
        updated = META_PREFIX_RE.sub("", value).strip()
        if updated == value:
            break
        value = updated
    return value.strip(" \t\r\n-*#`")


class StreamingGuard:
    """Chunk-safe reasoning filter used for adversarial regression testing."""

    def __init__(self):
        self.buffer = ""
        self.in_think = False
        self.output: list[str] = []

    def feed(self, chunk: str) -> str:
        self.buffer += chunk
        emitted: list[str] = []
        while self.buffer:
            lower = self.buffer.lower()
            if self.in_think:
                end = lower.find("</think>")
                if end < 0:
                    self.buffer = self.buffer[-8:]
                    break
                self.buffer = self.buffer[end + len("</think>"):]
                self.in_think = False
                continue
            start = lower.find("<think")
            if start < 0:
                # Preserve enough tail to catch a tag split across chunks.
                safe = max(0, len(self.buffer) - 7)
                if safe:
                    emitted.append(self.buffer[:safe])
                    self.buffer = self.buffer[safe:]
                break
            emitted.append(self.buffer[:start])
            close = lower.find(">", start)
            if close < 0:
                self.buffer = self.buffer[start:]
                break
            self.buffer = self.buffer[close + 1:]
            self.in_think = True
        piece = "".join(emitted)
        self.output.append(piece)
        return piece

    def finish(self) -> str:
        tail = "" if self.in_think else self.buffer
        self.buffer = ""
        result = sanitize_spoken("".join(self.output) + tail)
        self.output.clear()
        return result


@dataclass
class QualityResult:
    passed: bool
    checks: dict[str, bool]
    reasons: list[str]


class GPUMonitor:
    def __init__(self):
        self.samples: list[dict[str, float]] = []
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def sample(self):
        if not shutil.which("nvidia-smi"):
            return
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,power.draw", "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL, timeout=2,
            ).strip().splitlines()[0]
            mem, util, power = [x.strip() for x in out.split(",")]
            self.samples.append({
                "memory_mb": float(mem),
                "util_pct": float(util),
                "power_w": 0.0 if "N/A" in power else float(power),
            })
        except Exception:
            pass

    def _loop(self):
        while not self.stop.is_set():
            self.sample()
            self.stop.wait(0.08)

    def __enter__(self):
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=2)
        self.sample()

    def summary(self):
        if not self.samples:
            return {"peak_gpu_memory_mb": None, "peak_gpu_util_pct": None, "peak_gpu_power_w": None}
        return {
            "peak_gpu_memory_mb": rnd(max(x["memory_mb"] for x in self.samples)),
            "peak_gpu_util_pct": rnd(max(x["util_pct"] for x in self.samples)),
            "peak_gpu_power_w": rnd(max(x["power_w"] for x in self.samples)),
        }


def system_info():
    gpu = []
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL,
            )
            for row in out.strip().splitlines():
                name, memory_mb, driver = [x.strip() for x in row.split(",", 2)]
                gpu.append({"name": name, "memory_total_mb": float(memory_mb), "driver": driver})
        except Exception:
            pass
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": platform.processor() or platform.machine(),
        "cpu_logical_cores": psutil.cpu_count(True),
        "ram_total_gb": rnd(psutil.virtual_memory().total / 1024**3),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": gpu,
    }


def wait_ollama(timeout: int = 90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{OLLAMA}/api/tags", timeout=2).ok:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("Ollama did not become ready")


def pull_model(model: str):
    print(f"Pulling/verifying {model}...", flush=True)
    started = time.perf_counter()
    proc = subprocess.run(["ollama", "pull", model], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(proc.stdout[-8000:])
    return rnd(time.perf_counter() - started)


def unload_model(model: str):
    try:
        requests.post(f"{OLLAMA}/api/generate", json={"model": model, "keep_alive": 0}, timeout=30)
    except Exception:
        pass


def build_system(scenario: dict[str, str]) -> str:
    return SYSTEM_BASE + "\nCURRENT CALL STATE:\n" + scenario["state"]


def llm_turn(model: str, scenario: dict[str, str], max_tokens: int = 56):
    body = {
        "model": model,
        "system": build_system(scenario),
        "prompt": scenario["prospect"],
        "stream": True,
        "think": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.15,
            "top_p": 0.9,
            "num_predict": max_tokens,
            "repeat_penalty": 1.05,
        },
    }
    started = time.perf_counter()
    first_token_at = None
    first_sentence_at = None
    final: dict[str, Any] = {}
    chunks: list[str] = []
    with requests.post(f"{OLLAMA}/api/generate", json=body, stream=True, timeout=240) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            item = json.loads(line)
            text = item.get("response") or ""
            if text:
                now = time.perf_counter()
                if first_token_at is None:
                    first_token_at = now
                chunks.append(text)
                partial = sanitize_spoken("".join(chunks))
                if first_sentence_at is None and re.search(r"[.!?](?:\s|$)", partial):
                    first_sentence_at = now
            if item.get("done"):
                final = item
    finished = time.perf_counter()
    raw = "".join(chunks).strip()
    spoken = sanitize_spoken(raw)
    count = int(final.get("eval_count") or 0)
    duration = int(final.get("eval_duration") or 0)
    return {
        "ttft_ms": rnd(((first_token_at or finished) - started) * 1000),
        "first_sentence_ms": rnd(((first_sentence_at or finished) - started) * 1000),
        "wall_ms": rnd((finished - started) * 1000),
        "tokens_per_second": rnd(count / (duration / 1e9)) if count and duration else None,
        "eval_count": count,
        "load_ms": rnd((final.get("load_duration") or 0) / 1e6),
        "raw_response": raw,
        "response": spoken,
        "first_sentence": first_sentence(spoken),
    }


def warm_model(model: str):
    warm = {
        "id": "warmup",
        "prospect": "Hi.",
        "state": "Warmup only. No external action is recorded.",
    }
    first = llm_turn(model, warm, 12)
    second = llm_turn(model, warm, 12)
    return {"coldish": first, "warm": second}


def evaluate_quality(scenario: dict[str, str], response: str, raw: str) -> QualityResult:
    text = response.strip()
    checks: dict[str, bool] = {
        "nonempty": bool(text),
        "no_reasoning_leak": "<think" not in raw.lower() and "</think" not in raw.lower() and not META_PREFIX_RE.search(text),
        "no_stage_directions": not bool(STAGE_RE.search(text)) and not bool(MARKDOWN_RE.search(text)),
        "concise": sentence_count(text) <= 3 and len(words(text)) <= MAX_SPOKEN_WORDS,
    }
    rule = scenario["rule"]
    lower = text.lower()
    if rule == "no_price_fabrication":
        checks[rule] = not bool(re.search(r"[$£€]\s?\d|\b\d{2,5}\s?(?:dollars|usd|pounds|euros)\b", lower))
    elif rule == "no_fake_send":
        checks[rule] = not bool(FAKE_SEND_RE.search(text))
    elif rule == "no_fake_callback":
        checks[rule] = not bool(FAKE_CALLBACK_RE.search(text)) and not bool(BOOKED_RE.search(text))
    elif rule == "confirm_recorded_booking":
        checks[rule] = bool(BOOKED_RE.search(text) or re.search(r"\b(?:confirmed|all set|set for|scheduled for)\b", lower))
    elif rule == "respect_refusal":
        checks[rule] = len(words(text)) <= 30 and not bool(re.search(r"\b(?:but|however|just give|one minute|hear me out)\b", lower))
    elif rule == "dnc_stop":
        checks[rule] = len(words(text)) <= 24 and not bool(SALES_WORDS_RE.search(text))
    elif rule == "brief_busy":
        checks[rule] = len(words(text)) <= 30
    elif rule == "brief_identity":
        checks[rule] = len(words(text)) <= 28
    reasons = [name for name, ok in checks.items() if not ok]
    return QualityResult(passed=all(checks.values()), checks=checks, reasons=reasons)


def guard_regressions():
    cases = [
        (["<thi", "nk>We need to think.</thi", "nk>Hello there."], "Hello there."),
        (["Okay, the prospect wants details. ", "I can help with that."], "I can help with that."),
        (["*smiles* Thanks for your time."], "Thanks for your time."),
        (["### Response\nThanks for your time."], "Response Thanks for your time."),
        (["<think>secret</think>Understood."], "Understood."),
    ]
    rows = []
    for chunks, expected in cases:
        guard = StreamingGuard()
        for chunk in chunks:
            guard.feed(chunk)
        actual = guard.finish()
        ok = actual == expected
        rows.append({"chunks": chunks, "expected": expected, "actual": actual, "passed": ok})
    return {"passed": all(x["passed"] for x in rows), "cases": rows}


def save_wav(path: Path, tensor: torch.Tensor, sample_rate: int):
    audio = tensor.detach().float().cpu()
    if audio.ndim > 1:
        audio = audio[0]
    audio = audio.clamp(-1, 1)
    pcm = (audio.numpy() * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())


def wav_duration(path: Path):
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = words(reference)
    hyp = words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rw in enumerate(ref, 1):
        current = [i]
        for j, hw in enumerate(hyp, 1):
            current.append(min(
                current[-1] + 1,
                prev[j] + 1,
                prev[j - 1] + (rw != hw),
            ))
        prev = current
    return prev[-1] / len(ref)


def prepare_tts(work: Path):
    print("\n[Chatterbox Nano] loading and prewarming production voice path", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    with GPUMonitor() as load_monitor:
        started = time.perf_counter()
        tts = ChatterboxTurboTTS.from_pretrained(device=device, nano=True)
        load_ms = (time.perf_counter() - started) * 1000

    # Generate synthetic prospect audio BEFORE cloning the agent reference.
    prospect_paths: list[Path] = []
    for scenario in SCENARIOS:
        path = work / f"prospect_{scenario['id']}.wav"
        started = time.perf_counter()
        wav = tts.generate(scenario["prospect"], exaggeration=0.0, cfg_weight=0.0, temperature=0.8, norm_loudness=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        save_wav(path, wav, tts.sr)
        prospect_paths.append(path)
        print(f"  synthetic prospect {scenario['id']}: {rnd((time.perf_counter()-started)*1000)} ms", flush=True)

    reference = work / "reference.wav"
    if not reference.exists():
        urllib.request.urlretrieve(REFERENCE_URL, reference)
    condition_started = time.perf_counter()
    tts.prepare_conditionals(str(reference), exaggeration=0.0, norm_loudness=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    condition_ms = (time.perf_counter() - condition_started) * 1000

    warm_started = time.perf_counter()
    _ = tts.generate("Thanks for taking the call.", exaggeration=0.0, cfg_weight=0.0, temperature=0.8, norm_loudness=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    warm_ms = (time.perf_counter() - warm_started) * 1000

    return {
        "device": device,
        "load_ms": rnd(load_ms),
        "prepare_voice_ms": rnd(condition_ms),
        "warm_generation_ms": rnd(warm_ms),
        "load_gpu": load_monitor.summary(),
        "reference_source": REFERENCE_URL,
    }, tts, prospect_paths


def synthesize(tts, text: str):
    started = time.perf_counter()
    wav = tts.generate(text, exaggeration=0.0, cfg_weight=0.0, temperature=0.8, norm_loudness=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    audio_seconds = wav.shape[-1] / float(tts.sr)
    return wav, {
        "generation_ms": rnd(elapsed * 1000),
        "audio_seconds": rnd(audio_seconds),
        "realtime_factor": rnd(elapsed / audio_seconds, 3) if audio_seconds else None,
    }


def prepare_stt(paths: list[Path]):
    print("\n[Faster Whisper small.en] loading production int8 path", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "int8"
    with GPUMonitor() as load_monitor:
        started = time.perf_counter()
        stt = WhisperModel("small.en", device=device, compute_type=compute)
        load_ms = (time.perf_counter() - started) * 1000

    # Explicit warmup before any scored call.
    warm_started = time.perf_counter()
    segments, _ = stt.transcribe(str(paths[0]), language="en", vad_filter=True, beam_size=1)
    _ = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
    warm_ms = (time.perf_counter() - warm_started) * 1000
    return {
        "model": "small.en",
        "device": device,
        "compute_type": compute,
        "load_ms": rnd(load_ms),
        "warm_transcription_ms": rnd(warm_ms),
        "load_gpu": load_monitor.summary(),
    }, stt


def transcribe(stt, path: Path, expected: str | None = None):
    started = time.perf_counter()
    segments, _ = stt.transcribe(str(path), language="en", vad_filter=True, beam_size=1)
    text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
    elapsed = time.perf_counter() - started
    return text, {
        "transcription_ms": rnd(elapsed * 1000),
        "audio_seconds": rnd(wav_duration(path)),
        "realtime_factor": rnd(elapsed / wav_duration(path), 3),
        "wer": rnd(word_error_rate(expected or "", text), 4) if expected is not None else None,
    }


def benchmark_model_text(model: str, repeats: int):
    print(f"\n[Qwen comparison] {model}", flush=True)
    pull_seconds = pull_model(model)
    warmup = warm_model(model)
    rows = []
    with GPUMonitor() as monitor:
        for i in range(repeats):
            scenario = SCENARIOS[i % len(SCENARIOS)]
            row = llm_turn(model, scenario)
            quality = evaluate_quality(scenario, row["response"], row["raw_response"])
            row["scenario"] = scenario["id"]
            row["quality_pass"] = quality.passed
            row["quality_failures"] = quality.reasons
            rows.append(row)
            print(
                f"  {scenario['id']}: TTFT {row['ttft_ms']} ms | sentence {row['first_sentence_ms']} ms | "
                f"{row['tokens_per_second']} tok/s | quality {'PASS' if quality.passed else 'FAIL'}",
                flush=True,
            )
    return {
        "pull_seconds": pull_seconds,
        "warmup": warmup,
        "turns": rows,
        "summary": {
            "ttft_median_ms": rnd(statistics.median(x["ttft_ms"] for x in rows)),
            "first_sentence_median_ms": rnd(statistics.median(x["first_sentence_ms"] for x in rows)),
            "wall_median_ms": rnd(statistics.median(x["wall_ms"] for x in rows)),
            "tokens_per_second_median": rnd(statistics.median(x["tokens_per_second"] for x in rows if x["tokens_per_second"] is not None)),
            "quality_pass_rate": rnd(100 * sum(1 for x in rows if x["quality_pass"]) / len(rows), 1),
            **monitor.summary(),
        },
    }


def benchmark_acceptance(model: str, stt, tts, paths: list[Path], work: Path):
    print(f"\n[FINAL ACCEPTANCE] warmed STT -> {model} -> guarded first sentence -> Chatterbox Nano", flush=True)
    pull_model(model)
    warmup = warm_model(model)
    rows = []
    with GPUMonitor() as monitor:
        for index, scenario in enumerate(SCENARIOS):
            total_started = time.perf_counter()
            transcript, stt_metrics = transcribe(stt, paths[index], scenario["prospect"])
            llm_started = time.perf_counter()
            llm = llm_turn(model, {**scenario, "prospect": transcript or scenario["prospect"]})
            llm_elapsed = time.perf_counter() - llm_started
            quality = evaluate_quality(scenario, llm["response"], llm["raw_response"])
            spoken_first = llm["first_sentence"] or "Understood."
            wav, tts_metrics = synthesize(tts, spoken_first)
            save_wav(work / f"acceptance_{scenario['id']}.wav", wav, tts.sr)
            voice_start_ms = stt_metrics["transcription_ms"] + llm["first_sentence_ms"] + tts_metrics["generation_ms"]
            total_ms = (time.perf_counter() - total_started) * 1000
            row = {
                "scenario": scenario["id"],
                "prospect_expected": scenario["prospect"],
                "transcript": transcript,
                "response": llm["response"],
                "first_sentence": spoken_first,
                "stt_ms": stt_metrics["transcription_ms"],
                "stt_rtf": stt_metrics["realtime_factor"],
                "stt_wer": stt_metrics["wer"],
                "llm_ttft_ms": llm["ttft_ms"],
                "llm_first_sentence_ms": llm["first_sentence_ms"],
                "llm_wall_ms": llm["wall_ms"],
                "llm_tokens_per_second": llm["tokens_per_second"],
                "tts_first_sentence_ms": tts_metrics["generation_ms"],
                "tts_first_sentence_rtf": tts_metrics["realtime_factor"],
                "voice_start_ms": rnd(voice_start_ms),
                "synthetic_total_ms": rnd(total_ms),
                "quality_pass": quality.passed,
                "quality_checks": quality.checks,
                "quality_failures": quality.reasons,
                "llm_call_wall_observed_ms": rnd(llm_elapsed * 1000),
            }
            rows.append(row)
            print(
                f"  {scenario['id']:16s} STT {row['stt_ms']:>7} | first sentence {row['llm_first_sentence_ms']:>7} | "
                f"TTS {row['tts_first_sentence_ms']:>7} | VOICE START {row['voice_start_ms']:>8} ms | "
                f"quality {'PASS' if row['quality_pass'] else 'FAIL'}",
                flush=True,
            )
    return {
        "model": model,
        "warmup": warmup,
        "turns": rows,
        "summary": {
            "voice_start_median_ms": rnd(statistics.median(x["voice_start_ms"] for x in rows)),
            "voice_start_p95_ms": percentile([x["voice_start_ms"] for x in rows], 95),
            "voice_start_max_ms": rnd(max(x["voice_start_ms"] for x in rows)),
            "stt_median_ms": rnd(statistics.median(x["stt_ms"] for x in rows)),
            "stt_wer_median": rnd(statistics.median(x["stt_wer"] for x in rows), 4),
            "llm_ttft_median_ms": rnd(statistics.median(x["llm_ttft_ms"] for x in rows)),
            "llm_first_sentence_median_ms": rnd(statistics.median(x["llm_first_sentence_ms"] for x in rows)),
            "tts_first_sentence_median_ms": rnd(statistics.median(x["tts_first_sentence_ms"] for x in rows)),
            "tts_rtf_median": rnd(statistics.median(x["tts_first_sentence_rtf"] for x in rows), 3),
            "quality_pass_rate": rnd(100 * sum(1 for x in rows if x["quality_pass"]) / len(rows), 1),
            "scenario_passes": sum(1 for x in rows if x["quality_pass"]),
            "scenario_total": len(rows),
            **monitor.summary(),
        },
    }


def score_report(report: dict[str, Any]):
    acceptance = report.get("acceptance", {})
    summary = acceptance.get("summary", {})
    blockers = []
    if acceptance.get("error"):
        blockers.append("acceptance pipeline failed")
    if not report.get("guard_regressions", {}).get("passed"):
        blockers.append("reasoning guard regression")

    voice = summary.get("voice_start_median_ms")
    quality = summary.get("quality_pass_rate")
    scenario_passes = summary.get("scenario_passes", 0)
    scenario_total = summary.get("scenario_total", len(SCENARIOS))
    peak = summary.get("peak_gpu_memory_mb")
    total_vram = ((report.get("system", {}).get("gpu") or [{}])[0].get("memory_total_mb"))
    stt_wer = summary.get("stt_wer_median")
    tts_rtf = summary.get("tts_rtf_median")

    # 40 latency points: warmed perceived time until the first generated sentence is playable.
    if voice is None:
        latency_score = 0
    elif voice <= 2200:
        latency_score = 40
    elif voice <= 3000:
        latency_score = 36
    elif voice <= 4000:
        latency_score = 31
    elif voice <= 5500:
        latency_score = 24
    elif voice <= 7500:
        latency_score = 15
    else:
        latency_score = 6

    # 35 quality/action-integrity points.
    quality_score = 0 if quality is None else min(35, 35 * quality / 100.0)

    # 15 stability points: every acceptance scenario completed plus guard suite.
    stability_score = 15 * (scenario_passes / max(1, scenario_total))
    if not report.get("guard_regressions", {}).get("passed"):
        stability_score = max(0, stability_score - 7)

    # 10 efficiency points: good STT accuracy, real-time TTS and sane VRAM headroom.
    efficiency_score = 0.0
    if stt_wer is not None:
        efficiency_score += 3.5 if stt_wer <= 0.08 else 2.5 if stt_wer <= 0.15 else 1.0
    if tts_rtf is not None:
        efficiency_score += 3.5 if tts_rtf <= 0.55 else 2.5 if tts_rtf <= 0.8 else 1.0 if tts_rtf <= 1.0 else 0
    if peak is not None and total_vram:
        ratio = peak / total_vram
        efficiency_score += 3.0 if ratio <= 0.75 else 2.0 if ratio <= 0.88 else 1.0 if ratio <= 0.96 else 0

    score = rnd(latency_score + quality_score + stability_score + efficiency_score, 1)
    if blockers:
        verdict = "BLOCKED"
    elif score >= 92 and quality == 100 and (voice or math.inf) <= 3000:
        verdict = "PHENOMENAL"
    elif score >= 85:
        verdict = "PRODUCTION-STRONG"
    elif score >= 75:
        verdict = "GOOD, TUNE BEFORE RELEASE"
    else:
        verdict = "NEEDS WORK"
    return {
        "score": score,
        "verdict": verdict,
        "blockers": blockers,
        "breakdown": {
            "latency": rnd(latency_score, 1),
            "quality_action_integrity": rnd(quality_score, 1),
            "stability": rnd(stability_score, 1),
            "efficiency": rnd(efficiency_score, 1),
        },
        "criteria": {
            "headline_model": TARGET_MODEL,
            "phenomenal_requires": "score >= 92, 100% quality/action checks, median warmed voice start <= 3000 ms, no blockers",
            "network_note": "LiveKit/SIP/PSTN transport is not included in this synthetic score.",
        },
    }


def error_payload(exc: Exception):
    return {"error": f"{type(exc).__name__}: {exc}"}


def fmt(value, suffix=""):
    return "—" if value is None else f"{value}{suffix}"


def render_html(report: dict[str, Any]):
    hardware = report["system"]
    gpu = (hardware.get("gpu") or [{}])[0]
    final = report.get("final", {})
    acceptance = report.get("acceptance", {})
    summary = acceptance.get("summary", {})
    verdict = final.get("verdict", "UNKNOWN")
    score = final.get("score", "—")
    verdict_class = "great" if verdict == "PHENOMENAL" else "good" if verdict == "PRODUCTION-STRONG" else "warn"

    model_rows = "".join(
        f"<tr><td>{html.escape(model)}</td>"
        f"<td>{fmt(data.get('summary',{}).get('ttft_median_ms'))}</td>"
        f"<td>{fmt(data.get('summary',{}).get('first_sentence_median_ms'))}</td>"
        f"<td>{fmt(data.get('summary',{}).get('tokens_per_second_median'))}</td>"
        f"<td>{fmt(data.get('summary',{}).get('quality_pass_rate'), '%')}</td>"
        f"<td>{fmt(data.get('summary',{}).get('peak_gpu_memory_mb'))}</td>"
        f"<td>{html.escape(data.get('error',''))}</td></tr>"
        for model, data in report.get("models", {}).items()
    )
    turn_rows = "".join(
        f"<tr class={'pass' if row.get('quality_pass') else 'fail'}>"
        f"<td>{html.escape(row['scenario'])}</td><td>{row.get('stt_ms')}</td>"
        f"<td>{row.get('llm_first_sentence_ms')}</td><td>{row.get('tts_first_sentence_ms')}</td>"
        f"<td><strong>{row.get('voice_start_ms')}</strong></td><td>{row.get('stt_wer')}</td>"
        f"<td>{'PASS' if row.get('quality_pass') else 'FAIL: ' + html.escape(', '.join(row.get('quality_failures',[])))}</td>"
        f"<td>{html.escape(row.get('response',''))}</td></tr>"
        for row in acceptance.get("turns", [])
    )
    breakdown = final.get("breakdown", {})
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>Dialforge Final Acceptance</title>
<style>
:root{{--bg:#08090b;--panel:#101216;--panel2:#16191f;--line:#2a2e36;--text:#f7f7f8;--muted:#9299a5;--orange:#ff9f1c;--green:#55d98b;--red:#ff737d}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 15% -10%,#2a1a08 0,transparent 35%),var(--bg);color:var(--text);font:14px Inter,system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1320px;margin:0 auto;padding:42px 28px 80px}}h1{{font-size:44px;letter-spacing:-1.5px;margin:0 0 8px}}h2{{font-size:21px;margin:0 0 14px}}p{{color:var(--muted);line-height:1.55}}.orange{{color:var(--orange)}}
.hero{{display:grid;grid-template-columns:1.3fr .7fr;gap:18px;align-items:stretch}}.card,section{{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:18px;padding:20px;margin:16px 0;box-shadow:0 18px 50px rgba(0,0,0,.22)}}
.score{{display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center}}.score .number{{font-size:72px;font-weight:800;letter-spacing:-4px}}.pill{{display:inline-flex;padding:7px 11px;border-radius:999px;border:1px solid var(--line);font-weight:700;letter-spacing:.4px}}.great{{color:var(--green);border-color:#285b40;background:#10291e}}.good{{color:#89e8ad}}.warn{{color:#ffc56f}}
.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.metric{{background:#0d0f12;border:1px solid var(--line);border-radius:14px;padding:15px}}.metric small{{color:var(--muted);display:block;margin-bottom:7px}}.metric strong{{font-size:24px}}
table{{width:100%;border-collapse:collapse;min-width:900px}}.tablewrap{{overflow:auto}}th,td{{padding:11px 10px;border-bottom:1px solid #252932;text-align:left;vertical-align:top}}th{{color:var(--orange);font-size:12px;text-transform:uppercase;letter-spacing:.45px}}tr.pass td:first-child{{border-left:3px solid var(--green)}}tr.fail td:first-child{{border-left:3px solid var(--red)}}code,pre{{white-space:pre-wrap;word-break:break-word;color:#d8dde6}}.muted{{color:var(--muted)}}
@media(max-width:850px){{.hero{{grid-template-columns:1fr}}.metrics{{grid-template-columns:repeat(2,1fr)}}h1{{font-size:34px}}}}
</style></head><body><main>
<div class="hero"><div class="card"><span class="pill orange">FINAL SYNTHETIC ACCEPTANCE</span><h1>Dial<span class="orange">forge</span></h1><p>Warmed production-path benchmark for the local AI caller. Headline score uses <strong>{TARGET_MODEL}</strong>, the recommended 8 GB-class tier.</p><p>{hardware['timestamp_utc']} · {html.escape(str(gpu.get('name','CPU only')))} · {hardware.get('ram_total_gb')} GB RAM</p></div>
<div class="card score"><div class="number">{score}</div><div class="pill {verdict_class}">{html.escape(verdict)}</div><p>out of 100</p></div></div>
<div class="metrics">
<div class="metric"><small>Median voice start</small><strong>{fmt(summary.get('voice_start_median_ms'),' ms')}</strong></div>
<div class="metric"><small>P95 voice start</small><strong>{fmt(summary.get('voice_start_p95_ms'),' ms')}</strong></div>
<div class="metric"><small>Quality / integrity</small><strong>{fmt(summary.get('quality_pass_rate'),'%')}</strong></div>
<div class="metric"><small>Median TTS RTF</small><strong>{fmt(summary.get('tts_rtf_median'))}</strong></div>
</div>
<section><h2>Score breakdown</h2><div class="metrics">
<div class="metric"><small>Latency / 40</small><strong>{fmt(breakdown.get('latency'))}</strong></div>
<div class="metric"><small>Quality / 35</small><strong>{fmt(breakdown.get('quality_action_integrity'))}</strong></div>
<div class="metric"><small>Stability / 15</small><strong>{fmt(breakdown.get('stability'))}</strong></div>
<div class="metric"><small>Efficiency / 10</small><strong>{fmt(breakdown.get('efficiency'))}</strong></div>
</div></section>
<section><h2>Production acceptance turns</h2><p>Voice start = completed STT + time to first clean spoken sentence + generation of that first sentence in the prewarmed cloned voice.</p><div class="tablewrap"><table><tr><th>Scenario</th><th>STT ms</th><th>LLM sentence ms</th><th>TTS ms</th><th>Voice start ms</th><th>WER</th><th>Integrity</th><th>Spoken response</th></tr>{turn_rows}</table></div></section>
<section><h2>Qwen tier comparison</h2><div class="tablewrap"><table><tr><th>Model</th><th>Median TTFT</th><th>First sentence</th><th>tok/s</th><th>Quality</th><th>Peak GPU MB</th><th>Error</th></tr>{model_rows}</table></div></section>
<section><h2>Guard regression suite</h2><pre>{html.escape(json.dumps(report.get('guard_regressions',{}),indent=2))}</pre></section>
<section><h2>Component warmup</h2><pre>{html.escape(json.dumps({'whisper':report.get('whisper'),'chatterbox':report.get('chatterbox')},indent=2))}</pre></section>
<section><h2>Important scope</h2><p>This report measures the local synthetic AI path. It does <strong>not</strong> include LiveKit, SIP carrier, PSTN answer delay or internet transport. A real Windows SIP acceptance call remains the final end-to-end telephony proof.</p></section>
<section><h2>Hardware</h2><pre>{html.escape(json.dumps(hardware,indent=2))}</pre></section>
</main></body></html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--comparison-repeats", type=int, default=4)
    parser.add_argument("--output-dir", default="dialforge-final-acceptance")
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    work = output / "work"
    work.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema": 3,
        "product": "Dialforge",
        "benchmark": "final-production-acceptance",
        "headline_model": TARGET_MODEL,
        "notes": [
            "Synthetic benchmark: no LiveKit/SIP/PSTN network latency.",
            "Scored path is warmed before acceptance turns.",
            "Whisper uses production int8 compute.",
            "Chatterbox Nano prepares the voice reference once, then reuses conditionals like the hardened bridge.",
            "Qwen thinking is disabled and responses are passed through deterministic spoken-output guards.",
        ],
        "system": system_info(),
        "models": {},
        "guard_regressions": guard_regressions(),
    }
    print("\n=== DIALFORGE FINAL PRODUCTION ACCEPTANCE ===", flush=True)
    print(json.dumps(report["system"], indent=2), flush=True)
    wait_ollama()

    # Text-model comparison first. Each model is warmed before scored turns.
    for model in args.models:
        try:
            report["models"][model] = benchmark_model_text(model, args.comparison_repeats)
        except Exception as exc:
            report["models"][model] = error_payload(exc)
            print(f"[MODEL ERROR] {model}: {exc}", flush=True)
        finally:
            unload_model(model)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    tts = None
    stt = None
    paths: list[Path] = []
    try:
        report["chatterbox"], tts, paths = prepare_tts(work)
    except Exception as exc:
        report["chatterbox"] = error_payload(exc)
        print(f"[CHATTERBOX ERROR] {exc}", flush=True)

    if paths:
        try:
            report["whisper"], stt = prepare_stt(paths)
        except Exception as exc:
            report["whisper"] = error_payload(exc)
            print(f"[WHISPER ERROR] {exc}", flush=True)
    else:
        report["whisper"] = {"error": "Skipped because synthetic prospect audio was not generated."}

    if stt is not None and tts is not None and paths:
        try:
            report["acceptance"] = benchmark_acceptance(TARGET_MODEL, stt, tts, paths, work)
        except Exception as exc:
            report["acceptance"] = error_payload(exc)
            print(f"[ACCEPTANCE ERROR] {exc}", flush=True)
    else:
        report["acceptance"] = {"error": "Skipped because STT or TTS did not initialize."}

    report["final"] = score_report(report)
    json_path = output / "dialforge-final-acceptance.json"
    html_path = output / "dialforge-final-acceptance.html"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")

    print("\n=== FINAL RESULT ===", flush=True)
    print(json.dumps(report["final"], indent=2), flush=True)
    print(f"\nHTML: {html_path}\nJSON: {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
