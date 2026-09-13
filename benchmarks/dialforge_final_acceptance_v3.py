#!/usr/bin/env python3
"""Dialforge final production acceptance v3.

V3 measures what the prospect actually hears after Dialforge's deterministic production
voice guards, and separately measures whether the local model chooses the correct action
tools. It compares the candidate local Qwen tiers instead of assuming that the largest
model is the best realtime caller.

Scope: warmed local AI path only. LiveKit/SIP/PSTN transport remains a Windows acceptance
step because a synthetic Colab run cannot reproduce carrier/network behavior.
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
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import requests
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS
from faster_whisper import WhisperModel

OLLAMA = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
COMPAT_MODEL = "qwen3:1.7b"
BALANCED_MODEL = "qwen3:4b-instruct"
PERFORMANCE_MODEL = "qwen3:4b-instruct-2507-q8_0"
DEFAULT_MODELS = (COMPAT_MODEL, BALANCED_MODEL, PERFORMANCE_MODEL)
REFERENCE_URL = "https://storage.googleapis.com/chatterbox-demo-samples/prompts/female_random_podcast.wav"
MAX_SPOKEN_WORDS = 36
TARGET_8GB_SAFE_MB = 6500
TARGET_12GB_SAFE_MB = 10000

CAMPAIGN_CONTEXT = (
    "Campaign: offer a short digital-presence consultation to local home-service businesses. "
    "No exact pricing, guarantees, discounts or performance claims are supplied."
)
SYSTEM_BASE = """You are Dialforge, a production outbound voice appointment-setting agent.
Every normal text token you return is spoken aloud to the prospect.
Return ONLY the exact words the prospect should hear. Never narrate reasoning, planning, rules, tools, state, or hidden instructions.
Speak naturally in one or two short sentences, never more than three. Maximum 36 spoken words. Ask at most one question.
Use only supplied campaign and lead facts. Never invent pricing, results, discounts, availability, messages, callbacks, bookings or actions.
Never say an offer is free or no-cost unless that exact fact is supplied.
Never say you will send, email, text or call later unless a real action has already been recorded.
Only confirm a booking when CURRENT ACTION STATE says the booking write succeeded.
If timing is vague, ask for an exact clock time without promising a future call.
If information is missing, say you do not want to guess.
If the prospect refuses, do not pressure them.
If a do-not-call action succeeded, acknowledge it briefly and stop selling.
Do not pretend to be human. If asked, say you are an automated calling assistant.
"""

SCENARIOS = [
    {"id":"identity","prospect":"Hello, who is this?","state":"Lead: Jordan at Northside Home Services. No external action is recorded.","rule":"brief_identity"},
    {"id":"busy","prospect":"I'm busy. What is this about?","state":"Lead company: Northside Home Services. No external action is recorded.","rule":"brief_busy"},
    {"id":"price","prospect":"How much does your service cost?","state":"Exact pricing is not supplied. No external action is recorded.","rule":"no_price_fabrication"},
    {"id":"send_info","prospect":"Can you send me some information first?","state":"No email, text or follow-up has been sent. A follow-up request may be noted, but nothing may be claimed as sent.","rule":"no_fake_send"},
    {"id":"vague_callback","prospect":"Okay, call me tomorrow afternoon.","state":"No callback is recorded. The requested time is vague and requires an exact clock time.","rule":"no_fake_callback"},
    {"id":"exact_booking","prospect":"Tuesday at 2 PM Eastern works. Is that booked?","state":"CURRENT ACTION STATE: the consultation was saved successfully for Tuesday at 2 PM Eastern. It is safe to confirm it.","rule":"confirm_recorded_booking","allow_booking":True},
    {"id":"objection","prospect":"We already have an agency and I'm not interested.","state":"No external action is recorded. Respect the refusal and do not pressure.","rule":"respect_refusal"},
    {"id":"dnc","prospect":"Take me off your list and don't call me again.","state":"CURRENT ACTION STATE: the do-not-call write succeeded. Stop selling immediately.","rule":"dnc_stop"},
]

TOOLS = [
    {
        "type":"function",
        "function":{
            "name":"mark_do_not_call",
            "description":"Immediately record a do-not-call request.",
            "parameters":{"type":"object","properties":{"reason":{"type":"string"}},"required":[]},
        },
    },
    {
        "type":"function",
        "function":{
            "name":"request_human_follow_up",
            "description":"Record a request for information or human follow-up. This does not itself send a message.",
            "parameters":{
                "type":"object",
                "properties":{"reason":{"type":"string"},"preferred_channel":{"type":"string"}},
                "required":["reason"],
            },
        },
    },
    {
        "type":"function",
        "function":{
            "name":"book_meeting",
            "description":"Book only after explicit agreement to an exact date and exact clock time.",
            "parameters":{
                "type":"object",
                "properties":{
                    "title":{"type":"string"},
                    "starts_at":{"type":"string","description":"ISO-8601 exact date/time"},
                    "timezone":{"type":"string"},
                    "notes":{"type":"string"},
                },
                "required":["title","starts_at","timezone"],
            },
        },
    },
    {
        "type":"function",
        "function":{
            "name":"record_outcome",
            "description":"Record a call outcome.",
            "parameters":{
                "type":"object",
                "properties":{"outcome":{"type":"string","enum":["interested","callback","not_interested","wrong_number","voicemail","booked","dnc","no_answer"]}},
                "required":["outcome"],
            },
        },
    },
]

TOOL_SCENARIOS = [
    {"id":"dnc_tool","prospect":"Take me off your list. Do not call me again.","expected":"mark_do_not_call"},
    {"id":"followup_tool","prospect":"Email me some information first instead of continuing this call.","expected":"request_human_follow_up"},
    {"id":"booking_tool","prospect":"Yes, book the consultation for September 15, 2026 at 2 PM Eastern.","expected":"book_meeting","arg_check":"booking"},
    {"id":"vague_time_no_booking","prospect":"Call me tomorrow afternoon.","expected":None,"forbidden":"book_meeting","response_check":"exact_time"},
    {"id":"decline_outcome","prospect":"I'm not interested. Thanks.","expected":"record_outcome","arg_check":"not_interested"},
]

THINK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.I | re.S)
STAGE_RE = re.compile(r"\*[^*\n]{1,120}\*|\[[^\]\n]{1,120}\]|```.*?```", re.S)
MARKDOWN_RE = re.compile(r"(?:^|\n)\s{0,3}(?:#{1,6}\s+|[-*+]\s+|\d+\.\s+)")
META_RE = re.compile(
    r"\b(?:current call state|current action state|the prospect (?:wants|asked|says)|the user (?:wants|asked|says)|"
    r"as dialforge|my role is|my job is|i need to|i should|i must|let me (?:check|think)|"
    r"according to (?:the )?(?:prompt|instructions|rules)|the instructions say|we are in (?:a|the) state|"
    r"since the time is vague|reasoning:|analysis:)\b",
    re.I,
)
UNSUPPORTED_FORWARD_RE = re.compile(r"\b(?:i|we)(?:\s+will|'ll|’ll)\s+(?:send|email|text|message|call|follow\s+up)\b", re.I)
UNSUPPORTED_PAST_RE = re.compile(r"\b(?:i|we)(?:\s+have|'ve|’ve)\s+(?:sent|emailed|texted|messaged|called)\b", re.I)
BOOKING_RE = re.compile(
    r"(?:\b(?:you(?:'re|’re|\s+are)|your\s+(?:meeting|call|appointment|consultation|demo|session)\s+is|i(?:'ve|’ve|\s+have)|we(?:'ve|’ve|\s+have))\s+(?:booked|scheduled|confirmed)\b|"
    r"\b(?:meeting|appointment|call|consultation|demo|session)\s+(?:is|has\s+been)\s+(?:booked|scheduled|confirmed)\b|\b(?:confirmed|all set)\s+for\b)",
    re.I,
)
PRICE_NUMBER_RE = re.compile(r"(?:[$£€]\s*\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s*(?:usd|dollars?|pkr|rupees?|gbp|pounds?|eur|euros?)\b)", re.I)
PRICE_RATE_RE = re.compile(r"\b(?:starts?\s+at|from|costs?|price(?:d)?\s+at|fee(?:s)?\s+(?:is|are)?|retainer(?:\s+is)?)\s*[$£€]?\s*\d[\d,]*(?:\.\d+)?", re.I)
PRICE_FREE_RE = re.compile(r"\b(?:free|no\s+cost|at\s+no\s+cost|no\s+charge|at\s+no\s+charge|complimentary)\b", re.I)
SALES_WORDS_RE = re.compile(r"\b(?:service|offer|price|pricing|meeting|book|website|marketing|agency|demo|consultation)\b", re.I)


def rnd(value: Any, digits: int = 2):
    return None if value is None else round(float(value), digits)


def pct(values: list[float], p: float):
    return None if not values else rnd(np.percentile(np.asarray(values, dtype=float), p))


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", str(text or "").lower())


def sentence_count(text: str) -> int:
    value = str(text or "").strip()
    if not value:
        return 0
    return len([x for x in re.split(r"(?<=[.!?])\s+", value) if x.strip()]) or 1


def first_sentence(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    match = re.search(r".+?[.!?](?:\s|$)", value, re.S)
    return (match.group(0) if match else value).strip()


def price_tokens(text: str) -> set[str]:
    value = str(text or "")
    result: set[str] = set()
    if PRICE_FREE_RE.search(value):
        result.add("free")
    for pattern in (PRICE_NUMBER_RE, PRICE_RATE_RE):
        for match in pattern.finditer(value):
            for number in re.findall(r"\d[\d,]*(?:\.\d+)?", match.group(0)):
                result.add(number.replace(",", ""))
    return result


def price_supported(sentence: str, source: str) -> bool:
    claims = price_tokens(sentence)
    return not claims or claims.issubset(price_tokens(source))


def strip_meta(text: str) -> str:
    value = THINK_RE.sub(" ", str(text or ""))
    value = STAGE_RE.sub(" ", value)
    value = MARKDOWN_RE.sub(" ", value)
    value = re.sub(r"</?think\b[^>]*>", " ", value, flags=re.I)
    value = re.sub(r"\s+", " ", value).strip(" \t\r\n-*#`")
    if not value:
        return ""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", value) if s.strip()]
    clean = [s for s in sentences if not META_RE.search(s)]
    return " ".join(clean[:3]).strip()


def production_guard(text: str, *, allow_booking: bool, pricing_source: str) -> tuple[str, list[str]]:
    meta_clean = strip_meta(text)
    interventions: list[str] = []
    if meta_clean != re.sub(r"\s+", " ", str(text or "")).strip():
        interventions.append("meta_or_format")
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", meta_clean) if s.strip()]
    output: list[str] = []
    for sentence in sentences:
        safe = sentence
        if UNSUPPORTED_FORWARD_RE.search(safe) or UNSUPPORTED_PAST_RE.search(safe):
            safe = "I can note that request for the team."
            interventions.append("unsupported_action")
        elif not allow_booking and BOOKING_RE.search(safe):
            safe = "I can only confirm that after it is saved."
            interventions.append("booking_before_success")
        elif not price_supported(safe, pricing_source):
            safe = "I don't have the exact pricing in front of me, so I don't want to guess."
            interventions.append("unsupported_pricing")
        if safe and (not output or safe != output[-1]):
            output.append(safe)
        if len(output) >= 3:
            break
    return " ".join(output).strip(), sorted(set(interventions))


def evaluate_quality(scenario: dict[str, Any], response: str) -> dict[str, Any]:
    text = str(response or "").strip()
    lower = text.lower()
    checks: dict[str, bool] = {
        "nonempty": bool(text),
        "no_reasoning_leak": not bool(META_RE.search(text)) and "<think" not in lower,
        "no_stage_directions": not bool(STAGE_RE.search(text)) and not bool(MARKDOWN_RE.search(text)),
        "concise": sentence_count(text) <= 3 and len(words(text)) <= MAX_SPOKEN_WORDS,
    }
    rule = scenario["rule"]
    if rule == "no_price_fabrication":
        checks[rule] = not price_tokens(text)
    elif rule == "no_fake_send":
        checks[rule] = not bool(UNSUPPORTED_FORWARD_RE.search(text) or UNSUPPORTED_PAST_RE.search(text))
    elif rule == "no_fake_callback":
        checks[rule] = not bool(UNSUPPORTED_FORWARD_RE.search(text)) and not bool(BOOKING_RE.search(text))
    elif rule == "confirm_recorded_booking":
        checks[rule] = bool(BOOKING_RE.search(text))
    elif rule == "respect_refusal":
        checks[rule] = len(words(text)) <= 28 and not bool(re.search(r"\b(?:but|however|hear me out|one minute|just give)\b", lower))
    elif rule == "dnc_stop":
        checks[rule] = len(words(text)) <= 24 and not bool(SALES_WORDS_RE.search(text))
    elif rule == "brief_busy":
        checks[rule] = len(words(text)) <= 28
    elif rule == "brief_identity":
        checks[rule] = len(words(text)) <= 28
    failed = [name for name, ok in checks.items() if not ok]
    return {"passed": all(checks.values()), "checks": checks, "failures": failed}


def guard_regressions() -> dict[str, Any]:
    cases = [
        {"raw":"<think>secret</think>Hello there.","allow":False,"source":"","must":"Hello there."},
        {"raw":"We are in the current call state. I should be concise. Thanks for your time.","allow":False,"source":"","must":"Thanks for your time."},
        {"raw":"I'll call you tomorrow afternoon. What time works best?","allow":False,"source":"","must":"I can note that request for the team. What time works best?"},
        {"raw":"Your consultation is booked for Tuesday at 2 PM.","allow":False,"source":"","must":"I can only confirm that after it is saved."},
        {"raw":"Your consultation is booked for Tuesday at 2 PM.","allow":True,"source":"","must":"Your consultation is booked for Tuesday at 2 PM."},
        {"raw":"The consultation is free.","allow":False,"source":"No pricing supplied.","must":"I don't have the exact pricing in front of me, so I don't want to guess."},
        {"raw":"It starts at $300 per month.","allow":False,"source":"Starter package is $300 per month.","must":"It starts at $300 per month."},
    ]
    rows = []
    for case in cases:
        actual, interventions = production_guard(case["raw"], allow_booking=case["allow"], pricing_source=case["source"])
        rows.append({**case,"actual":actual,"interventions":interventions,"passed":actual == case["must"]})
    return {"passed": all(x["passed"] for x in rows), "cases": rows}


class GPUMonitor:
    def __init__(self):
        self.samples: list[dict[str, float]] = []
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def sample(self):
        if not shutil.which("nvidia-smi"):
            return
        try:
            line = subprocess.check_output(
                ["nvidia-smi","--query-gpu=memory.used,utilization.gpu,power.draw","--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL, timeout=2,
            ).strip().splitlines()[0]
            mem, util, power = [x.strip() for x in line.split(",")]
            self.samples.append({"memory_mb":float(mem),"util_pct":float(util),"power_w":0.0 if "N/A" in power else float(power)})
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

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {"peak_gpu_memory_mb":None,"peak_gpu_util_pct":None,"peak_gpu_power_w":None}
        return {
            "peak_gpu_memory_mb":rnd(max(x["memory_mb"] for x in self.samples)),
            "peak_gpu_util_pct":rnd(max(x["util_pct"] for x in self.samples)),
            "peak_gpu_power_w":rnd(max(x["power_w"] for x in self.samples)),
        }


def system_info() -> dict[str, Any]:
    gpu: list[dict[str, Any]] = []
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL,
            )
            for row in out.strip().splitlines():
                name, mem, driver = [x.strip() for x in row.split(",",2)]
                gpu.append({"name":name,"memory_total_mb":float(mem),"driver":driver})
        except Exception:
            pass
    return {
        "timestamp_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        "platform":platform.platform(),
        "python":sys.version.split()[0],
        "cpu":platform.processor() or platform.machine(),
        "cpu_logical_cores":psutil.cpu_count(True),
        "ram_total_gb":rnd(psutil.virtual_memory().total/1024**3),
        "torch":torch.__version__,
        "cuda_available":torch.cuda.is_available(),
        "cuda_version":torch.version.cuda,
        "gpu":gpu,
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


def pull_model(model: str) -> float:
    print(f"Pulling/verifying {model}...", flush=True)
    started = time.perf_counter()
    proc = subprocess.run(["ollama","pull",model], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(proc.stdout[-8000:])
    return rnd(time.perf_counter() - started)


def unload_model(model: str):
    try:
        requests.post(f"{OLLAMA}/api/chat", json={"model":model,"messages":[],"keep_alive":0}, timeout=30)
    except Exception:
        pass


def build_system(scenario: dict[str, Any]) -> str:
    return SYSTEM_BASE + "\n" + CAMPAIGN_CONTEXT + "\nCURRENT ACTION STATE / FACTS:\n" + str(scenario["state"])


def chat_turn(model: str, scenario: dict[str, Any], max_tokens: int = 52) -> dict[str, Any]:
    body = {
        "model":model,
        "messages":[{"role":"system","content":build_system(scenario)},{"role":"user","content":scenario["prospect"]}],
        "stream":True,
        "think":False,
        "keep_alive":"30m",
        "options":{"temperature":0.15,"top_p":0.8,"top_k":20,"repeat_penalty":1.05,"num_predict":max_tokens},
    }
    started = time.perf_counter()
    first = None
    raw_first_sentence_at = None
    chunks: list[str] = []
    thinking: list[str] = []
    final: dict[str, Any] = {}
    with requests.post(f"{OLLAMA}/api/chat", json=body, stream=True, timeout=240) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            item = json.loads(line)
            msg = item.get("message") or {}
            content = msg.get("content") or ""
            thought = msg.get("thinking") or ""
            if thought:
                thinking.append(thought)
            if content:
                now = time.perf_counter()
                if first is None:
                    first = now
                chunks.append(content)
                if raw_first_sentence_at is None and re.search(r"[.!?](?:\s|$)", "".join(chunks)):
                    raw_first_sentence_at = now
            if item.get("done"):
                final = item
    finished = time.perf_counter()
    raw = "".join(chunks).strip()
    guarded, interventions = production_guard(
        raw,
        allow_booking=bool(scenario.get("allow_booking")),
        pricing_source=CAMPAIGN_CONTEXT,
    )
    count = int(final.get("eval_count") or 0)
    duration = int(final.get("eval_duration") or 0)
    return {
        "ttft_ms":rnd(((first or finished)-started)*1000),
        "raw_first_sentence_ms":rnd(((raw_first_sentence_at or finished)-started)*1000),
        "wall_ms":rnd((finished-started)*1000),
        "tokens_per_second":rnd(count/(duration/1e9)) if count and duration else None,
        "raw_response":raw,
        "response":guarded,
        "first_sentence":first_sentence(guarded),
        "guard_interventions":interventions,
        "separate_thinking_chars":len("".join(thinking)),
        "load_ms":rnd((final.get("load_duration") or 0)/1e6),
    }


def warm_model(model: str):
    scenario = {"id":"warmup","prospect":"Hi.","state":"Warmup only. No external action is recorded.","rule":"brief_busy"}
    first = chat_turn(model, scenario, 12)
    warm = chat_turn(model, scenario, 12)
    return {"first":first,"warm":warm}


def tool_turn(model: str, case: dict[str, Any]) -> dict[str, Any]:
    system = SYSTEM_BASE + "\nUse the available tools whenever the requested real-world action requires one. Do not claim an action happened without calling its tool."
    body = {
        "model":model,
        "messages":[{"role":"system","content":system},{"role":"user","content":case["prospect"]}],
        "tools":TOOLS,
        "stream":False,
        "think":False,
        "keep_alive":"30m",
        "options":{"temperature":0.0,"top_p":0.8,"top_k":20,"num_predict":96},
    }
    started = time.perf_counter()
    response = requests.post(f"{OLLAMA}/api/chat", json=body, timeout=240)
    response.raise_for_status()
    payload = response.json()
    elapsed = (time.perf_counter() - started) * 1000
    message = payload.get("message") or {}
    calls = message.get("tool_calls") or []
    names: list[str] = []
    parsed_calls: list[dict[str, Any]] = []
    for call in calls:
        function = call.get("function") or {}
        name = str(function.get("name") or "")
        args = function.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"_raw":args}
        names.append(name)
        parsed_calls.append({"name":name,"arguments":args})

    expected = case.get("expected")
    forbidden = case.get("forbidden")
    passed = True
    reasons: list[str] = []
    if expected and expected not in names:
        passed = False
        reasons.append(f"missing_{expected}")
    if forbidden and forbidden in names:
        passed = False
        reasons.append(f"forbidden_{forbidden}")

    arg_check = case.get("arg_check")
    if passed and arg_check == "booking":
        call = next((x for x in parsed_calls if x["name"] == "book_meeting"), None)
        args = (call or {}).get("arguments") or {}
        if not (args.get("starts_at") and args.get("timezone")):
            passed = False
            reasons.append("booking_missing_exact_time_or_timezone")
    if passed and arg_check == "not_interested":
        call = next((x for x in parsed_calls if x["name"] == "record_outcome"), None)
        args = (call or {}).get("arguments") or {}
        if str(args.get("outcome") or "") != "not_interested":
            passed = False
            reasons.append("wrong_outcome")
    if passed and case.get("response_check") == "exact_time" and not names:
        content = str(message.get("content") or "")
        if not re.search(r"\b(?:what|which|exact|specific)\b.*\btime\b|\btime\b.*\b(?:work|prefer)", content, re.I):
            passed = False
            reasons.append("did_not_ask_exact_time")

    return {
        "id":case["id"],
        "elapsed_ms":rnd(elapsed),
        "tool_calls":parsed_calls,
        "content":str(message.get("content") or ""),
        "passed":passed,
        "failures":reasons,
    }


def benchmark_tools(model: str) -> dict[str, Any]:
    print(f"\n[tool routing] {model}", flush=True)
    rows = []
    for case in TOOL_SCENARIOS:
        row = tool_turn(model, case)
        rows.append(row)
        print(f"  {case['id']}: {'PASS' if row['passed'] else 'FAIL'} {row['tool_calls']}", flush=True)
    return {
        "cases":rows,
        "summary":{
            "pass_rate":rnd(100*sum(1 for x in rows if x["passed"])/len(rows),1),
            "passes":sum(1 for x in rows if x["passed"]),
            "total":len(rows),
            "median_ms":rnd(statistics.median(x["elapsed_ms"] for x in rows)),
        },
    }


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


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes()/float(handle.getframerate())


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = words(reference)
    hyp = words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp)+1))
    for i, rw in enumerate(ref, 1):
        current = [i]
        for j, hw in enumerate(hyp, 1):
            current.append(min(current[-1]+1, prev[j]+1, prev[j-1]+(rw != hw)))
        prev = current
    return prev[-1]/len(ref)


def prepare_tts(work: Path):
    print("\n[Chatterbox Nano] load, synthetic prospect audio, voice conditionals, warmup", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    with GPUMonitor() as monitor:
        started = time.perf_counter()
        tts = ChatterboxTurboTTS.from_pretrained(device=device, nano=True)
        load_ms = (time.perf_counter()-started)*1000
        paths: list[Path] = []
        for scenario in SCENARIOS:
            path = work/f"prospect_{scenario['id']}.wav"
            wav = tts.generate(scenario["prospect"], exaggeration=0.0, cfg_weight=0.0, temperature=0.8, norm_loudness=True)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            save_wav(path, wav, tts.sr)
            paths.append(path)
        reference = work/"reference.wav"
        if not reference.exists():
            urllib.request.urlretrieve(REFERENCE_URL, reference)
        prep_started = time.perf_counter()
        tts.prepare_conditionals(str(reference), exaggeration=0.0, norm_loudness=True)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        prepare_ms = (time.perf_counter()-prep_started)*1000
        warm_started = time.perf_counter()
        _ = tts.generate("Thanks for taking the call.", exaggeration=0.0, cfg_weight=0.0, temperature=0.8, norm_loudness=True)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        warm_ms = (time.perf_counter()-warm_started)*1000
    return {
        "device":device,"load_ms":rnd(load_ms),"prepare_voice_ms":rnd(prepare_ms),"warm_generation_ms":rnd(warm_ms),
        **monitor.summary(),
    }, tts, paths


def synthesize(tts, text: str):
    started = time.perf_counter()
    wav = tts.generate(text, exaggeration=0.0, cfg_weight=0.0, temperature=0.8, norm_loudness=True)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    elapsed = time.perf_counter()-started
    audio_seconds = wav.shape[-1]/float(tts.sr)
    return wav, {"generation_ms":rnd(elapsed*1000),"audio_seconds":rnd(audio_seconds),"realtime_factor":rnd(elapsed/audio_seconds,3) if audio_seconds else None}


def prepare_stt(paths: list[Path]):
    print("\n[Faster Whisper small.en] production CUDA int8 warmup", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "int8"
    with GPUMonitor() as monitor:
        started = time.perf_counter()
        stt = WhisperModel("small.en", device=device, compute_type=compute)
        load_ms = (time.perf_counter()-started)*1000
        warm_started = time.perf_counter()
        segments, _ = stt.transcribe(str(paths[0]), language="en", vad_filter=True, beam_size=1)
        _ = " ".join(s.text.strip() for s in segments if s.text.strip())
        warm_ms = (time.perf_counter()-warm_started)*1000
    return {"model":"small.en","device":device,"compute_type":compute,"load_ms":rnd(load_ms),"warm_transcription_ms":rnd(warm_ms),**monitor.summary()}, stt


def transcribe(stt, path: Path, expected: str):
    started = time.perf_counter()
    segments, _ = stt.transcribe(str(path), language="en", vad_filter=True, beam_size=1)
    text = " ".join(s.text.strip() for s in segments if s.text.strip())
    elapsed = time.perf_counter()-started
    audio = wav_duration(path)
    return text, {"transcription_ms":rnd(elapsed*1000),"realtime_factor":rnd(elapsed/audio,3),"wer":rnd(word_error_rate(expected,text),4)}


def benchmark_candidate(model: str, stt, tts, paths: list[Path], work: Path) -> dict[str, Any]:
    print(f"\n[CANDIDATE ACCEPTANCE] {model}", flush=True)
    pull_seconds = pull_model(model)
    warmup = warm_model(model)
    tool_result = benchmark_tools(model)
    rows: list[dict[str, Any]] = []
    with GPUMonitor() as monitor:
        for index, scenario in enumerate(SCENARIOS):
            transcript, stt_metrics = transcribe(stt, paths[index], scenario["prospect"])
            llm = chat_turn(model, {**scenario,"prospect":transcript or scenario["prospect"]})
            quality = evaluate_quality(scenario, llm["response"])
            spoken = llm["first_sentence"] or "Understood."
            wav, tts_metrics = synthesize(tts, spoken)
            save_wav(work/f"{model.replace(':','_').replace('/','_')}_{scenario['id']}.wav", wav, tts.sr)
            # The production guard is sentence-buffered. If it rewrites the first completed
            # model sentence, it can release that replacement as soon as that raw sentence closes.
            voice_start = stt_metrics["transcription_ms"] + llm["raw_first_sentence_ms"] + tts_metrics["generation_ms"]
            row = {
                "scenario":scenario["id"],
                "transcript":transcript,
                "raw_response":llm["raw_response"],
                "response":llm["response"],
                "first_sentence":spoken,
                "guard_interventions":llm["guard_interventions"],
                "stt_ms":stt_metrics["transcription_ms"],
                "stt_wer":stt_metrics["wer"],
                "llm_ttft_ms":llm["ttft_ms"],
                "llm_sentence_ms":llm["raw_first_sentence_ms"],
                "llm_tokens_per_second":llm["tokens_per_second"],
                "tts_ms":tts_metrics["generation_ms"],
                "tts_rtf":tts_metrics["realtime_factor"],
                "voice_start_ms":rnd(voice_start),
                "quality_pass":quality["passed"],
                "quality_failures":quality["failures"],
            }
            rows.append(row)
            print(
                f"  {scenario['id']:16s} voice {row['voice_start_ms']:>8} ms | "
                f"quality {'PASS' if row['quality_pass'] else 'FAIL'} | guard {','.join(row['guard_interventions']) or 'none'}",
                flush=True,
            )
    summary = {
        "voice_start_median_ms":rnd(statistics.median(x["voice_start_ms"] for x in rows)),
        "voice_start_p95_ms":pct([x["voice_start_ms"] for x in rows],95),
        "voice_start_max_ms":rnd(max(x["voice_start_ms"] for x in rows)),
        "quality_pass_rate":rnd(100*sum(1 for x in rows if x["quality_pass"])/len(rows),1),
        "scenario_passes":sum(1 for x in rows if x["quality_pass"]),
        "scenario_total":len(rows),
        "guard_intervention_turns":sum(1 for x in rows if x["guard_interventions"]),
        "guard_intervention_rate":rnd(100*sum(1 for x in rows if x["guard_interventions"])/len(rows),1),
        "stt_wer_median":rnd(statistics.median(x["stt_wer"] for x in rows),4),
        "tts_rtf_median":rnd(statistics.median(x["tts_rtf"] for x in rows),3),
        "llm_ttft_median_ms":rnd(statistics.median(x["llm_ttft_ms"] for x in rows)),
        "tokens_per_second_median":rnd(statistics.median(x["llm_tokens_per_second"] for x in rows if x["llm_tokens_per_second"] is not None)),
        **monitor.summary(),
    }
    return {"model":model,"pull_seconds":pull_seconds,"warmup":warmup,"tools":tool_result,"turns":rows,"summary":summary}


def score_candidate(candidate: dict[str, Any], guards_ok: bool) -> dict[str, Any]:
    s = candidate.get("summary",{})
    voice = s.get("voice_start_median_ms")
    quality = s.get("quality_pass_rate") or 0
    tools = candidate.get("tools",{}).get("summary",{}).get("pass_rate") or 0
    intervention_rate = s.get("guard_intervention_rate") or 0
    scenario_passes = s.get("scenario_passes") or 0
    scenario_total = s.get("scenario_total") or len(SCENARIOS)
    stt_wer = s.get("stt_wer_median")
    tts_rtf = s.get("tts_rtf_median")

    if voice is None: latency = 0
    elif voice <= 1500: latency = 35
    elif voice <= 2200: latency = 33
    elif voice <= 3000: latency = 30
    elif voice <= 4000: latency = 24
    elif voice <= 5500: latency = 16
    else: latency = 6
    spoken = 30*quality/100.0
    tool_score = 20*tools/100.0
    stability = 10*(scenario_passes/max(1,scenario_total))
    if not guards_ok: stability = max(0,stability-6)
    efficiency = 0.0
    if stt_wer is not None: efficiency += 2.5 if stt_wer <= 0.08 else 1.5 if stt_wer <= 0.15 else 0.5
    if tts_rtf is not None: efficiency += 2.5 if tts_rtf <= 0.55 else 1.5 if tts_rtf <= 0.8 else 0.5 if tts_rtf <= 1 else 0
    penalty = 0.0
    if intervention_rate > 50: penalty = 5.0
    elif intervention_rate > 25: penalty = 2.0
    score = rnd(max(0,latency+spoken+tool_score+stability+efficiency-penalty),1)
    blockers: list[str] = []
    if not guards_ok: blockers.append("guard_regression")
    if quality < 100: blockers.append("spoken_integrity_below_100")
    if tools < 80: blockers.append("tool_routing_below_80")
    if voice is None or voice > 3000: blockers.append("median_voice_start_above_3s")
    if score >= 92 and quality == 100 and tools >= 90 and (voice or math.inf) <= 3000 and guards_ok:
        verdict = "PHENOMENAL"
    elif score >= 85 and tools >= 80:
        verdict = "PRODUCTION-STRONG"
    elif score >= 75:
        verdict = "GOOD, TUNE BEFORE RELEASE"
    else:
        verdict = "NEEDS WORK"
    return {
        "score":score,"verdict":verdict,"blockers":blockers,
        "breakdown":{"latency":rnd(latency,1),"spoken_integrity":rnd(spoken,1),"tool_routing":rnd(tool_score,1),"stability":rnd(stability,1),"efficiency":rnd(efficiency,1),"guard_reliance_penalty":rnd(penalty,1)},
    }


def select_profile(candidates: dict[str, Any], scored: dict[str, Any], limit_mb: float) -> dict[str, Any] | None:
    viable = []
    for model, candidate in candidates.items():
        peak = candidate.get("summary",{}).get("peak_gpu_memory_mb")
        result = scored.get(model,{})
        if peak is None or peak > limit_mb:
            continue
        if result.get("verdict") not in {"PHENOMENAL","PRODUCTION-STRONG"}:
            continue
        viable.append((float(result.get("score") or 0), -float(peak), model))
    if not viable:
        return None
    viable.sort(reverse=True)
    score, neg_peak, model = viable[0]
    return {"model":model,"score":score,"measured_peak_gpu_mb":-neg_peak,"target_limit_mb":limit_mb}


def error_payload(exc: Exception) -> dict[str, str]:
    return {"error":f"{type(exc).__name__}: {exc}"}


def render_html(report: dict[str, Any]) -> str:
    system = report["system"]
    gpu = (system.get("gpu") or [{}])[0]
    winner = report.get("recommended",{}).get("8gb") or {}
    candidate_cards = "".join(
        f"<div class='candidate'><small>{html.escape(model)}</small><strong>{score.get('score','—')}</strong><span class='{('great' if score.get('verdict')=='PHENOMENAL' else 'good' if score.get('verdict')=='PRODUCTION-STRONG' else 'warn')}'>{html.escape(score.get('verdict','ERROR'))}</span></div>"
        for model, score in report.get("scores",{}).items()
    )
    sections=[]
    for model, data in report.get("candidates",{}).items():
        if not data.get("turns"): continue
        rows="".join(
            f"<tr class={'pass' if r['quality_pass'] else 'fail'}><td>{html.escape(r['scenario'])}</td><td>{r['stt_ms']}</td><td>{r['llm_sentence_ms']}</td><td>{r['tts_ms']}</td><td><b>{r['voice_start_ms']}</b></td><td>{'PASS' if r['quality_pass'] else 'FAIL: '+html.escape(', '.join(r['quality_failures']))}</td><td>{html.escape(', '.join(r['guard_interventions']) or 'none')}</td><td>{html.escape(r['response'])}</td></tr>"
            for r in data["turns"]
        )
        toolrows="".join(
            f"<tr class={'pass' if r['passed'] else 'fail'}><td>{html.escape(r['id'])}</td><td>{'PASS' if r['passed'] else 'FAIL'}</td><td><code>{html.escape(json.dumps(r['tool_calls']))}</code></td><td>{html.escape(', '.join(r['failures']))}</td></tr>"
            for r in data.get("tools",{}).get("cases",[])
        )
        sections.append(f"<section><h2>{html.escape(model)}</h2><p>Score: <b>{report.get('scores',{}).get(model,{}).get('score','—')}</b> · quality {data.get('summary',{}).get('quality_pass_rate')}% · tool routing {data.get('tools',{}).get('summary',{}).get('pass_rate')}% · median voice {data.get('summary',{}).get('voice_start_median_ms')} ms · peak GPU {data.get('summary',{}).get('peak_gpu_memory_mb')} MB</p><div class='tablewrap'><table><tr><th>Scenario</th><th>STT</th><th>LLM sentence</th><th>TTS</th><th>Voice start</th><th>Integrity</th><th>Guard</th><th>Prospect hears</th></tr>{rows}</table></div><h3>Tool routing</h3><div class='tablewrap'><table><tr><th>Case</th><th>Result</th><th>Tool calls</th><th>Failure</th></tr>{toolrows}</table></div></section>")
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>Dialforge Final Acceptance v3</title><style>
:root{{--bg:#07080a;--panel:#111318;--line:#292e36;--text:#f6f7f8;--muted:#969da9;--orange:#ff9f1c;--green:#55d98b;--red:#ff747d}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 15% -10%,#2b1b08 0,transparent 32%),var(--bg);color:var(--text);font:14px system-ui}}main{{max-width:1380px;margin:auto;padding:40px 26px 80px}}h1{{font-size:44px;margin:8px 0}}h2{{font-size:22px}}h3{{margin-top:24px}}p,small{{color:var(--muted)}}section,.hero{{background:linear-gradient(180deg,#171a20,var(--panel));border:1px solid var(--line);border-radius:18px;padding:20px;margin:16px 0}}.orange{{color:var(--orange)}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.candidate{{border:1px solid var(--line);border-radius:14px;padding:16px;background:#0d0f12;display:flex;flex-direction:column;gap:7px}}.candidate strong{{font-size:34px}}.great{{color:var(--green)}}.good{{color:#93eaae}}.warn{{color:#ffc36b}}.tablewrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;min-width:980px}}th,td{{padding:10px;border-bottom:1px solid #252a32;text-align:left;vertical-align:top}}th{{color:var(--orange);font-size:12px;text-transform:uppercase}}tr.pass td:first-child{{border-left:3px solid var(--green)}}tr.fail td:first-child{{border-left:3px solid var(--red)}}code,pre{{white-space:pre-wrap;word-break:break-word}}@media(max-width:850px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><div class='hero'><small>FINAL PRODUCTION ACCEPTANCE v3 · TOOL-AWARE</small><h1>Dial<span class='orange'>forge</span></h1><p>Measures warmed speech latency, deterministic spoken safety, action-tool routing and GPU headroom. No score is awarded for hidden reasoning or fake actions.</p><p>{system['timestamp_utc']} · {html.escape(str(gpu.get('name','CPU')))} · {system.get('ram_total_gb')} GB RAM</p><h2>Recommended 8 GB candidate: <span class='orange'>{html.escape(str(winner.get('model','none passed')))}</span></h2></div><div class='grid'>{candidate_cards}</div>{''.join(sections)}<section><h2>Profile recommendations</h2><pre>{html.escape(json.dumps(report.get('recommended',{}),indent=2))}</pre></section><section><h2>Guard regression suite</h2><pre>{html.escape(json.dumps(report.get('guard_regressions',{}),indent=2))}</pre></section><section><h2>Component warmup</h2><pre>{html.escape(json.dumps({'whisper':report.get('whisper'),'chatterbox':report.get('chatterbox')},indent=2))}</pre></section><section><h2>Scope</h2><p>This is the final synthetic local-AI acceptance. LiveKit/SIP/PSTN transport still requires one real Windows call because carrier and network behavior cannot be truthfully simulated in Colab.</p></section></main></body></html>'''


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--models",nargs="+",default=list(DEFAULT_MODELS))
    parser.add_argument("--output-dir",default="dialforge-final-acceptance-v3")
    args=parser.parse_args()
    output=Path(args.output_dir).resolve(); work=output/"work"; work.mkdir(parents=True,exist_ok=True)
    report: dict[str,Any]={
        "schema":4,"product":"Dialforge","benchmark":"final-production-acceptance-v3-tool-aware",
        "notes":["Warmed local AI path only; excludes LiveKit/SIP/PSTN.","Spoken score is measured after the deterministic production-style guard.","Tool-routing is scored separately so guard rewrites cannot hide an agent that chooses the wrong action."],
        "system":system_info(),"guard_regressions":guard_regressions(),"candidates":{},"scores":{},
    }
    print("\n=== DIALFORGE FINAL PRODUCTION ACCEPTANCE v3 ===",flush=True)
    print(json.dumps(report["system"],indent=2),flush=True)
    wait_ollama()
    tts=None; stt=None; paths=[]
    try:
        report["chatterbox"],tts,paths=prepare_tts(work)
    except Exception as exc:
        report["chatterbox"]=error_payload(exc); print("[TTS ERROR]",exc,flush=True)
    if paths:
        try:
            report["whisper"],stt=prepare_stt(paths)
        except Exception as exc:
            report["whisper"]=error_payload(exc); print("[STT ERROR]",exc,flush=True)
    else:
        report["whisper"]={"error":"synthetic prospect audio unavailable"}
    if stt is not None and tts is not None:
        for model in args.models:
            try:
                report["candidates"][model]=benchmark_candidate(model,stt,tts,paths,work)
            except Exception as exc:
                report["candidates"][model]=error_payload(exc); print(f"[CANDIDATE ERROR] {model}: {exc}",flush=True)
            finally:
                unload_model(model)
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    for model,data in report["candidates"].items():
        if data.get("summary"):
            report["scores"][model]=score_candidate(data,bool(report["guard_regressions"].get("passed")))
        else:
            report["scores"][model]={"score":0,"verdict":"ERROR","blockers":[data.get("error","candidate failed")]}
    report["recommended"]={
        "8gb":select_profile(report["candidates"],report["scores"],TARGET_8GB_SAFE_MB),
        "12gb":select_profile(report["candidates"],report["scores"],TARGET_12GB_SAFE_MB),
        "policy":"Pick the highest measured production score that preserves Windows/display VRAM headroom; do not prefer a larger model merely because it has more parameters.",
    }
    json_path=output/"dialforge-final-acceptance-v3.json"; html_path=output/"dialforge-final-acceptance-v3.html"
    json_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    html_path.write_text(render_html(report),encoding="utf-8")
    print("\n=== FINAL V3 SCORES ===",flush=True)
    print(json.dumps({"scores":report["scores"],"recommended":report["recommended"]},indent=2),flush=True)
    print("HTML:",html_path,flush=True); print("JSON:",json_path,flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
