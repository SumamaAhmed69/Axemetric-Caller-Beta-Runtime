#!/usr/bin/env python3
"""Dialforge Sales Acceptance v2.2 resilience wrapper.

Loads the validated v2 sales benchmark from the same immutable Git commit, then
replaces only the Ollama chat transport with a retrying, non-fatal transport.
A transient/model-specific Ollama 5xx must never destroy the entire benchmark.
Persistent failures are recorded as failed turns/tool cases rather than hidden.
"""
from __future__ import annotations

import base64
import importlib.util
import json
import os
import pathlib
import time
import urllib.request
from typing import Any

import requests

ROOT = pathlib.Path("/content") if pathlib.Path("/content").exists() else pathlib.Path.cwd()
OWNER = "SumamaAhmed69"
REPO = "Axemetric-Caller-Beta-Runtime"
BASE_PATH = "benchmarks/dialforge_sales_acceptance.py"
BASE_LOCAL = ROOT / "dialforge_sales_acceptance_base_v2.py"
OLLAMA_LOG = ROOT / "dialforge-final-ollama.log"


def _api_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Dialforge-Sales-v2.2", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _source_sha() -> str:
    pinned = os.environ.get("DIALFORGE_SOURCE_SHA", "").strip()
    if pinned:
        return pinned
    return _api_json(
        f"https://api.github.com/repos/{OWNER}/{REPO}/commits/main?x={time.time_ns()}"
    )["sha"]


def _load_base():
    sha = _source_sha()
    print(f"Loading Sales Acceptance v2 base from immutable commit {sha[:12]}...", flush=True)
    item = _api_json(
        f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{BASE_PATH}?ref={sha}&x={time.time_ns()}"
    )
    if item.get("encoding") != "base64" or not item.get("content"):
        raise RuntimeError("GitHub did not return the v2 sales benchmark base.")
    payload = base64.b64decode(item["content"])
    text = payload.decode("utf-8")
    compile(text, str(BASE_LOCAL), "exec")
    BASE_LOCAL.write_bytes(payload)
    spec = importlib.util.spec_from_file_location("dialforge_sales_v2_base", BASE_LOCAL)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import the sales benchmark base.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base()


def _ollama_tail() -> str:
    try:
        if OLLAMA_LOG.exists():
            return OLLAMA_LOG.read_text(encoding="utf-8", errors="replace")[-12000:]
    except Exception:
        pass
    return "(Ollama server log unavailable)"


def _response_detail(response: requests.Response | None) -> str:
    if response is None:
        return "(no HTTP response)"
    try:
        payload = response.json()
        return json.dumps(payload, ensure_ascii=False)[:4000]
    except Exception:
        return (response.text or "(empty response body)")[:4000]


def resilient_chat(model: str, messages: list[dict[str, Any]], tools=None) -> dict[str, Any]:
    """Production-like Ollama request with deterministic retries and graceful failure."""
    tool_mode = tools is not None
    attempts = 4 if tool_mode else 3
    last_error = "unknown Ollama error"

    for attempt in range(1, attempts + 1):
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "keep_alive": "30m",
            "options": {
                "temperature": 0.0 if tool_mode else 0.15,
                "top_p": 0.8,
                "num_predict": 128 if tool_mode else 80,
            },
        }
        if tool_mode:
            payload["tools"] = tools

        response = None
        started = time.perf_counter()
        try:
            response = requests.post(base.OLLAMA + "/api/chat", json=payload, timeout=180)
            elapsed = (time.perf_counter() - started) * 1000.0
            if response.status_code >= 500:
                last_error = f"HTTP {response.status_code}: {_response_detail(response)}"
                print(
                    f"[Ollama retry {attempt}/{attempts}] {model} "
                    f"{'tool' if tool_mode else 'chat'} request failed: {last_error}",
                    flush=True,
                )
                try:
                    base.unload(model)
                except Exception:
                    pass
                time.sleep(min(1.5 * attempt, 5.0))
                continue

            response.raise_for_status()
            data = response.json()
            msg = data.get("message") or {}
            content = str(msg.get("content") or "")
            eval_count = float(data.get("eval_count") or 0)
            eval_ns = float(data.get("eval_duration") or 0)
            tok_s = (eval_count / (eval_ns / 1e9)) if eval_ns > 0 else None
            load_ms = float(data.get("load_duration") or 0) / 1e6
            total_ms = float(data.get("total_duration") or 0) / 1e6
            approx_first = (
                max(
                    0.0,
                    min(elapsed, total_ms if total_ms else elapsed)
                    - (max(0, eval_count - 10) / (tok_s or 1)) * 1000,
                )
                if eval_count
                else elapsed
            )
            return {
                "content": content,
                "tool_calls": msg.get("tool_calls") or [],
                "elapsed_ms": round(elapsed, 2),
                "load_ms": round(load_ms, 2),
                "tok_s": round(tok_s, 2) if tok_s else None,
                "first_sentence_ms": round(max(1.0, approx_first), 2),
                "transport_error": None,
                "attempts": attempt,
            }
        except (requests.RequestException, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}; body={_response_detail(response)}"
            print(
                f"[Ollama retry {attempt}/{attempts}] {model} "
                f"{'tool' if tool_mode else 'chat'} request failed: {last_error}",
                flush=True,
            )
            try:
                base.unload(model)
            except Exception:
                pass
            time.sleep(min(1.5 * attempt, 5.0))

    print(
        f"[Ollama persistent failure] {model} {'tool' if tool_mode else 'chat'}: {last_error}",
        flush=True,
    )
    print("--- Ollama server log tail ---\n" + _ollama_tail(), flush=True)
    # Do not fabricate a model answer. Empty content/tool_calls scores as a real failure,
    # while allowing the rest of the benchmark to complete.
    return {
        "content": "",
        "tool_calls": [],
        "elapsed_ms": 0.0,
        "load_ms": 0.0,
        "tok_s": None,
        "first_sentence_ms": 0.0,
        "transport_error": last_error,
        "attempts": attempts,
    }


base.chat = resilient_chat


def main() -> int:
    print("=== DIALFORGE CONSULTATIVE SALES ACCEPTANCE v2.2 RESILIENT ===", flush=True)
    return int(base.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
