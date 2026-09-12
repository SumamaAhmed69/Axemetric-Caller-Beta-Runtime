#!/usr/bin/env python3
"""Dialforge cloud benchmark for Colab T4 and similar NVIDIA GPUs.

Benchmarks Qwen 3 1.7B/4B/8B via Ollama, faster-whisper small.en,
Chatterbox Nano, and full synthetic STT -> LLM -> TTS turns for every Qwen tier.
Failures are isolated per component/model so one unsupported tier does not destroy the report.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import platform
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
DEFAULT_MODELS = ("qwen3:1.7b", "qwen3:4b", "qwen3:8b")
REFERENCE_URL = "https://storage.googleapis.com/chatterbox-demo-samples/prompts/female_random_podcast.wav"
SYSTEM = (
    "You are an outbound appointment-setting assistant. Be concise, natural, truthful and respectful. "
    "Never fabricate facts. Do not pressure a prospect after a refusal. Reply in one or two short spoken sentences."
)
PROSPECT_LINES = [
    "Hello, who is this?",
    "I'm busy. What is this about?",
    "How much does your service cost?",
    "Can you send me some information first?",
    "Okay, call me tomorrow afternoon.",
]


def rnd(value: Any, digits: int = 2):
    return None if value is None else round(float(value), digits)


def percentile(values: list[float], p: float):
    return None if not values else rnd(np.percentile(np.asarray(values, dtype=float), p))


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
            self.stop.wait(0.1)

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
        "cpu_physical_cores": psutil.cpu_count(False),
        "ram_total_gb": rnd(psutil.virtual_memory().total / 1024**3),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": gpu,
    }


def wait_ollama(timeout: int = 60):
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
    started = time.perf_counter()
    proc = subprocess.run(["ollama", "pull", model], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(proc.stdout[-5000:])
    return rnd(time.perf_counter() - started)


def unload_model(model: str):
    try:
        requests.post(f"{OLLAMA}/api/generate", json={"model": model, "keep_alive": 0}, timeout=30)
    except Exception:
        pass


def llm_turn(model: str, prompt: str, max_tokens: int = 72):
    body = {
        "model": model,
        "system": SYSTEM,
        "prompt": prompt,
        "stream": True,
        "think": False,
        "keep_alive": "5m",
        "options": {"temperature": 0.2, "num_predict": max_tokens},
    }
    started = time.perf_counter()
    first = None
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
                if first is None:
                    first = time.perf_counter()
                chunks.append(text)
            if item.get("done"):
                final = item
    finished = time.perf_counter()
    count = int(final.get("eval_count") or 0)
    duration = int(final.get("eval_duration") or 0)
    return {
        "ttft_ms": rnd(((first or finished) - started) * 1000),
        "wall_ms": rnd((finished - started) * 1000),
        "tokens_per_second": rnd(count / (duration / 1e9)) if count and duration else None,
        "eval_count": count,
        "load_ms": rnd((final.get("load_duration") or 0) / 1e6),
        "response": "".join(chunks).strip(),
    }


def benchmark_qwen(model: str, repeats: int):
    print(f"\n[Qwen] {model}", flush=True)
    pull_seconds = pull_model(model)
    llm_turn(model, "Say hello in one short sentence.", 24)
    rows = []
    with GPUMonitor() as monitor:
        for i in range(repeats):
            row = llm_turn(model, PROSPECT_LINES[i % len(PROSPECT_LINES)])
            rows.append(row)
            print(f"  {i + 1}/{repeats}: TTFT {row['ttft_ms']} ms, {row['tokens_per_second']} tok/s", flush=True)
    ttft = [x["ttft_ms"] for x in rows]
    wall = [x["wall_ms"] for x in rows]
    tps = [x["tokens_per_second"] for x in rows if x["tokens_per_second"] is not None]
    result = {
        "pull_seconds": pull_seconds,
        "turns": rows,
        "summary": {
            "ttft_median_ms": rnd(statistics.median(ttft)),
            "ttft_p95_ms": percentile(ttft, 95),
            "wall_median_ms": rnd(statistics.median(wall)),
            "tokens_per_second_median": rnd(statistics.median(tps)) if tps else None,
            **monitor.summary(),
        },
    }
    unload_model(model)
    return result


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


def benchmark_tts(work: Path, repeats: int):
    print("\n[Chatterbox Nano]", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with GPUMonitor() as load_monitor:
        tts = ChatterboxTurboTTS.from_pretrained(device=device, nano=True)
    load_seconds = time.perf_counter() - started
    reference = work / "reference.wav"
    if not reference.exists():
        urllib.request.urlretrieve(REFERENCE_URL, reference)
    _ = tts.generate("Hello, this is a warmup sentence.")
    rows = []
    with GPUMonitor() as monitor:
        for i in range(repeats):
            text = [
                "Thanks for taking the call. I will keep this brief.",
                "We help local service businesses improve their digital presence.",
                "I can arrange a short follow up at a better time.",
            ][i % 3]
            started = time.perf_counter()
            wav = tts.generate(text, audio_prompt_path=str(reference))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            audio_seconds = wav.shape[-1] / float(tts.sr)
            save_wav(work / f"agent_{i}.wav", wav, tts.sr)
            rows.append({
                "generation_ms": rnd(elapsed * 1000),
                "audio_seconds": rnd(audio_seconds),
                "realtime_factor": rnd(elapsed / audio_seconds, 3),
            })
            print(f"  {i + 1}/{repeats}: {rnd(elapsed * 1000)} ms, RTF {rnd(elapsed / audio_seconds, 3)}", flush=True)
    vals = [x["generation_ms"] for x in rows]
    rtfs = [x["realtime_factor"] for x in rows]
    peak = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None
    return {
        "device": device,
        "load_seconds": rnd(load_seconds),
        "reference_source": REFERENCE_URL,
        "load_gpu": load_monitor.summary(),
        "samples": rows,
        "summary": {
            "generation_median_ms": rnd(statistics.median(vals)),
            "generation_p95_ms": percentile(vals, 95),
            "realtime_factor_median": rnd(statistics.median(rtfs), 3),
            "torch_peak_allocated_mb": rnd(peak),
            **monitor.summary(),
        },
    }, tts


def make_prospect_audio(tts, work: Path):
    paths = []
    for i, text in enumerate(PROSPECT_LINES):
        path = work / f"prospect_{i}.wav"
        paths.append(path)
        if not path.exists():
            wav = tts.generate(text)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            save_wav(path, wav, tts.sr)
    return paths


def benchmark_stt(paths: list[Path], repeats: int):
    print("\n[Faster Whisper small.en]", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "float16" if device == "cuda" else "int8"
    started = time.perf_counter()
    with GPUMonitor() as load_monitor:
        stt = WhisperModel("small.en", device=device, compute_type=compute)
    load_seconds = time.perf_counter() - started
    rows = []
    with GPUMonitor() as monitor:
        for i in range(repeats):
            path = paths[i % len(paths)]
            audio_seconds = wav_duration(path)
            started = time.perf_counter()
            segments, _ = stt.transcribe(str(path), language="en", vad_filter=True, beam_size=1)
            text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
            elapsed = time.perf_counter() - started
            rows.append({
                "transcription_ms": rnd(elapsed * 1000),
                "audio_seconds": rnd(audio_seconds),
                "realtime_factor": rnd(elapsed / audio_seconds, 3),
                "text": text,
            })
            print(f"  {i + 1}/{repeats}: {rnd(elapsed * 1000)} ms -> {text[:60]!r}", flush=True)
    vals = [x["transcription_ms"] for x in rows]
    rtfs = [x["realtime_factor"] for x in rows]
    return {
        "model": "small.en",
        "device": device,
        "compute_type": compute,
        "load_seconds": rnd(load_seconds),
        "load_gpu": load_monitor.summary(),
        "samples": rows,
        "summary": {
            "transcription_median_ms": rnd(statistics.median(vals)),
            "transcription_p95_ms": percentile(vals, 95),
            "realtime_factor_median": rnd(statistics.median(rtfs), 3),
            **monitor.summary(),
        },
    }, stt


def transcribe(stt, path: Path):
    started = time.perf_counter()
    segments, _ = stt.transcribe(str(path), language="en", vad_filter=True, beam_size=1)
    text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
    return text, time.perf_counter() - started


def benchmark_pipeline(model: str, stt, tts, paths: list[Path], work: Path, turns: int):
    print(f"\n[Full pipeline] STT -> {model} -> Chatterbox Nano", flush=True)
    pull_model(model)
    reference = work / "reference.wav"
    rows = []
    with GPUMonitor() as monitor:
        for i in range(turns):
            total_started = time.perf_counter()
            text, stt_seconds = transcribe(stt, paths[i % len(paths)])
            llm = llm_turn(model, text or PROSPECT_LINES[i % len(PROSPECT_LINES)])
            answer = llm["response"] or "Thanks for your time."
            tts_started = time.perf_counter()
            wav = tts.generate(answer, audio_prompt_path=str(reference))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            tts_seconds = time.perf_counter() - tts_started
            total_seconds = time.perf_counter() - total_started
            row = {
                "turn": i + 1,
                "transcript": text,
                "response": answer,
                "stt_ms": rnd(stt_seconds * 1000),
                "llm_ttft_ms": llm["ttft_ms"],
                "llm_wall_ms": llm["wall_ms"],
                "llm_tokens_per_second": llm["tokens_per_second"],
                "tts_ms": rnd(tts_seconds * 1000),
                "total_response_ms": rnd(total_seconds * 1000),
                "generated_audio_seconds": rnd(wav.shape[-1] / float(tts.sr)),
            }
            rows.append(row)
            print(
                f"  {i + 1}/{turns}: STT {row['stt_ms']} | LLM TTFT {row['llm_ttft_ms']} | "
                f"TTS {row['tts_ms']} | TOTAL {row['total_response_ms']} ms",
                flush=True,
            )
    totals = [x["total_response_ms"] for x in rows]
    result = {
        "model": model,
        "turn_count": turns,
        "turns": rows,
        "summary": {
            "stt_median_ms": rnd(statistics.median([x["stt_ms"] for x in rows])),
            "llm_ttft_median_ms": rnd(statistics.median([x["llm_ttft_ms"] for x in rows])),
            "tts_median_ms": rnd(statistics.median([x["tts_ms"] for x in rows])),
            "total_median_ms": rnd(statistics.median(totals)),
            "total_p95_ms": percentile(totals, 95),
            "total_max_ms": rnd(max(totals)),
            **monitor.summary(),
        },
    }
    unload_model(model)
    return result


def error_payload(exc: Exception):
    return {"error": f"{type(exc).__name__}: {exc}"}


def render_html(report: dict[str, Any]):
    hardware = report["system"]
    gpu = (hardware.get("gpu") or [{}])[0].get("name", "CPU only")
    qwen_rows = "".join(
        f"<tr><td>{html.escape(model)}</td><td>{data.get('summary',{}).get('ttft_median_ms')}</td>"
        f"<td>{data.get('summary',{}).get('ttft_p95_ms')}</td><td>{data.get('summary',{}).get('tokens_per_second_median')}</td>"
        f"<td>{data.get('summary',{}).get('peak_gpu_memory_mb')}</td><td>{html.escape(data.get('error',''))}</td></tr>"
        for model, data in report.get("qwen", {}).items()
    )
    pipeline_rows = "".join(
        f"<tr><td>{html.escape(model)}</td><td>{data.get('summary',{}).get('stt_median_ms')}</td>"
        f"<td>{data.get('summary',{}).get('llm_ttft_median_ms')}</td><td>{data.get('summary',{}).get('tts_median_ms')}</td>"
        f"<td>{data.get('summary',{}).get('total_median_ms')}</td><td>{data.get('summary',{}).get('total_p95_ms')}</td>"
        f"<td>{data.get('summary',{}).get('peak_gpu_memory_mb')}</td><td>{html.escape(data.get('error',''))}</td></tr>"
        for model, data in report.get("pipelines", {}).items()
    )
    turn_sections = []
    for model, data in report.get("pipelines", {}).items():
        if not data.get("turns"):
            continue
        rows = "".join(
            f"<tr><td>{x['turn']}</td><td>{x['stt_ms']}</td><td>{x['llm_ttft_ms']}</td><td>{x['tts_ms']}</td>"
            f"<td>{x['total_response_ms']}</td><td>{html.escape(x['response'][:160])}</td></tr>"
            for x in data["turns"]
        )
        turn_sections.append(
            f"<section><h2>{html.escape(model)} pipeline turns</h2><table><tr><th>Turn</th><th>STT ms</th>"
            f"<th>LLM TTFT ms</th><th>TTS ms</th><th>Total ms</th><th>Response</th></tr>{rows}</table></section>"
        )
    return f'''<!doctype html><meta charset="utf-8"><title>Dialforge Benchmark</title>
<style>body{{font:14px system-ui;background:#101113;color:#f4f4f5;margin:32px;max-width:1250px}}h1{{font-size:34px}}h2{{margin-top:0}}.cards{{display:flex;gap:10px;flex-wrap:wrap}}.card,section{{background:#18181b;border:1px solid #303036;border-radius:14px;padding:16px;margin:14px 0}}.card{{min-width:180px}}strong{{font-size:22px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid #303036;text-align:left;vertical-align:top}}th{{color:#fb923c}}</style>
<h1>Dialforge Cloud Benchmark</h1><p>{hardware['timestamp_utc']} · synthetic local AI pipeline · no SIP/PSTN</p>
<div class="cards"><div class="card">GPU<br><strong>{html.escape(str(gpu))}</strong></div><div class="card">RAM<br><strong>{hardware['ram_total_gb']} GB</strong></div><div class="card">Pipeline models<br><strong>{len(report.get('pipelines',{}))}</strong></div></div>
<section><h2>Qwen standalone</h2><table><tr><th>Model</th><th>Median TTFT ms</th><th>P95 TTFT ms</th><th>Median tok/s</th><th>Peak GPU MB</th><th>Error</th></tr>{qwen_rows}</table></section>
<section><h2>Full STT → LLM → TTS comparison</h2><table><tr><th>Model</th><th>STT median ms</th><th>LLM TTFT median ms</th><th>TTS median ms</th><th>Total median ms</th><th>Total P95 ms</th><th>Peak GPU MB</th><th>Error</th></tr>{pipeline_rows}</table></section>
{''.join(turn_sections)}
<section><h2>Whisper</h2><pre>{html.escape(json.dumps(report.get('whisper',{}).get('summary',{}), indent=2))}</pre></section>
<section><h2>Chatterbox Nano</h2><pre>{html.escape(json.dumps(report.get('chatterbox',{}).get('summary',{}), indent=2))}</pre></section>
<section><h2>Hardware</h2><pre>{html.escape(json.dumps(hardware, indent=2))}</pre></section>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--qwen-repeats", type=int, default=3)
    parser.add_argument("--component-repeats", type=int, default=3)
    parser.add_argument("--pipeline-turns", type=int, default=5)
    parser.add_argument("--pipeline-models", nargs="+", default=None)
    parser.add_argument("--output-dir", default="dialforge-benchmark")
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    work = output / "work"
    work.mkdir(parents=True, exist_ok=True)
    pipeline_models = args.pipeline_models or args.models
    report: dict[str, Any] = {
        "schema": 2,
        "product": "Dialforge",
        "notes": [
            "Synthetic benchmark: no LiveKit/SIP/PSTN network latency.",
            "Qwen thinking disabled for latency-sensitive voice-agent use.",
            "Full STT -> LLM -> TTS pipeline is tested separately for every selected Qwen tier.",
        ],
        "system": system_info(),
        "qwen": {},
        "pipelines": {},
    }
    print("Dialforge Cloud Benchmark\n" + json.dumps(report["system"], indent=2), flush=True)
    wait_ollama()

    for model in args.models:
        try:
            report["qwen"][model] = benchmark_qwen(model, args.qwen_repeats)
        except Exception as exc:
            report["qwen"][model] = error_payload(exc)
            print(f"[Qwen ERROR] {model}: {exc}", flush=True)
            unload_model(model)

    tts = None
    stt = None
    paths: list[Path] = []
    try:
        report["chatterbox"], tts = benchmark_tts(work, args.component_repeats)
        paths = make_prospect_audio(tts, work)
    except Exception as exc:
        report["chatterbox"] = error_payload(exc)
        print(f"[Chatterbox ERROR] {exc}", flush=True)

    if tts is not None and paths:
        try:
            report["whisper"], stt = benchmark_stt(paths, args.component_repeats)
        except Exception as exc:
            report["whisper"] = error_payload(exc)
            print(f"[Whisper ERROR] {exc}", flush=True)
    else:
        report["whisper"] = {"error": "Skipped because synthetic prospect audio could not be generated."}

    if stt is not None and tts is not None and paths:
        for model in pipeline_models:
            try:
                report["pipelines"][model] = benchmark_pipeline(model, stt, tts, paths, work, args.pipeline_turns)
            except Exception as exc:
                report["pipelines"][model] = error_payload(exc)
                print(f"[Pipeline ERROR] {model}: {exc}", flush=True)
                unload_model(model)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    else:
        for model in pipeline_models:
            report["pipelines"][model] = {"error": "Skipped because STT or TTS did not initialize."}

    json_path = output / "dialforge-benchmark-report.json"
    html_path = output / "dialforge-benchmark-report.html"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")
    print(f"\nReports:\n  {html_path}\n  {json_path}", flush=True)


if __name__ == "__main__":
    main()
