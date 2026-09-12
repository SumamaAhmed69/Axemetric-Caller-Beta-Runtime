#!/usr/bin/env python3
"""Dialforge cloud benchmark: Ollama/Qwen3 + faster-whisper + Chatterbox Nano.

Intended for an ephemeral NVIDIA GPU notebook/VM such as Google Colab.
No SIP/PSTN calls are made. The full-pipeline test is synthetic audio -> STT -> LLM -> TTS.
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
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import requests
import torch
import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS
from faster_whisper import WhisperModel

OLLAMA = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
MODELS = ("qwen3:1.7b", "qwen3:4b", "qwen3:8b")
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


def rnd(v, n=2):
    return None if v is None else round(float(v), n)


def pct(values, p):
    return None if not values else rnd(np.percentile(np.asarray(values, dtype=float), p))


class GPUMonitor:
    def __init__(self):
        self.samples = []
        self.stop = threading.Event()
        self.thread = None

    def sample(self):
        if not shutil.which("nvidia-smi"):
            return
        try:
            out = subprocess.check_output([
                "nvidia-smi", "--query-gpu=memory.used,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits"
            ], text=True, stderr=subprocess.DEVNULL, timeout=2).strip().splitlines()[0]
            mem, util, power = [x.strip() for x in out.split(",")]
            self.samples.append({
                "memory_mb": float(mem), "util_pct": float(util),
                "power_w": 0.0 if "N/A" in power else float(power)
            })
        except Exception:
            pass

    def loop(self):
        while not self.stop.is_set():
            self.sample(); self.stop.wait(0.1)

    def __enter__(self):
        self.thread = threading.Thread(target=self.loop, daemon=True); self.thread.start(); return self

    def __exit__(self, *_):
        self.stop.set(); self.thread.join(timeout=2); self.sample()

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
            out = subprocess.check_output([
                "nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"
            ], text=True, stderr=subprocess.DEVNULL)
            for row in out.strip().splitlines():
                name, mem, driver = [x.strip() for x in row.split(",", 2)]
                gpu.append({"name": name, "memory_total_mb": float(mem), "driver": driver})
        except Exception:
            pass
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(), "python": sys.version.split()[0],
        "cpu": platform.processor() or platform.machine(),
        "cpu_logical_cores": psutil.cpu_count(True), "cpu_physical_cores": psutil.cpu_count(False),
        "ram_total_gb": rnd(psutil.virtual_memory().total / 1024**3),
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda, "gpu": gpu,
    }


def wait_ollama(timeout=45):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{OLLAMA}/api/tags", timeout=2).ok: return
        except Exception: pass
        time.sleep(1)
    raise RuntimeError("Ollama did not become ready")


def pull(model):
    t = time.perf_counter()
    p = subprocess.run(["ollama", "pull", model], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode: raise RuntimeError(p.stdout[-4000:])
    return rnd(time.perf_counter() - t)


def llm_turn(model, prompt, max_tokens=72):
    body = {
        "model": model, "system": SYSTEM, "prompt": prompt, "stream": True, "think": False,
        "keep_alive": "10m", "options": {"temperature": 0.2, "num_predict": max_tokens}
    }
    started = time.perf_counter(); first = None; final = {}; chunks = []
    with requests.post(f"{OLLAMA}/api/generate", json=body, stream=True, timeout=180) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line: continue
            item = json.loads(line); text = item.get("response") or ""
            if text:
                if first is None: first = time.perf_counter()
                chunks.append(text)
            if item.get("done"): final = item
    finished = time.perf_counter()
    count = int(final.get("eval_count") or 0); duration = int(final.get("eval_duration") or 0)
    return {
        "ttft_ms": rnd(((first or finished) - started) * 1000),
        "wall_ms": rnd((finished - started) * 1000),
        "tokens_per_second": rnd(count / (duration / 1e9)) if count and duration else None,
        "eval_count": count, "load_ms": rnd((final.get("load_duration") or 0) / 1e6),
        "response": "".join(chunks).strip(),
    }


def benchmark_qwen(model, repeats):
    print(f"\n[Qwen] {model}"); pull_s = pull(model)
    llm_turn(model, "Say hello in one short sentence.", 24)
    rows = []
    with GPUMonitor() as mon:
        for i in range(repeats):
            row = llm_turn(model, PROSPECT_LINES[i % len(PROSPECT_LINES)]); rows.append(row)
            print(f"  {i+1}/{repeats}: TTFT {row['ttft_ms']} ms, {row['tokens_per_second']} tok/s")
    ttft = [x["ttft_ms"] for x in rows]; wall = [x["wall_ms"] for x in rows]
    tps = [x["tokens_per_second"] for x in rows if x["tokens_per_second"] is not None]
    return {"pull_seconds": pull_s, "turns": rows, "summary": {
        "ttft_median_ms": rnd(statistics.median(ttft)), "ttft_p95_ms": pct(ttft, 95),
        "wall_median_ms": rnd(statistics.median(wall)),
        "tokens_per_second_median": rnd(statistics.median(tps)) if tps else None, **mon.summary()
    }}


def save_audio(path, wav, sr):
    if wav.ndim == 1: wav = wav.unsqueeze(0)
    ta.save(str(path), wav.detach().cpu(), sr)


def benchmark_tts(work, repeats):
    print("\n[Chatterbox Nano]"); device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    with GPUMonitor() as load_mon: tts = ChatterboxTurboTTS.from_pretrained(device=device, nano=True)
    load_s = time.perf_counter() - t
    ref = work / "reference.wav"
    if not ref.exists(): urllib.request.urlretrieve(REFERENCE_URL, ref)
    _ = tts.generate("Hello, this is a warmup sentence.")
    rows = []
    with GPUMonitor() as mon:
        for i in range(repeats):
            text = ["Thanks for taking the call. I will keep this brief.",
                    "We help local service businesses improve their digital presence.",
                    "I can arrange a short follow up at a better time."][i % 3]
            t = time.perf_counter(); wav = tts.generate(text, audio_prompt_path=str(ref))
            if torch.cuda.is_available(): torch.cuda.synchronize()
            elapsed = time.perf_counter() - t; audio_s = wav.shape[-1] / float(tts.sr)
            save_audio(work / f"agent_{i}.wav", wav, tts.sr)
            rows.append({"generation_ms": rnd(elapsed*1000), "audio_seconds": rnd(audio_s),
                         "realtime_factor": rnd(elapsed/audio_s, 3)})
            print(f"  {i+1}/{repeats}: {rnd(elapsed*1000)} ms, RTF {rnd(elapsed/audio_s,3)}")
    vals = [x["generation_ms"] for x in rows]; rtfs = [x["realtime_factor"] for x in rows]
    peak = torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else None
    return {"device": device, "load_seconds": rnd(load_s), "reference_source": REFERENCE_URL,
            "load_gpu": load_mon.summary(), "samples": rows, "summary": {
                "generation_median_ms": rnd(statistics.median(vals)), "generation_p95_ms": pct(vals,95),
                "realtime_factor_median": rnd(statistics.median(rtfs),3), "torch_peak_allocated_mb": rnd(peak),
                **mon.summary()}}, tts


def prospect_audio(tts, work):
    paths = []
    for i, text in enumerate(PROSPECT_LINES):
        path = work / f"prospect_{i}.wav"; paths.append(path)
        if not path.exists():
            wav = tts.generate(text)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            save_audio(path, wav, tts.sr)
    return paths


def benchmark_stt(paths, repeats):
    print("\n[Faster Whisper small.en]"); device = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "float16" if device == "cuda" else "int8"; t = time.perf_counter()
    with GPUMonitor() as load_mon: stt = WhisperModel("small.en", device=device, compute_type=compute)
    load_s = time.perf_counter()-t; rows=[]
    with GPUMonitor() as mon:
        for i in range(repeats):
            path = paths[i % len(paths)]; info = ta.info(str(path)); audio_s = info.num_frames/info.sample_rate
            t=time.perf_counter(); segs,_=stt.transcribe(str(path),language="en",vad_filter=True,beam_size=1)
            text=" ".join(s.text.strip() for s in segs if s.text.strip()); elapsed=time.perf_counter()-t
            rows.append({"transcription_ms":rnd(elapsed*1000),"audio_seconds":rnd(audio_s),
                         "realtime_factor":rnd(elapsed/audio_s,3),"text":text})
            print(f"  {i+1}/{repeats}: {rnd(elapsed*1000)} ms -> {text[:60]!r}")
    vals=[x["transcription_ms"] for x in rows]; rtfs=[x["realtime_factor"] for x in rows]
    return {"model":"small.en","device":device,"compute_type":compute,"load_seconds":rnd(load_s),
            "load_gpu":load_mon.summary(),"samples":rows,"summary":{
                "transcription_median_ms":rnd(statistics.median(vals)),"transcription_p95_ms":pct(vals,95),
                "realtime_factor_median":rnd(statistics.median(rtfs),3),**mon.summary()}}, stt


def transcribe(stt,path):
    t=time.perf_counter(); segs,_=stt.transcribe(str(path),language="en",vad_filter=True,beam_size=1)
    return " ".join(s.text.strip() for s in segs if s.text.strip()), time.perf_counter()-t


def benchmark_pipeline(model,stt,tts,paths,work,turns):
    print(f"\n[Full pipeline] STT -> {model} -> Chatterbox Nano"); ref=work/"reference.wav"; rows=[]
    with GPUMonitor() as mon:
        for i in range(turns):
            total=time.perf_counter(); text,stt_s=transcribe(stt,paths[i%len(paths)])
            llm=llm_turn(model,text or PROSPECT_LINES[i%len(PROSPECT_LINES)])
            answer=llm["response"] or "Thanks for your time."
            t=time.perf_counter(); wav=tts.generate(answer,audio_prompt_path=str(ref))
            if torch.cuda.is_available(): torch.cuda.synchronize()
            tts_s=time.perf_counter()-t; total_s=time.perf_counter()-total
            row={"turn":i+1,"transcript":text,"response":answer,"stt_ms":rnd(stt_s*1000),
                 "llm_ttft_ms":llm["ttft_ms"],"llm_wall_ms":llm["wall_ms"],
                 "llm_tokens_per_second":llm["tokens_per_second"],"tts_ms":rnd(tts_s*1000),
                 "total_response_ms":rnd(total_s*1000),"generated_audio_seconds":rnd(wav.shape[-1]/float(tts.sr))}
            rows.append(row); print(f"  {i+1}/{turns}: STT {row['stt_ms']} | LLM TTFT {row['llm_ttft_ms']} | TTS {row['tts_ms']} | TOTAL {row['total_response_ms']} ms")
    total=[x["total_response_ms"] for x in rows]
    return {"model":model,"turn_count":turns,"turns":rows,"summary":{
        "stt_median_ms":rnd(statistics.median([x["stt_ms"] for x in rows])),
        "llm_ttft_median_ms":rnd(statistics.median([x["llm_ttft_ms"] for x in rows])),
        "tts_median_ms":rnd(statistics.median([x["tts_ms"] for x in rows])),
        "total_median_ms":rnd(statistics.median(total)),"total_p95_ms":pct(total,95),"total_max_ms":rnd(max(total)),
        **mon.summary()}}


def render_html(r):
    hw=r["system"]; pipe=r.get("pipeline",{}); ps=pipe.get("summary",{}); gpu=(hw.get("gpu") or [{}])[0].get("name","CPU only")
    qrows="".join(f"<tr><td>{html.escape(m)}</td><td>{d.get('summary',{}).get('ttft_median_ms')}</td><td>{d.get('summary',{}).get('ttft_p95_ms')}</td><td>{d.get('summary',{}).get('tokens_per_second_median')}</td><td>{d.get('summary',{}).get('peak_gpu_memory_mb')}</td></tr>" for m,d in r.get("qwen",{}).items())
    prows="".join(f"<tr><td>{x['turn']}</td><td>{x['stt_ms']}</td><td>{x['llm_ttft_ms']}</td><td>{x['tts_ms']}</td><td>{x['total_response_ms']}</td><td>{html.escape(x['response'][:140])}</td></tr>" for x in pipe.get("turns",[]))
    return f'''<!doctype html><meta charset="utf-8"><title>Dialforge Benchmark</title><style>body{{font:14px system-ui;background:#101113;color:#f4f4f5;margin:32px;max-width:1150px}}h1{{font-size:34px}}.cards{{display:flex;gap:10px;flex-wrap:wrap}}.card,section{{background:#18181b;border:1px solid #303036;border-radius:14px;padding:16px;margin:14px 0}}.card{{min-width:180px}}strong{{font-size:22px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid #303036;text-align:left;vertical-align:top}}th{{color:#fb923c}}</style><h1>Dialforge Cloud Benchmark</h1><p>{hw['timestamp_utc']} · synthetic local AI pipeline · no SIP/PSTN</p><div class="cards"><div class="card">GPU<br><strong>{html.escape(str(gpu))}</strong></div><div class="card">RAM<br><strong>{hw['ram_total_gb']} GB</strong></div><div class="card">Pipeline median<br><strong>{ps.get('total_median_ms')} ms</strong></div><div class="card">Pipeline P95<br><strong>{ps.get('total_p95_ms')} ms</strong></div></div><section><h2>Qwen tiers</h2><table><tr><th>Model</th><th>Median TTFT ms</th><th>P95 TTFT ms</th><th>Median tok/s</th><th>Peak GPU MB</th></tr>{qrows}</table></section><section><h2>Full pipeline</h2><table><tr><th>Turn</th><th>STT ms</th><th>LLM TTFT ms</th><th>TTS ms</th><th>Total ms</th><th>Response</th></tr>{prows}</table></section><section><h2>Hardware</h2><pre>{html.escape(json.dumps(hw,indent=2))}</pre></section>'''


def main():
    p=argparse.ArgumentParser(); p.add_argument("--models",nargs="+",default=list(MODELS)); p.add_argument("--qwen-repeats",type=int,default=3); p.add_argument("--component-repeats",type=int,default=3); p.add_argument("--pipeline-turns",type=int,default=5); p.add_argument("--pipeline-model",default="qwen3:4b"); p.add_argument("--output-dir",default="dialforge-benchmark"); a=p.parse_args()
    out=Path(a.output_dir).resolve(); work=out/"work"; work.mkdir(parents=True,exist_ok=True)
    report={"schema":1,"product":"Dialforge","notes":["Synthetic benchmark: no LiveKit/SIP/PSTN network latency.","Qwen thinking disabled for latency-sensitive voice-agent use."],"system":system_info(),"qwen":{}}
    print("Dialforge Cloud Benchmark\n",json.dumps(report["system"],indent=2)); wait_ollama()
    for model in a.models:
        try: report["qwen"][model]=benchmark_qwen(model,a.qwen_repeats)
        except Exception as e: report["qwen"][model]={"error":f"{type(e).__name__}: {e}"}; print("ERROR",e)
    tts_report,tts=benchmark_tts(work,a.component_repeats); report["chatterbox"]=tts_report
    paths=prospect_audio(tts,work); stt_report,stt=benchmark_stt(paths,a.component_repeats); report["whisper"]=stt_report
    if a.pipeline_model not in a.models: pull(a.pipeline_model)
    report["pipeline"]=benchmark_pipeline(a.pipeline_model,stt,tts,paths,work,a.pipeline_turns)
    (out/"dialforge-benchmark-report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    (out/"dialforge-benchmark-report.html").write_text(render_html(report),encoding="utf-8")
    print("\nReports:",out/"dialforge-benchmark-report.html",out/"dialforge-benchmark-report.json")

if __name__=="__main__": main()
