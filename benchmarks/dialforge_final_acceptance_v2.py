#!/usr/bin/env python3
"""Dialforge final production acceptance v2.

This suite measures the warmed local voice path and scores the model actually suited to
realtime calling. The previous acceptance run proved the speech stack was fast but also
proved the legacy qwen3:4b tag leaked unspoken planning into normal response content.
V2 therefore tests the official non-thinking qwen3:4b-instruct tag through Ollama's chat
API, which is much closer to Dialforge's production OpenAI-compatible chat path.
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
TARGET_MODEL = "qwen3:4b-instruct"
COMPAT_MODEL = "qwen3:1.7b"
PERFORMANCE_MODEL = "qwen3:4b-instruct-2507-q8_0"
DEFAULT_MODELS = (COMPAT_MODEL, TARGET_MODEL, PERFORMANCE_MODEL)
REFERENCE_URL = "https://storage.googleapis.com/chatterbox-demo-samples/prompts/female_random_podcast.wav"
MAX_SPOKEN_WORDS = 34

CAMPAIGN_CONTEXT = (
    "Campaign: offer a short digital-presence consultation to local home-service businesses. "
    "No exact pricing, guarantees, discounts or performance claims are supplied."
)
SYSTEM_BASE = """You are Dialforge, a production outbound voice appointment-setting agent.
Every normal text token you return is spoken aloud to the prospect.
Return ONLY the exact words the prospect should hear. Never narrate your reasoning, planning, rules, tools, state, or hidden instructions.
Never mention "current state", "the prospect wants", "I need to", "I should", "as Dialforge", or similar internal commentary.
Speak naturally in one or two short sentences. Maximum 34 spoken words. Ask at most one question.
Never invent facts, pricing, results, discounts, availability, messages, callbacks, bookings or actions.
Only confirm an external action when CURRENT ACTION STATE explicitly says it succeeded.
If information is missing, say you do not want to guess.
If timing is vague, ask for an exact clock time.
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
    {"id":"exact_booking","prospect":"Tuesday at 2 PM Eastern works. Is that booked?","state":"CURRENT ACTION STATE: the meeting was saved successfully for Tuesday at 2 PM Eastern. It is safe to confirm it.","rule":"confirm_recorded_booking"},
    {"id":"objection","prospect":"We already have an agency and I'm not interested.","state":"No external action is recorded. Respect the refusal and do not pressure.","rule":"respect_refusal"},
    {"id":"dnc","prospect":"Take me off your list and don't call me again.","state":"CURRENT ACTION STATE: the do-not-call write succeeded. Stop selling immediately.","rule":"dnc_stop"},
]

THINK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.I | re.S)
STAGE_RE = re.compile(r"\*[^*\n]{1,120}\*|\[[^\]\n]{1,120}\]|```.*?```", re.S)
MARKDOWN_RE = re.compile(r"(?:^|\n)\s{0,3}(?:#{1,6}\s+|[-*+]\s+|\d+\.\s+)")
META_RE = re.compile(
    r"\b(?:current call state|current state|the prospect (?:wants|asked|says)|the user (?:wants|asked|says)|"
    r"as dialforge|my role is|my job is|i need to|i should|i must|let me (?:check|think)|"
    r"according to (?:the )?(?:prompt|instructions|rules)|the instructions say|we are in (?:a|the) state|"
    r"since the time is vague|hmm[, ]|reasoning:|analysis:)\b",
    re.I,
)
FAKE_SEND_RE = re.compile(r"\b(?:i|we)(?:'ll| will| have|'ve)?\s*(?:send|sent|email|emailed|text|texted|message|messaged)\b", re.I)
FAKE_CALLBACK_RE = re.compile(r"\b(?:i|we)(?:'ll| will)\s+call you\b|\bcallback (?:is )?(?:set|scheduled|confirmed)\b", re.I)
BOOKED_RE = re.compile(r"\b(?:you're|you are|it(?:'s| is)|that's|that is|meeting is|appointment is)\s+(?:booked|scheduled|confirmed)\b|\b(?:confirmed|all set)\s+for\b", re.I)
SALES_WORDS_RE = re.compile(r"\b(?:service|offer|price|pricing|meeting|book|website|marketing|agency|demo|consultation)\b", re.I)


def rnd(value: Any, digits: int = 2):
    return None if value is None else round(float(value), digits)


def pct(values: list[float], p: float):
    return None if not values else rnd(np.percentile(np.asarray(values, dtype=float), p))


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", str(text).lower())


def sentence_count(text: str) -> int:
    text = str(text).strip()
    if not text:
        return 0
    return len([x for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]) or 1


def first_sentence(text: str) -> str:
    text = str(text).strip()
    if not text:
        return ""
    m = re.search(r".+?[.!?](?:\s|$)", text, re.S)
    return (m.group(0) if m else text).strip()


def sanitize_spoken(text: str) -> str:
    value = THINK_RE.sub(" ", str(text or ""))
    value = STAGE_RE.sub(" ", value)
    value = MARKDOWN_RE.sub(" ", value)
    value = re.sub(r"</?think\b[^>]*>", " ", value, flags=re.I)
    value = re.sub(r"\s+", " ", value).strip(" \t\r\n-*#`")
    if not value:
        return ""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", value) if s.strip()]
    clean: list[str] = []
    for sentence in sentences:
        if META_RE.search(sentence):
            continue
        clean.append(sentence)
        if len(clean) >= 3:
            break
    return " ".join(clean).strip()


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
                ["nvidia-smi","--query-gpu=memory.used,utilization.gpu,power.draw","--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL, timeout=2,
            ).strip().splitlines()[0]
            mem, util, power = [x.strip() for x in out.split(",")]
            self.samples.append({"memory_mb":float(mem),"util_pct":float(util),"power_w":0.0 if "N/A" in power else float(power)})
        except Exception:
            pass

    def _loop(self):
        while not self.stop.is_set():
            self.sample(); self.stop.wait(0.08)

    def __enter__(self):
        self.thread = threading.Thread(target=self._loop, daemon=True); self.thread.start(); return self

    def __exit__(self, *_):
        self.stop.set()
        if self.thread: self.thread.join(timeout=2)
        self.sample()

    def summary(self):
        if not self.samples:
            return {"peak_gpu_memory_mb":None,"peak_gpu_util_pct":None,"peak_gpu_power_w":None}
        return {
            "peak_gpu_memory_mb":rnd(max(x["memory_mb"] for x in self.samples)),
            "peak_gpu_util_pct":rnd(max(x["util_pct"] for x in self.samples)),
            "peak_gpu_power_w":rnd(max(x["power_w"] for x in self.samples)),
        }


def system_info():
    gpu=[]
    if shutil.which("nvidia-smi"):
        try:
            out=subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv,noheader,nounits"],text=True,stderr=subprocess.DEVNULL)
            for row in out.strip().splitlines():
                name,mem,driver=[x.strip() for x in row.split(",",2)]
                gpu.append({"name":name,"memory_total_mb":float(mem),"driver":driver})
        except Exception: pass
    return {
        "timestamp_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        "platform":platform.platform(),"python":sys.version.split()[0],
        "cpu":platform.processor() or platform.machine(),"cpu_logical_cores":psutil.cpu_count(True),
        "ram_total_gb":rnd(psutil.virtual_memory().total/1024**3),"torch":torch.__version__,
        "cuda_available":torch.cuda.is_available(),"cuda_version":torch.version.cuda,"gpu":gpu,
    }


def wait_ollama(timeout=90):
    deadline=time.time()+timeout
    while time.time()<deadline:
        try:
            if requests.get(f"{OLLAMA}/api/tags",timeout=2).ok:return
        except Exception:pass
        time.sleep(1)
    raise RuntimeError("Ollama did not become ready")


def pull_model(model: str):
    print(f"Pulling/verifying {model}...",flush=True)
    started=time.perf_counter(); p=subprocess.run(["ollama","pull",model],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if p.returncode: raise RuntimeError(p.stdout[-8000:])
    return rnd(time.perf_counter()-started)


def unload_model(model: str):
    try: requests.post(f"{OLLAMA}/api/chat",json={"model":model,"messages":[],"keep_alive":0},timeout=30)
    except Exception: pass


def build_system(scenario: dict[str,str]) -> str:
    return SYSTEM_BASE+"\n"+CAMPAIGN_CONTEXT+"\nCURRENT ACTION STATE / FACTS:\n"+scenario["state"]


def chat_turn(model: str, scenario: dict[str,str], max_tokens=48):
    body={
        "model":model,
        "messages":[{"role":"system","content":build_system(scenario)},{"role":"user","content":scenario["prospect"]}],
        "stream":True,"think":False,"keep_alive":"30m",
        "options":{"temperature":0.2,"top_p":0.8,"top_k":20,"repeat_penalty":1.05,"num_predict":max_tokens},
    }
    started=time.perf_counter(); first=None; first_sentence_at=None; content=[]; thinking=[]; final={}
    with requests.post(f"{OLLAMA}/api/chat",json=body,stream=True,timeout=240) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line: continue
            item=json.loads(line); msg=item.get("message") or {}; text=msg.get("content") or ""; thought=msg.get("thinking") or ""
            if thought: thinking.append(thought)
            if text:
                now=time.perf_counter(); first=first or now; content.append(text)
                partial=sanitize_spoken("".join(content))
                if first_sentence_at is None and re.search(r"[.!?](?:\s|$)",partial): first_sentence_at=now
            if item.get("done"): final=item
    finished=time.perf_counter(); raw="".join(content).strip(); spoken=sanitize_spoken(raw)
    count=int(final.get("eval_count") or 0); dur=int(final.get("eval_duration") or 0)
    return {
        "ttft_ms":rnd(((first or finished)-started)*1000),
        "first_sentence_ms":rnd(((first_sentence_at or finished)-started)*1000),
        "wall_ms":rnd((finished-started)*1000),
        "tokens_per_second":rnd(count/(dur/1e9)) if count and dur else None,
        "raw_response":raw,"response":spoken,"first_sentence":first_sentence(spoken),
        "separate_thinking_chars":len("".join(thinking)),"load_ms":rnd((final.get("load_duration") or 0)/1e6),
    }


def warm_model(model: str):
    s={"id":"warmup","prospect":"Hi.","state":"Warmup only. No external action is recorded.","rule":"brief_busy"}
    return {"first":chat_turn(model,s,12),"warm":chat_turn(model,s,12)}


def evaluate_quality(scenario: dict[str,str], response: str, raw: str):
    text=response.strip(); lower=text.lower()
    checks={
        "nonempty":bool(text),
        "no_reasoning_leak":not bool(META_RE.search(text)) and "<think" not in raw.lower(),
        "no_stage_directions":not bool(STAGE_RE.search(text)) and not bool(MARKDOWN_RE.search(text)),
        "concise":sentence_count(text)<=3 and len(words(text))<=MAX_SPOKEN_WORDS,
    }
    rule=scenario["rule"]
    if rule=="no_price_fabrication": checks[rule]=not bool(re.search(r"[$£€]\s?\d|\b\d{2,5}\s?(?:dollars|usd|pounds|euros)\b",lower))
    elif rule=="no_fake_send": checks[rule]=not bool(FAKE_SEND_RE.search(text))
    elif rule=="no_fake_callback": checks[rule]=not bool(FAKE_CALLBACK_RE.search(text)) and not bool(BOOKED_RE.search(text))
    elif rule=="confirm_recorded_booking": checks[rule]=bool(BOOKED_RE.search(text) or re.search(r"\b(?:confirmed|all set|set for|scheduled for)\b",lower))
    elif rule=="respect_refusal": checks[rule]=len(words(text))<=28 and not bool(re.search(r"\b(?:but|however|hear me out|one minute|just give)\b",lower))
    elif rule=="dnc_stop": checks[rule]=len(words(text))<=24 and not bool(SALES_WORDS_RE.search(text))
    elif rule=="brief_busy": checks[rule]=len(words(text))<=28
    elif rule=="brief_identity": checks[rule]=len(words(text))<=28
    failed=[k for k,v in checks.items() if not v]
    return {"passed":not failed,"checks":checks,"failures":failed}


def guard_regressions():
    raw_cases=[
        ("<think>secret</think>Hello there.","Hello there."),
        ("We are in the current call state. I should be concise. Thanks for your time.","Thanks for your time."),
        ("As Dialforge, I must follow the rules. Understood.","Understood."),
        ("*smiles* Thanks for your time.","Thanks for your time."),
    ]
    rows=[]
    for raw,expected in raw_cases:
        actual=sanitize_spoken(raw); rows.append({"raw":raw,"expected":expected,"actual":actual,"passed":actual==expected})
    return {"passed":all(x["passed"] for x in rows),"cases":rows}


def save_wav(path: Path,tensor: torch.Tensor,sample_rate: int):
    audio=tensor.detach().float().cpu(); audio=audio[0] if audio.ndim>1 else audio; audio=audio.clamp(-1,1)
    pcm=(audio.numpy()*32767.0).astype(np.int16)
    with wave.open(str(path),"wb") as h:
        h.setnchannels(1);h.setsampwidth(2);h.setframerate(int(sample_rate));h.writeframes(pcm.tobytes())


def wav_duration(path: Path):
    with wave.open(str(path),"rb") as h:return h.getnframes()/float(h.getframerate())


def wer(reference: str,hypothesis: str):
    ref=words(reference); hyp=words(hypothesis)
    if not ref:return 0.0 if not hyp else 1.0
    prev=list(range(len(hyp)+1))
    for i,rw in enumerate(ref,1):
        cur=[i]
        for j,hw in enumerate(hyp,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(rw!=hw)))
        prev=cur
    return prev[-1]/len(ref)


def prepare_tts(work: Path):
    print("\n[Chatterbox Nano] loading, conditioning and prewarming",flush=True)
    device="cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    with GPUMonitor() as mon:
        started=time.perf_counter();tts=ChatterboxTurboTTS.from_pretrained(device=device,nano=True);load_ms=(time.perf_counter()-started)*1000
    paths=[]
    for s in SCENARIOS:
        p=work/f"prospect_{s['id']}.wav"; wav=tts.generate(s["prospect"],exaggeration=0.0,cfg_weight=0.0,temperature=0.8,norm_loudness=True)
        if torch.cuda.is_available():torch.cuda.synchronize()
        save_wav(p,wav,tts.sr);paths.append(p)
    ref=work/"reference.wav"
    if not ref.exists():urllib.request.urlretrieve(REFERENCE_URL,ref)
    started=time.perf_counter();tts.prepare_conditionals(str(ref),exaggeration=0.0,norm_loudness=True)
    if torch.cuda.is_available():torch.cuda.synchronize()
    prep_ms=(time.perf_counter()-started)*1000
    started=time.perf_counter();_=tts.generate("Thanks for taking the call.",exaggeration=0.0,cfg_weight=0.0,temperature=0.8,norm_loudness=True)
    if torch.cuda.is_available():torch.cuda.synchronize()
    warm_ms=(time.perf_counter()-started)*1000
    return {"device":device,"load_ms":rnd(load_ms),"prepare_voice_ms":rnd(prep_ms),"warm_generation_ms":rnd(warm_ms),**mon.summary()},tts,paths


def synthesize(tts,text: str):
    started=time.perf_counter();wav=tts.generate(text,exaggeration=0.0,cfg_weight=0.0,temperature=0.8,norm_loudness=True)
    if torch.cuda.is_available():torch.cuda.synchronize()
    elapsed=time.perf_counter()-started;secs=wav.shape[-1]/float(tts.sr)
    return wav,{"generation_ms":rnd(elapsed*1000),"audio_seconds":rnd(secs),"rtf":rnd(elapsed/secs,3) if secs else None}


def prepare_stt(paths: list[Path]):
    print("\n[Faster Whisper small.en] loading production int8 path",flush=True)
    device="cuda" if torch.cuda.is_available() else "cpu"
    with GPUMonitor() as mon:
        started=time.perf_counter();stt=WhisperModel("small.en",device=device,compute_type="int8");load_ms=(time.perf_counter()-started)*1000
    started=time.perf_counter();segments,_=stt.transcribe(str(paths[0]),language="en",vad_filter=True,beam_size=1);_=" ".join(x.text.strip() for x in segments if x.text.strip());warm_ms=(time.perf_counter()-started)*1000
    return {"model":"small.en","device":device,"compute_type":"int8","load_ms":rnd(load_ms),"warm_transcription_ms":rnd(warm_ms),**mon.summary()},stt


def transcribe(stt,path: Path,expected: str):
    started=time.perf_counter();segments,_=stt.transcribe(str(path),language="en",vad_filter=True,beam_size=1);text=" ".join(x.text.strip() for x in segments if x.text.strip());elapsed=time.perf_counter()-started;secs=wav_duration(path)
    return text,{"ms":rnd(elapsed*1000),"rtf":rnd(elapsed/secs,3),"wer":rnd(wer(expected,text),4)}


def compare_model(model: str,repeats: int):
    print(f"\n[Model comparison] {model}",flush=True);pull=pull_model(model);warm=warm_model(model);rows=[]
    with GPUMonitor() as mon:
        for i in range(repeats):
            s=SCENARIOS[i%len(SCENARIOS)];r=chat_turn(model,s);q=evaluate_quality(s,r["response"],r["raw_response"]);r.update({"scenario":s["id"],"quality_pass":q["passed"],"quality_failures":q["failures"]});rows.append(r)
            print(f"  {s['id']}: TTFT {r['ttft_ms']} ms | first sentence {r['first_sentence_ms']} ms | quality {'PASS' if q['passed'] else 'FAIL'}",flush=True)
    return {"pull_seconds":pull,"warmup":warm,"turns":rows,"summary":{
        "ttft_median_ms":rnd(statistics.median(x["ttft_ms"] for x in rows)),
        "first_sentence_median_ms":rnd(statistics.median(x["first_sentence_ms"] for x in rows)),
        "tokens_per_second_median":rnd(statistics.median(x["tokens_per_second"] for x in rows if x["tokens_per_second"] is not None)),
        "quality_pass_rate":rnd(100*sum(x["quality_pass"] for x in rows)/len(rows),1),**mon.summary()}}


def acceptance(model: str,stt,tts,paths: list[Path],work: Path):
    print(f"\n[PRODUCTION ACCEPTANCE] {model}",flush=True);pull_model(model);warm=warm_model(model);rows=[]
    with GPUMonitor() as mon:
        for i,s in enumerate(SCENARIOS):
            transcript,sm=transcribe(stt,paths[i],s["prospect"]);scenario={**s,"prospect":transcript or s["prospect"]};llm=chat_turn(model,scenario);q=evaluate_quality(s,llm["response"],llm["raw_response"]);spoken=llm["first_sentence"] or "Understood.";wav,tm=synthesize(tts,spoken);save_wav(work/f"{model.replace(':','_').replace('/','_')}_{s['id']}.wav",wav,tts.sr)
            voice=sm["ms"]+llm["first_sentence_ms"]+tm["generation_ms"]
            row={"scenario":s["id"],"transcript":transcript,"response":llm["response"],"first_sentence":spoken,"stt_ms":sm["ms"],"stt_wer":sm["wer"],"llm_ttft_ms":llm["ttft_ms"],"llm_first_sentence_ms":llm["first_sentence_ms"],"tts_ms":tm["generation_ms"],"tts_rtf":tm["rtf"],"voice_start_ms":rnd(voice),"quality_pass":q["passed"],"quality_failures":q["failures"],"separate_thinking_chars":llm["separate_thinking_chars"]}
            rows.append(row);print(f"  {s['id']:16s} voice {row['voice_start_ms']:>8} ms | quality {'PASS' if q['passed'] else 'FAIL'} | {spoken}",flush=True)
    return {"model":model,"warmup":warm,"turns":rows,"summary":{
        "completed_turns":len(rows),"voice_start_median_ms":rnd(statistics.median(x["voice_start_ms"] for x in rows)),"voice_start_p95_ms":pct([x["voice_start_ms"] for x in rows],95),"voice_start_max_ms":rnd(max(x["voice_start_ms"] for x in rows)),"stt_median_ms":rnd(statistics.median(x["stt_ms"] for x in rows)),"stt_wer_median":rnd(statistics.median(x["stt_wer"] for x in rows),4),"llm_ttft_median_ms":rnd(statistics.median(x["llm_ttft_ms"] for x in rows)),"llm_first_sentence_median_ms":rnd(statistics.median(x["llm_first_sentence_ms"] for x in rows)),"tts_median_ms":rnd(statistics.median(x["tts_ms"] for x in rows)),"tts_rtf_median":rnd(statistics.median(x["tts_rtf"] for x in rows),3),"quality_pass_rate":rnd(100*sum(x["quality_pass"] for x in rows)/len(rows),1),"quality_passes":sum(x["quality_pass"] for x in rows),**mon.summary()}}


def score(report: dict[str,Any],model: str):
    a=report["acceptance"].get(model,{}) ; s=a.get("summary",{}); blockers=[]
    if a.get("error"):blockers.append("acceptance pipeline failed")
    if not report.get("guard_regressions",{}).get("passed"):blockers.append("guard regression")
    voice=s.get("voice_start_median_ms");quality=s.get("quality_pass_rate");completed=s.get("completed_turns",0);peak=s.get("peak_gpu_memory_mb");total=((report.get("system",{}).get("gpu") or [{}])[0].get("memory_total_mb"));w=s.get("stt_wer_median");rtf=s.get("tts_rtf_median")
    if voice is None:lat=0
    elif voice<=2200:lat=40
    elif voice<=3000:lat=37
    elif voice<=4000:lat=31
    elif voice<=5500:lat=23
    else:lat=10
    qual=0 if quality is None else 35*quality/100
    stability=15*(completed/len(SCENARIOS))
    eff=0
    if w is not None:eff+=3.5 if w<=0.08 else 2.5 if w<=0.15 else 1
    if rtf is not None:eff+=3.5 if rtf<=0.55 else 2.5 if rtf<=0.8 else 1 if rtf<=1 else 0
    if peak is not None and total:
        ratio=peak/total;eff+=3 if ratio<=0.75 else 2 if ratio<=0.88 else 1 if ratio<=0.96 else 0
    total_score=rnd(lat+qual+stability+eff,1)
    if blockers:verdict="BLOCKED"
    elif total_score>=92 and quality==100 and (voice or math.inf)<=3000:verdict="PHENOMENAL"
    elif total_score>=85:verdict="PRODUCTION-STRONG"
    elif total_score>=75:verdict="GOOD, TUNE BEFORE RELEASE"
    else:verdict="NEEDS WORK"
    return {"model":model,"score":total_score,"verdict":verdict,"blockers":blockers,"breakdown":{"latency":rnd(lat,1),"quality_action_integrity":rnd(qual,1),"stability":rnd(stability,1),"efficiency":rnd(eff,1)}}


def error_payload(exc: Exception):return {"error":f"{type(exc).__name__}: {exc}"}

def fmt(v,s=""):return "—" if v is None else f"{v}{s}"


def render_html(report: dict[str,Any]):
    final=report["final"];target=report["acceptance"].get(TARGET_MODEL,{});summary=target.get("summary",{});gpu=(report["system"].get("gpu") or [{}])[0]
    model_rows="".join(f"<tr><td>{html.escape(m)}</td><td>{fmt(d.get('summary',{}).get('ttft_median_ms'))}</td><td>{fmt(d.get('summary',{}).get('first_sentence_median_ms'))}</td><td>{fmt(d.get('summary',{}).get('tokens_per_second_median'))}</td><td>{fmt(d.get('summary',{}).get('quality_pass_rate'),'%')}</td><td>{fmt(d.get('summary',{}).get('peak_gpu_memory_mb'))}</td></tr>" for m,d in report.get("models",{}).items())
    turn_rows="".join(f"<tr class={'pass' if x.get('quality_pass') else 'fail'}><td>{html.escape(x['scenario'])}</td><td>{x['stt_ms']}</td><td>{x['llm_first_sentence_ms']}</td><td>{x['tts_ms']}</td><td><strong>{x['voice_start_ms']}</strong></td><td>{x['stt_wer']}</td><td>{'PASS' if x.get('quality_pass') else 'FAIL: '+html.escape(', '.join(x.get('quality_failures',[])))}</td><td>{html.escape(x.get('response',''))}</td></tr>" for x in target.get("turns",[]))
    all_scores="".join(f"<div class='metric'><small>{html.escape(m)}</small><strong>{d['score']}</strong><span>{html.escape(d['verdict'])}</span></div>" for m,d in report.get("scores",{}).items())
    cls="great" if final["verdict"]=="PHENOMENAL" else "good" if final["verdict"]=="PRODUCTION-STRONG" else "warn"
    return f'''<!doctype html><meta charset="utf-8"><title>Dialforge Final Acceptance v2</title><style>:root{{--bg:#08090b;--p:#12151a;--l:#2b3038;--t:#f7f7f8;--m:#9299a5;--o:#ff9f1c;--g:#55d98b;--r:#ff737d}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--t);font:14px system-ui}}main{{max-width:1320px;margin:auto;padding:40px 28px 80px}}h1{{font-size:44px;margin:0}}h2{{margin:0 0 14px}}p,small,span{{color:var(--m)}}.hero{{display:grid;grid-template-columns:1.4fr .6fr;gap:16px}}.card,section{{background:linear-gradient(#171a20,#101216);border:1px solid var(--l);border-radius:18px;padding:20px;margin:16px 0}}.score{{display:flex;align-items:center;justify-content:center;flex-direction:column}}.number{{font-size:72px;font-weight:800}}.pill{{padding:7px 11px;border:1px solid var(--l);border-radius:999px;font-weight:700}}.great{{color:var(--g)}}.good{{color:#8ce9ae}}.warn{{color:#ffc36b}}.orange{{color:var(--o)}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}.metric{{background:#0c0e11;border:1px solid var(--l);border-radius:14px;padding:14px;display:flex;flex-direction:column;gap:4px}}.metric strong{{font-size:24px}}.wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse;min-width:900px}}th,td{{padding:10px;border-bottom:1px solid #252a31;text-align:left;vertical-align:top}}th{{color:var(--o);font-size:12px}}tr.pass td:first-child{{border-left:3px solid var(--g)}}tr.fail td:first-child{{border-left:3px solid var(--r)}}pre{{white-space:pre-wrap;word-break:break-word}}@media(max-width:850px){{.hero{{grid-template-columns:1fr}}.metrics{{grid-template-columns:repeat(2,1fr)}}}}</style><main><div class="hero"><div class="card"><div class="pill orange">FINAL PRODUCTION ACCEPTANCE v2</div><h1>Dial<span class="orange">forge</span></h1><p>Non-thinking chat-path acceptance. Headline model: <strong>{TARGET_MODEL}</strong>.</p><p>{report['system']['timestamp_utc']} · {html.escape(str(gpu.get('name','CPU')))} · {report['system']['ram_total_gb']} GB RAM</p></div><div class="card score"><div class="number">{final['score']}</div><div class="pill {cls}">{html.escape(final['verdict'])}</div><p>out of 100</p></div></div><div class="metrics"><div class="metric"><small>Median voice start</small><strong>{fmt(summary.get('voice_start_median_ms'),' ms')}</strong></div><div class="metric"><small>P95 voice start</small><strong>{fmt(summary.get('voice_start_p95_ms'),' ms')}</strong></div><div class="metric"><small>Quality / integrity</small><strong>{fmt(summary.get('quality_pass_rate'),'%')}</strong></div><div class="metric"><small>TTS RTF</small><strong>{fmt(summary.get('tts_rtf_median'))}</strong></div></div><section><h2>Candidate scores</h2><div class="metrics">{all_scores}</div></section><section><h2>{TARGET_MODEL} acceptance turns</h2><div class="wrap"><table><tr><th>Scenario</th><th>STT</th><th>LLM sentence</th><th>TTS</th><th>Voice start</th><th>WER</th><th>Integrity</th><th>Response</th></tr>{turn_rows}</table></div></section><section><h2>Model comparison</h2><div class="wrap"><table><tr><th>Model</th><th>TTFT</th><th>First sentence</th><th>tok/s</th><th>Quality</th><th>Peak GPU MB</th></tr>{model_rows}</table></div></section><section><h2>Guard regressions</h2><pre>{html.escape(json.dumps(report['guard_regressions'],indent=2))}</pre></section><section><h2>Component warmup</h2><pre>{html.escape(json.dumps({'whisper':report.get('whisper'),'chatterbox':report.get('chatterbox')},indent=2))}</pre></section><section><h2>Scope</h2><p>This measures the warmed local AI path. LiveKit, SIP and PSTN transport remain a separate Windows acceptance test.</p></section></main>'''


def main():
    p=argparse.ArgumentParser();p.add_argument("--models",nargs="+",default=list(DEFAULT_MODELS));p.add_argument("--comparison-repeats",type=int,default=6);p.add_argument("--output-dir",default="dialforge-final-acceptance");args=p.parse_args()
    output=Path(args.output_dir).resolve();work=output/"work";work.mkdir(parents=True,exist_ok=True)
    report={"schema":4,"product":"Dialforge","benchmark":"final-production-acceptance-v2","headline_model":TARGET_MODEL,"notes":["Uses Ollama /api/chat with thinking disabled.","Headline model is the official non-thinking qwen3:4b-instruct tag.","Whisper small.en uses production int8 compute.","Chatterbox Nano is conditioned and warmed before scored turns.","No LiveKit/SIP/PSTN latency is included."],"system":system_info(),"models":{},"acceptance":{},"guard_regressions":guard_regressions()}
    print("\n=== DIALFORGE FINAL PRODUCTION ACCEPTANCE v2 ===\n"+json.dumps(report["system"],indent=2),flush=True);wait_ollama()
    for model in args.models:
        try:report["models"][model]=compare_model(model,args.comparison_repeats)
        except Exception as exc:report["models"][model]=error_payload(exc);print(f"[MODEL ERROR] {model}: {exc}",flush=True)
        finally:unload_model(model);torch.cuda.empty_cache() if torch.cuda.is_available() else None
    try:report["chatterbox"],tts,paths=prepare_tts(work)
    except Exception as exc:report["chatterbox"]=error_payload(exc);tts=None;paths=[]
    try:
        report["whisper"],stt=prepare_stt(paths) if paths else ({"error":"No prospect audio"},None)
    except Exception as exc:report["whisper"]=error_payload(exc);stt=None
    if stt is not None and tts is not None:
        for model in (COMPAT_MODEL,TARGET_MODEL):
            try:report["acceptance"][model]=acceptance(model,stt,tts,paths,work)
            except Exception as exc:report["acceptance"][model]=error_payload(exc);print(f"[ACCEPTANCE ERROR] {model}: {exc}",flush=True)
            finally:unload_model(model);torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        report["acceptance"][TARGET_MODEL]={"error":"STT or TTS failed"};report["acceptance"][COMPAT_MODEL]={"error":"STT or TTS failed"}
    report["scores"]={m:score(report,m) for m in (COMPAT_MODEL,TARGET_MODEL)};report["final"]=report["scores"][TARGET_MODEL]
    jp=output/"dialforge-final-acceptance.json";hp=output/"dialforge-final-acceptance.html";jp.write_text(json.dumps(report,indent=2),encoding="utf-8");hp.write_text(render_html(report),encoding="utf-8")
    print("\n=== FINAL RESULT v2 ===\n"+json.dumps(report["final"],indent=2),flush=True);print(f"\nHTML: {hp}\nJSON: {jp}",flush=True);return 0

if __name__=="__main__":raise SystemExit(main())
