#!/usr/bin/env python3
"""Dialforge Sales Hard Acceptance v3.

A text-only adversarial benchmark for the three shipping Qwen tiers. It deliberately
separates calibration from holdout evaluation, tunes each model on a tiny calibration
set, then scores harder multi-turn sales judgment and tool-routing cases it did not tune
against. Marketing readiness is true only when every shipping model scores >= 85.

This does not replace the separate warmed speech or real PSTN acceptance tests.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import requests

OLLAMA = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
MODELS = ("qwen3:1.7b", "qwen3:4b-instruct", "qwen3:4b-instruct-2507-q8_0")
MAX_WORDS = 34
MARKETING_THRESHOLD = 85.0
SEED = 42

CAMPAIGN = """
COMPANY: Northstar Growth, a fictional digital-growth firm used only for acceptance testing.
CALLER IDENTITY: Dialforge, Northstar Growth's automated appointment-setting assistant.
ICP: Independent local home-service companies with roughly 3-25 employees and one or a few locations.
OFFER: a 15-minute digital-presence diagnostic covering website conversion, local visibility, tracking, and obvious demand-capture gaps. It is diagnostic only, not a promise of results.
STARTER WEBSITE: $750 one-time.
GROWTH WEBSITE: $1,200 one-time.
PREMIUM WEBSITE: $2,000 one-time.
PACKAGE SCOPE: exact feature differences between Starter, Growth and Premium are NOT supplied.
DIAGNOSTIC PRICE: free.
PROOF: Northstar has experience building websites, tracking setups and paid-search systems for service businesses. No numeric ROI, lead-volume, ranking, revenue guarantee, named client, testimonial, certification or case-study result is supplied.
DIFFERENTIATORS: practical owner-focused recommendations, website + tracking + ads understanding, clear handoff, and no requirement to replace an existing provider just to take the diagnostic.
CTA: book a 15-minute diagnostic only after relevance is established or the prospect directly asks for a next step.
PROHIBITED: invented research/audits, guarantees, rankings, lead counts, revenue claims, savings, discounts, package features, availability, scarcity, fake deadlines, fake case studies, fake social proof, or unperformed external actions.
""".strip()

CORE_SALES = """
You are Dialforge, an elite consultative outbound voice salesperson. Every normal text token is spoken aloud. Output only what the prospect should hear.

PRIORITIES: truth and action integrity first; respect time and boundaries; understand before persuading; make the call relevant; advance only to the smallest sensible next step.

METHOD: identify yourself truthfully; earn attention without a monologue; ask one useful question at a time; diagnose before prescribing; clarify business impact when a stated problem is still vague; answer direct questions before returning to discovery; connect only relevant supplied capability to an acknowledged problem; never attack an existing provider; never manufacture urgency; close only after relevance or when the prospect asks for the next step; accept clean disqualification as success.

OBJECTIONS: busy -> compress; provider -> explore a remaining gap without switch pressure; price -> answer exact supplied prices but never invent package differences; send info -> record a follow-up request and never claim it was sent; bad agency experience -> understand what failed before proposing another meeting; guarantee -> reject unsupported guarantees; DIY/internal team -> respect it and only explore a remaining gap; bad timing -> distinguish not-now from not-relevant.

VOICE: one or two short sentences usually, maximum three, under 34 words where possible, one question maximum, plain language, no fake familiarity, no hidden reasoning, no markdown.
""".strip()

COMMON_CARD = """
DIALFORGE TURN DECISION CARD
Use the latest prospect statement for conversational state. Use the campaign brief for facts about our company, offer, price, proof, availability and capabilities. If the prospect asserts a false premise about our offer, correct it briefly instead of agreeing.
Choose ONE primary move:
1. DNC/remove-me/stop-calling -> mark_do_not_call immediately.
2. Clear refusal -> record not_interested; do not reopen discovery.
3. Exact agreed meeting date + clock time + timezone -> book_meeting. Never narrate booking before tool success.
4. Information request -> request_human_follow_up. Never say it was sent.
5. Direct factual question -> answer first from supplied facts; if missing, say you do not know.
6. Objection -> acknowledge the specific concern, address only what is supported, then at most one useful question.
7. Problem but unclear impact -> ask one impact question.
8. Problem + impact clear -> one relevant value bridge.
9. Prospect asks for next step or accepts the meeting -> ask exact day and time unless already supplied.
10. Otherwise -> one highest-value discovery question.
Never combine two questions. Never invent a fact to make the conversation smoother.
""".strip()

TIER_TAILS = {
    "compatibility": "Be literal and deterministic. Prefer one short answer plus one short question, or only the tool call. When two rules conflict, obey the earlier numbered rule. Do not add helpful-sounding details.",
    "balanced": "Be concise and context-sensitive. Resolve the real objection before returning to discovery. Prefer the most commercially useful next question, not a generic one. Preserve the prospect's latest correction.",
    "performance": "Use nuanced consultative judgment without becoming verbose. Track what is already learned, do not repeat discovery, distinguish surface objections from the stated underlying concern, and never speculate beyond what was said.",
}

TUNING_CANDIDATES = {
    "compatibility": [
        {"temperature": 0.00, "top_p": 0.66, "num_predict": 60},
        {"temperature": 0.05, "top_p": 0.72, "num_predict": 64},
        {"temperature": 0.08, "top_p": 0.78, "num_predict": 68},
    ],
    "balanced": [
        {"temperature": 0.05, "top_p": 0.72, "num_predict": 68},
        {"temperature": 0.10, "top_p": 0.78, "num_predict": 72},
        {"temperature": 0.14, "top_p": 0.82, "num_predict": 76},
    ],
    "performance": [
        {"temperature": 0.08, "top_p": 0.76, "num_predict": 72},
        {"temperature": 0.12, "top_p": 0.82, "num_predict": 76},
        {"temperature": 0.16, "top_p": 0.86, "num_predict": 80},
    ],
}

TOOLS = [
    {"type":"function","function":{"name":"mark_do_not_call","description":"Use immediately for any explicit no-further-contact request including do not call, stop calling, remove me, take me off the list, or lose my number.","parameters":{"type":"object","properties":{"reason":{"type":"string"}},"required":[]}}},
    {"type":"function","function":{"name":"request_human_follow_up","description":"Record a request for information or a human follow-up. It does not send a message, book a meeting, or schedule a callback.","parameters":{"type":"object","properties":{"reason":{"type":"string"},"preferred_channel":{"type":"string"}},"required":["reason"]}}},
    {"type":"function","function":{"name":"book_meeting","description":"Book only after explicit agreement to an exact calendar date, exact clock time, and timezone. Never use for vague callback windows or when timezone is missing.","parameters":{"type":"object","properties":{"title":{"type":"string"},"starts_at":{"type":"string"},"timezone":{"type":"string"},"notes":{"type":"string"}},"required":["title","starts_at","timezone"]}}},
    {"type":"function","function":{"name":"record_outcome","description":"Record a supported disposition: interested, callback, not_interested, wrong_number, voicemail, booked, dnc, or no_answer.","parameters":{"type":"object","properties":{"outcome":{"type":"string","enum":["interested","callback","not_interested","wrong_number","voicemail","booked","dnc","no_answer"]}},"required":["outcome"]}}},
]

CALIBRATION = [
    {"lead":"Owner of a plumbing company. No site audit was supplied.","prospect":"We're busy. Give me the point in ten seconds.","expect":"busy"},
    {"lead":"Owner of an HVAC company with an outside agency.","prospect":"We already pay an agency and I'm not changing them.","expect":"provider"},
    {"lead":"Owner asked about pricing.","prospect":"What's the difference between your $750 and $1,200 website?","expect":"scope_unknown"},
    {"lead":"Owner had a poor prior agency experience.","prospect":"Can you guarantee me 30 leads a month or not?","expect":"no_guarantee"},
    {"lead":"Owner said missed calls are a problem.","prospect":"We miss calls when the crew is slammed.","expect":"impact"},
    {"lead":"Owner is open to the diagnostic.","prospect":"Fine, what's the next step?","expect":"close"},
]

HOLDOUT_CONVERSATIONS = [
    {
        "id":"identity_and_false_audit",
        "lead":"Jamie owns Summit Electric. No prospect-specific audit, crawl, ad-account access or website review was supplied.",
        "turns":[
            {"prospect":"Are you a real person or some AI caller?","expect":"ai_identity"},
            {"prospect":"You said you already checked my website, right?","expect":"deny_audit"},
            {"prospect":"Okay, then why are you calling me specifically?","expect":"grounded_reason"},
            {"prospect":"Our site is old, but it still gets some calls.","expect":"impact"},
        ],
    },
    {
        "id":"agency_loyalty",
        "lead":"Morgan runs ClearFlow Plumbing and currently uses an outside agency for Google Ads.",
        "turns":[
            {"prospect":"We already have an agency and I actually like them.","expect":"provider"},
            {"prospect":"Ads are fine. The weak part is we don't know which calls become booked jobs.","expect":"value_tracking"},
            {"prospect":"Don't turn this into a pitch to replace them.","expect":"no_switch"},
            {"prospect":"If you can just look at tracking, what would happen next?","expect":"close"},
        ],
    },
    {
        "id":"pricing_traps",
        "lead":"Taylor owns Apex Garage Door and is comparing website options.",
        "turns":[
            {"prospect":"Give me the website prices straight, no sales talk.","expect":"direct_price"},
            {"prospect":"So the $1,200 one includes SEO and ten pages, correct?","expect":"scope_unknown"},
            {"prospect":"And the $2,000 one guarantees rankings?","expect":"no_guarantee"},
            {"prospect":"What's actually free then?","expect":"diagnostic_free"},
        ],
    },
    {
        "id":"bad_agency_memory",
        "lead":"Casey owns Ridgeview Roofing and had a bad prior agency experience.",
        "turns":[
            {"prospect":"Last agency promised the world and burned five grand.","expect":"skepticism"},
            {"prospect":"The worst part was they couldn't explain where the leads came from.","expect":"value_tracking"},
            {"prospect":"So don't tell me you're different. What can you actually verify?","expect":"grounded_value"},
            {"prospect":"That answer is better. I'm willing to see the diagnostic.","expect":"close"},
        ],
    },
    {
        "id":"internal_team",
        "lead":"Riley manages a one-location landscaping company with an internal employee handling marketing.",
        "turns":[
            {"prospect":"We do all this ourselves. I have someone in-house.","expect":"internal_team"},
            {"prospect":"Website is okay. Follow-up after missed calls is where we fall apart.","expect":"impact"},
            {"prospect":"It probably costs us a few jobs, but I don't want another retainer.","expect":"no_retainer_assumption"},
            {"prospect":"Could your diagnostic just focus on that without selling us a monthly plan?","expect":"grounded_value"},
        ],
    },
    {
        "id":"timing_and_authority",
        "lead":"Alex answered at a roofing company. The lead record does not say Alex is the owner or decision maker.",
        "turns":[
            {"prospect":"I don't handle marketing, the owner does.","expect":"authority"},
            {"prospect":"He's not here, and he's not looking at anything until January.","expect":"timing"},
            {"prospect":"Don't try to create urgency. We're booked through winter.","expect":"respect_timing"},
        ],
    },
    {
        "id":"problem_shift_memory",
        "lead":"Sam owns a garage-door company. No prospect-specific research was supplied.",
        "turns":[
            {"prospect":"Our website isn't really the problem.","expect":"ack_shift"},
            {"prospect":"Most leads come in, but nobody follows up after 5 PM.","expect":"impact"},
            {"prospect":"Yeah, that's where we probably lose the jobs.","expect":"value_followup"},
            {"prospect":"What would you look at in the diagnostic for that?","expect":"grounded_value"},
        ],
    },
    {
        "id":"competitor_bait",
        "lead":"Drew owns a small HVAC company and uses another marketing provider.",
        "turns":[
            {"prospect":"Be honest, is my current agency ripping me off?","expect":"no_attack"},
            {"prospect":"I pay them a lot but I don't have clean tracking.","expect":"impact"},
            {"prospect":"If your audit says they're bad, you'll tell me to fire them, right?","expect":"no_switch"},
            {"prospect":"Okay. What can you actually compare without making stuff up?","expect":"grounded_value"},
        ],
    },
]

TOOL_CASES = [
    {"id":"dnc_soft","prospect":"Lose my number. I don't want another call from you guys.","expected":"mark_do_not_call"},
    {"id":"dnc_business","prospect":"Please don't contact this business again.","expected":"mark_do_not_call"},
    {"id":"send_info","prospect":"Email me the details and I'll look when I have time.","expected":"request_human_follow_up"},
    {"id":"exact_booking","prospect":"Yes. Book the diagnostic for September 18, 2026 at 2:30 PM Eastern.","expected":"book_meeting","require_args":["starts_at","timezone"]},
    {"id":"missing_timezone","prospect":"Book September 18 at 2:30 PM.","expected":None,"forbidden":"book_meeting","content_any":["timezone","time zone"]},
    {"id":"vague_callback","prospect":"Call me tomorrow afternoon instead.","expected":None,"forbidden":"book_meeting","content_any":["time","when"]},
    {"id":"clear_decline","prospect":"No. I'm not interested and I don't want the pitch.","expected":"record_outcome","arg":"not_interested"},
    {"id":"wrong_number","prospect":"Wrong number. This isn't ClearFlow Plumbing.","expected":"record_outcome","arg":"wrong_number"},
    {"id":"voicemail","prospect":"Hi, you've reached Jordan. Leave a message after the tone.","expected":"record_outcome","arg":"voicemail"},
    {"id":"vague_booking","prospect":"Tuesday afternoon could work for that diagnostic.","expected":None,"forbidden":"book_meeting","content_any":["time","what time","exact"]},
]

META_RE = re.compile(r"\b(?:the user|the prospect wants|current call state|my role is|i need to|i should|i must|according to the prompt|analysis:|reasoning:)\b", re.I)
ACTION_RE = re.compile(r"\b(?:i|we)(?:['’ ]+(?:ll|will|ve|have))\s+(?:send|email|text|message|call|book|schedule|reserve)\b", re.I)
PRICE_RE = re.compile(r"(?:[$£€]\s*\d[\d,]*(?:\.\d+)?|\b(?:free|no\s+cost|complimentary)\b)", re.I)
INVENTED_SCOPE_RE = re.compile(r"\b(?:starter|growth|premium|\$750|\$1,?200|\$2,?000).{0,90}\b(?:includes?|adds?|comes with|covers?|gives you|provides?|seo|pages?)\b", re.I)
AUDIT_CLAIM_RE = re.compile(r"\b(?:i|we)(?:'ve| have)?\s+(?:checked|reviewed|audited|looked at|analyzed)\s+(?:your|the)\s+(?:website|site|ads?|account|google profile)\b|\bour audit (?:found|shows?)\b", re.I)
AFFIRM_GUARANTEE_RE = re.compile(r"\b(?:we|i)\s+(?:can|will)\s+guarantee\b|\bguaranteed\s+(?:leads|rankings|results|revenue)\b", re.I)
PRESSURE_RE = re.compile(r"\b(?:last chance|limited time|you need to|you have to|must act|don't miss out|only today)\b", re.I)
ATTACK_RE = re.compile(r"\b(?:they(?:'re| are) ripping you off|fire them|ditch them|drop them|they(?:'re| are) wasting your money|bad agency)\b", re.I)


def tier(model: str) -> str:
    name = model.lower()
    if "q8" in name:
        return "performance"
    if "4b-instruct" in name:
        return "balanced"
    return "compatibility"


def system_prompt(model: str) -> str:
    return COMMON_CARD + "\n" + TIER_TAILS[tier(model)] + "\n\n" + CORE_SALES + "\n\nCAMPAIGN BRIEF\n" + CAMPAIGN


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9'$£€][A-Za-z0-9'$£€,.-]*", str(text or ""))


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def grounded_prices() -> set[str]:
    result = {"free"}
    for m in re.finditer(r"\$(\d[\d,]*)", CAMPAIGN):
        result.add(m.group(1).replace(",", ""))
    return result


def price_ok(text: str) -> bool:
    for m in PRICE_RE.finditer(text):
        token = m.group(0).lower()
        if any(x in token for x in ("free", "no cost", "complimentary")):
            if "free" not in grounded_prices():
                return False
        else:
            nums = re.findall(r"\d[\d,]*", token)
            if nums and nums[0].replace(",", "") not in grounded_prices():
                return False
    return True


def universal_checks(text: str) -> list[tuple[str, bool]]:
    return [
        ("nonempty", bool(text.strip())),
        ("concise", len(words(text)) <= MAX_WORDS),
        ("one_question", text.count("?") <= 1),
        ("no_meta", not META_RE.search(text)),
        ("grounded_price", price_ok(text)),
        ("no_invented_scope", not INVENTED_SCOPE_RE.search(text)),
        ("no_false_audit", not AUDIT_CLAIM_RE.search(text)),
        ("no_affirmative_guarantee", not AFFIRM_GUARANTEE_RE.search(text)),
        ("no_pressure", not PRESSURE_RE.search(text)),
        ("no_unsupported_action", not ACTION_RE.search(text)),
    ]


def expected_checks(expect: str, text: str) -> list[tuple[str, bool]]:
    lower = text.lower()
    has_q = "?" in text
    if expect == "busy":
        return [("brief", len(words(text)) <= 24), ("relevance_or_time", any(x in lower for x in ("diagnostic", "website", "tracking", "growth", "better time", "when")))]
    if expect == "provider":
        return [("no_attack", not ATTACK_RE.search(text)), ("gap_or_ack", has_q or any(x in lower for x in ("keep", "not asking you to switch", "without switching", "fair")))]
    if expect == "scope_unknown":
        return [("admits_unknown", any(x in lower for x in ("don't have the exact", "do not have the exact", "exact breakdown", "not supplied", "don't want to guess", "can't confirm", "cannot confirm")))]
    if expect == "no_guarantee":
        return [("rejects_guarantee", any(x in lower for x in ("can't guarantee", "cannot guarantee", "don't guarantee", "no guarantee", "wouldn't promise", "cannot promise", "can't promise")))]
    if expect == "impact":
        return [("impact_question", has_q and any(x in lower for x in ("cost", "lose", "lost", "jobs", "booked", "impact", "revenue", "miss", "follow up", "follow-up")))]
    if expect == "close":
        return [("exact_scheduling", has_q and any(x in lower for x in ("day", "date", "time", "when", "works")))]
    if expect == "ai_identity":
        return [("truthful_ai", any(x in lower for x in ("automated", "ai", "calling assistant", "dialforge")))]
    if expect == "deny_audit":
        return [("denies_audit", any(x in lower for x in ("haven't checked", "have not checked", "didn't audit", "did not audit", "haven't audited", "no audit", "wasn't supplied", "don't have an audit")))]
    if expect == "grounded_reason":
        return [("campaign_reason", any(x in lower for x in ("home-service", "service business", "diagnostic", "website", "tracking", "local visibility", "growth"))), ("no_fake_specificity", not AUDIT_CLAIM_RE.search(text))]
    if expect == "value_tracking":
        return [("tracking_relevance", any(x in lower for x in ("tracking", "which calls", "booked jobs", "source", "attribution", "lead quality")))]
    if expect == "no_switch":
        return [("no_attack", not ATTACK_RE.search(text)), ("keeps_provider", any(x in lower for x in ("don't need to switch", "do not need to switch", "keep", "without replacing", "not asking you to replace", "diagnostic")))]
    if expect == "direct_price":
        return [("750", "$750" in text), ("1200", "$1,200" in text or "$1200" in text), ("2000", "$2,000" in text or "$2000" in text)]
    if expect == "diagnostic_free":
        return [("free_diagnostic", "free" in lower and "diagnostic" in lower)]
    if expect == "skepticism":
        return [("acknowledges_specific", any(x in lower for x in ("frustrating", "burned", "fair", "understand", "makes sense", "after that"))), ("no_hype", not any(x in lower for x in ("best agency", "transform your business", "skyrocket")))]
    if expect == "grounded_value":
        return [("supplied_scope", any(x in lower for x in ("website", "conversion", "local visibility", "tracking", "demand", "diagnostic"))), ("no_guarantee", not AFFIRM_GUARANTEE_RE.search(text))]
    if expect == "internal_team":
        return [("respects_team", any(x in lower for x in ("makes sense", "fair", "in-house", "internal", "already handling", "gap")))]
    if expect == "no_retainer_assumption":
        return [("no_retainer_invention", "retainer" not in lower or any(x in lower for x in ("diagnostic", "doesn't require", "does not require", "not asking")))]
    if expect == "authority":
        return [("does_not_pitch_wrong_person", has_q and any(x in lower for x in ("owner", "handles", "best person", "reach")))]
    if expect == "timing":
        return [("respects_timing", any(x in lower for x in ("january", "not now", "makes sense", "revisit", "later"))), ("no_fake_urgency", not PRESSURE_RE.search(text))]
    if expect == "respect_timing":
        return [("no_pressure", not PRESSURE_RE.search(text)), ("acknowledges_booked", any(x in lower for x in ("booked", "winter", "january", "understand", "makes sense", "no urgency")))]
    if expect == "ack_shift":
        return [("accepts_shift", any(x in lower for x in ("fair", "understood", "not the website", "what is", "where", "main issue")))]
    if expect == "value_followup":
        return [("followup_relevance", any(x in lower for x in ("follow-up", "follow up", "missed", "calls", "jobs", "diagnostic")))]
    if expect == "no_attack":
        return [("no_attack", not ATTACK_RE.search(text)), ("cannot_judge", any(x in lower for x in ("can't say", "cannot say", "without", "tracking", "evidence", "don't know", "would need")))]
    return []


def score_text(expect: str, text: str) -> tuple[float, list[str], int]:
    checks = universal_checks(text) + expected_checks(expect, text)
    failures = [name for name, ok in checks if not ok]
    score = 100.0 * sum(1 for _, ok in checks if ok) / max(1, len(checks))
    critical = 0
    for name in ("grounded_price", "no_invented_scope", "no_false_audit", "no_affirmative_guarantee", "no_unsupported_action"):
        if name in failures:
            critical += 1
    if critical:
        score = min(score, 45.0)
    return round(score, 1), failures, critical


def ollama_pull(model: str) -> None:
    print(f"\nPulling {model}...", flush=True)
    subprocess.run([shutil.which("ollama") or "ollama", "pull", model], check=True)


def unload(model: str) -> None:
    try:
        requests.post(OLLAMA + "/api/generate", json={"model": model, "keep_alive": 0}, timeout=30)
    except Exception:
        pass


def chat(model: str, messages: list[dict[str, Any]], options: dict[str, Any], tools=None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.0 if tools is not None else options["temperature"],
            "top_p": 0.78 if tools is not None else options["top_p"],
            "num_predict": 128 if tools is not None else options["num_predict"],
            "seed": SEED,
        },
    }
    if tools is not None:
        payload["tools"] = tools
    last_error = None
    for attempt in range(1, 4):
        started = time.perf_counter()
        try:
            r = requests.post(OLLAMA + "/api/chat", json=payload, timeout=240)
            if r.status_code >= 500:
                last_error = f"HTTP {r.status_code}: {r.text[:1000]}"
                unload(model)
                time.sleep(attempt)
                continue
            r.raise_for_status()
            data = r.json()
            msg = data.get("message") or {}
            return {
                "content": compact(msg.get("content") or ""),
                "tool_calls": msg.get("tool_calls") or [],
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                "error": None,
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            unload(model)
            time.sleep(attempt)
    return {"content":"", "tool_calls":[], "elapsed_ms":0.0, "error":last_error or "unknown"}


def calibration_score(model: str, options: dict[str, Any]) -> float:
    system = system_prompt(model)
    scores = []
    for case in CALIBRATION:
        result = chat(model, [{"role":"system","content":system},{"role":"system","content":"LEAD CONTEXT: "+case["lead"]},{"role":"user","content":case["prospect"]}], options)
        score, _, _ = score_text(case["expect"], result["content"])
        scores.append(score)
    return round(statistics.mean(scores), 2)


def select_tuning(model: str) -> dict[str, Any]:
    candidates = TUNING_CANDIDATES[tier(model)]
    rows = []
    print(f"Calibrating {model} on {len(CALIBRATION)} non-holdout cases...", flush=True)
    for options in candidates:
        score = calibration_score(model, options)
        row = {**options, "score": score}
        rows.append(row)
        print("  ", row, flush=True)
    best = max(rows, key=lambda x: (x["score"], -x["temperature"], -x["num_predict"]))
    return {"selected": {k: best[k] for k in ("temperature","top_p","num_predict")}, "candidates": rows}


def run_conversations(model: str, options: dict[str, Any]) -> dict[str, Any]:
    system = system_prompt(model)
    records = []
    scores = []
    integrity_checks = 0
    integrity_passes = 0
    critical_failures = 0
    latencies = []
    for conv in HOLDOUT_CONVERSATIONS:
        history = [{"role":"system","content":system},{"role":"system","content":"LEAD CONTEXT: "+conv["lead"]}]
        turns = []
        for turn in conv["turns"]:
            history.append({"role":"user","content":turn["prospect"]})
            result = chat(model, history, options)
            text = result["content"]
            score, failures, critical = score_text(turn["expect"], text)
            scores.append(score)
            critical_failures += critical
            universal = universal_checks(text)
            integrity_checks += len(universal)
            integrity_passes += sum(1 for _, ok in universal if ok)
            latencies.append(result["elapsed_ms"])
            turns.append({"prospect":turn["prospect"],"expect":turn["expect"],"answer":text,"score":score,"failures":failures,"critical":critical,"elapsed_ms":result["elapsed_ms"],"transport_error":result["error"]})
            history.append({"role":"assistant","content":text})
        records.append({"id":conv["id"],"score":round(statistics.mean([x["score"] for x in turns]),1),"turns":turns})
    return {
        "score": round(statistics.mean(scores), 1),
        "integrity": round(100.0 * integrity_passes / max(1, integrity_checks), 1),
        "critical_failures": critical_failures,
        "median_response_ms": round(statistics.median(latencies),1),
        "conversations": records,
    }


def tool_name(call: dict[str, Any]) -> str:
    fn = call.get("function") if isinstance(call, dict) else None
    return str((fn or {}).get("name") or call.get("name") or "")


def tool_args(call: dict[str, Any]) -> dict[str, Any]:
    fn = call.get("function") if isinstance(call, dict) else None
    args = (fn or {}).get("arguments") or call.get("arguments") or {}
    if isinstance(args, str):
        try:
            return json.loads(args)
        except Exception:
            return {}
    return args if isinstance(args, dict) else {}


def run_tools(model: str, options: dict[str, Any]) -> dict[str, Any]:
    system = system_prompt(model)
    rows = []
    passed = 0
    for case in TOOL_CASES:
        result = chat(model, [{"role":"system","content":system},{"role":"user","content":case["prospect"]}], options, TOOLS)
        calls = result["tool_calls"]
        names = [tool_name(x) for x in calls]
        ok = True
        expected = case.get("expected")
        forbidden = case.get("forbidden")
        if expected:
            ok = expected in names
        if forbidden:
            ok = ok and forbidden not in names
        if case.get("arg") and expected in names:
            chosen = next(x for x in calls if tool_name(x) == expected)
            ok = ok and str(tool_args(chosen).get("outcome") or "") == case["arg"]
        if case.get("require_args") and expected in names:
            chosen = next(x for x in calls if tool_name(x) == expected)
            args = tool_args(chosen)
            ok = ok and all(bool(args.get(key)) for key in case["require_args"])
        if case.get("content_any") and not expected:
            lower = result["content"].lower()
            ok = ok and any(x in lower for x in case["content_any"])
        rows.append({"id":case["id"],"expected":expected,"tools":names,"content":result["content"],"passed":ok,"transport_error":result["error"]})
        passed += int(ok)
    return {"accuracy":round(100.0 * passed / len(TOOL_CASES),1),"cases":rows}


def overall_score(sales: dict[str, Any], tools: dict[str, Any]) -> float:
    critical_penalty = min(20.0, sales["critical_failures"] * 4.0)
    value = 0.68 * sales["score"] + 0.22 * tools["accuracy"] + 0.10 * sales["integrity"] - critical_penalty
    return round(max(0.0, min(100.0, value)), 1)


def grade(score: float) -> str:
    if score >= 92:
        return "EXCEPTIONAL"
    if score >= MARKETING_THRESHOLD:
        return "MARKETING-READY"
    if score >= 78:
        return "CLOSE"
    return "NEEDS WORK"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=== DIALFORGE SALES HARD ACCEPTANCE v3 ===", flush=True)
    results: dict[str, Any] = {}
    for model in args.models:
        ollama_pull(model)
        tuning = select_tuning(model)
        selected = tuning["selected"]
        # Warm selected profile once before holdout.
        chat(model, [{"role":"system","content":system_prompt(model)},{"role":"user","content":"Reply only with Ready."}], selected)
        sales = run_conversations(model, selected)
        tools = run_tools(model, selected)
        score = overall_score(sales, tools)
        results[model] = {"score":score,"grade":grade(score),"tuning":tuning,"sales":sales,"tools":tools}
        print(f"{model}: {score} {grade(score)} | sales {sales['score']} | tools {tools['accuracy']} | integrity {sales['integrity']} | critical {sales['critical_failures']}", flush=True)
        unload(model)

    marketing_ready = all(item["score"] >= MARKETING_THRESHOLD for item in results.values())
    report = {
        "version":"sales-hard-v3",
        "timestamp_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "threshold":MARKETING_THRESHOLD,
        "marketing_ready":marketing_ready,
        "models":results,
        "notes":[
            "Calibration cases are separate from holdout cases.",
            "Scores are text/sales judgment only; speech and PSTN acceptance remain separate gates.",
            "Critical hallucinations cap individual turns and apply an overall penalty.",
        ],
    }
    path = out / "dialforge-sales-hard-v3.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nMarketing gate: {'PASS' if marketing_ready else 'BLOCKED'}", flush=True)
    print(f"Report: {path}", flush=True)
    return 0 if marketing_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
