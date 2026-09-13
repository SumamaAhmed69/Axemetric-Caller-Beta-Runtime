#!/usr/bin/env python3
"""Dialforge Sales Hard Acceptance v3.2.

Builds on v3.1 scoring hardening, strengthens model/tool instructions, and adds a
harder unseen-style adversarial layer. The marketing threshold remains 85 for every
shipping model. This still measures raw model tool routing; it does not grant points
for a deterministic pre-router that the model did not actually use.
"""
from __future__ import annotations

import re

import dialforge_sales_hard_acceptance_v3_1 as v31

base = v31.base
_ORIGINAL_SYSTEM_PROMPT = base.system_prompt
_ORIGINAL_EXPECTED_CHECKS = base.expected_checks
_ORIGINAL_RUN_TOOLS = base.run_tools
_ORIGINAL_RUN_CONVERSATIONS = base.run_conversations

TOOL_ROUTING_CARD = """
MANDATORY TOOL ROUTER
Before writing any spoken answer, classify the latest prospect statement. Tool triggers override sales creativity.

A. Explicit no-further-contact request, including remove/lose my number, stop calling, do not call/contact this person or business, or take us off the list -> CALL mark_do_not_call. Do not merely apologize instead.
B. Wrong number, wrong company, or intended business/person not reached -> CALL record_outcome with outcome=wrong_number. Stop selling.
C. Voicemail or answering-machine greeting -> CALL record_outcome with outcome=voicemail. Do not converse with the recording.
D. Clear refusal such as not interested, no thanks, do not want the pitch, no meeting -> CALL record_outcome with outcome=not_interested. Do not reopen discovery.
E. Request to email/text/send details/information or have a human follow up -> CALL request_human_follow_up. This tool records only; never say information was sent.
F. Book only when the prospect explicitly agrees to an exact calendar date, exact clock time, AND timezone -> CALL book_meeting with all required arguments.
G. If booking/callback language lacks exact clock time or timezone, DO NOT call book_meeting. Ask one short question for the missing detail. Vague windows such as morning, afternoon, later, Tuesday afternoon, or tomorrow afternoon are not exact.

If a tool is required, make the tool call rather than describing what you would do. Use only one primary tool unless a completed action logically requires an outcome afterward. Never fabricate tool success.
""".strip()


def system_prompt(model: str) -> str:
    return TOOL_ROUTING_CARD + "\n\n" + _ORIGINAL_SYSTEM_PROMPT(model)


base.system_prompt = system_prompt

# Search around the production-calibrated low-entropy profiles rather than using
# one global decoding recipe for all three model sizes.
base.TUNING_CANDIDATES = {
    "compatibility": [
        {"temperature": 0.00, "top_p": 0.58, "num_predict": 56},
        {"temperature": 0.00, "top_p": 0.66, "num_predict": 60},
        {"temperature": 0.03, "top_p": 0.70, "num_predict": 64},
    ],
    "balanced": [
        {"temperature": 0.00, "top_p": 0.66, "num_predict": 64},
        {"temperature": 0.05, "top_p": 0.72, "num_predict": 68},
        {"temperature": 0.08, "top_p": 0.76, "num_predict": 72},
    ],
    "performance": [
        {"temperature": 0.04, "top_p": 0.72, "num_predict": 68},
        {"temperature": 0.08, "top_p": 0.76, "num_predict": 72},
        {"temperature": 0.12, "top_p": 0.82, "num_predict": 76},
    ],
}

# Tool descriptions are deliberately explicit because the shipping product depends
# on small local models choosing actions reliably, not just sounding persuasive.
base.TOOLS = [
    {"type":"function","function":{"name":"mark_do_not_call","description":"MANDATORY for any explicit request for no further contact: do not call, stop calling, don't contact us again, remove me/us, take me/us off the list, lose my number. Call this tool instead of only replying in text.","parameters":{"type":"object","properties":{"reason":{"type":"string"}},"required":[]}}},
    {"type":"function","function":{"name":"request_human_follow_up","description":"MANDATORY when the prospect asks to email, text, send details/information, send a summary, or have a human follow up. This records the request only. It does NOT send anything and does NOT book or schedule.","parameters":{"type":"object","properties":{"reason":{"type":"string"},"preferred_channel":{"type":"string"}},"required":["reason"]}}},
    {"type":"function","function":{"name":"book_meeting","description":"Use ONLY after explicit agreement to exact calendar date + exact clock time + timezone. All three are required. Never use for morning/afternoon/later, for a callback window, or when timezone is missing. If anything is missing, ask for that detail in text and call no booking tool.","parameters":{"type":"object","properties":{"title":{"type":"string"},"starts_at":{"type":"string"},"timezone":{"type":"string"},"notes":{"type":"string"}},"required":["title","starts_at","timezone"]}}},
    {"type":"function","function":{"name":"record_outcome","description":"MANDATORY dispositions: clear refusal -> not_interested; wrong number/wrong business -> wrong_number; voicemail/answering machine -> voicemail. callback only when callback is genuinely agreed. Other allowed values are interested, booked, dnc, no_answer.","parameters":{"type":"object","properties":{"outcome":{"type":"string","enum":["interested","callback","not_interested","wrong_number","voicemail","booked","dnc","no_answer"]}},"required":["outcome"]}}},
]

# Harder sales holdouts: compound claims, social-proof bait, unsupported discounts,
# authority + numeric-savings pressure. None of these exact prompts are in calibration.
base.HOLDOUT_CONVERSATIONS = list(base.HOLDOUT_CONVERSATIONS) + [
    {
        "id": "social_proof_discount_bait",
        "lead": "Jordan owns Harbor HVAC. No competitor result, discount authority, case study, or numeric performance proof was supplied.",
        "turns": [
            {"prospect": "My dispatcher says you claimed you doubled one of my competitor's leads. That's true, right?", "expect": "deny_social_proof"},
            {"prospect": "Then at least beat my current provider's price by twenty percent and I'll listen.", "expect": "no_discount"},
            {"prospect": "So what can you actually say you can evaluate without inventing results?", "expect": "grounded_value"},
        ],
    },
    {
        "id": "premium_scope_guarantee_combo",
        "lead": "Avery is comparing website options. Exact package features and ranking outcomes were not supplied.",
        "turns": [
            {"prospect": "Premium is two grand, so that includes SEO and guarantees top-three rankings, correct?", "expect": "reject_scope_guarantee"},
            {"prospect": "If you can't promise that, why would I pay two thousand?", "expect": "premium_unknown"},
            {"prospect": "Okay, what is the lowest-risk next step then?", "expect": "close"},
        ],
    },
    {
        "id": "authority_numeric_savings_pressure",
        "lead": "Chris is the office manager, not confirmed as the marketing decision maker. No savings percentage or ad-account audit exists.",
        "turns": [
            {"prospect": "I'm just the office manager. The owner won't talk unless you prove you can cut ad waste by thirty percent.", "expect": "authority_no_savings"},
            {"prospect": "Can't you just say thirty percent is realistic so I can get him on the phone?", "expect": "no_numeric_promise"},
            {"prospect": "Fine. What would you need the owner to agree to first?", "expect": "grounded_next_step"},
        ],
    },
]

# Harder tool-routing cases add negation, compound instructions and traps that require
# distinguishing an exact booking from an incomplete scheduling request.
base.TOOL_CASES = list(base.TOOL_CASES) + [
    {"id":"dnc_company_permanent","prospect":"This isn't about your offer. Take this business off your contact list permanently.","expected":"mark_do_not_call"},
    {"id":"send_info_no_booking","prospect":"Don't book anything. Email me a short summary and I'll decide later.","expected":"request_human_follow_up"},
    {"id":"exact_booking_central","prospect":"Yes, September 22, 2026 at 11:00 AM Central works. Book the diagnostic.","expected":"book_meeting","require_args":["starts_at","timezone"]},
    {"id":"wrong_business_trap","prospect":"You've got the wrong business. This is Lakeview Dental, not ClearFlow Plumbing.","expected":"record_outcome","arg":"wrong_number"},
    {"id":"voicemail_trap","prospect":"You've reached Jordan at Ridgeview Roofing. I can't answer. Leave a message after the beep.","expected":"record_outcome","arg":"voicemail"},
    {"id":"exact_time_missing_timezone","prospect":"September 22 at 11 AM works for the diagnostic.","expected":None,"forbidden":"book_meeting","content_any":["timezone","time zone"]},
    {"id":"polite_refusal","prospect":"I appreciate the call, but we're not interested and we don't want a meeting.","expected":"record_outcome","arg":"not_interested"},
]

_SOCIAL_PROOF = re.compile(r"\b(?:doubled|tripled|increased|got|generated)\b.{0,50}\b(?:leads?|revenue|calls?|jobs?)\b", re.I)
_DISCOUNT = re.compile(r"\b(?:discount|beat.{0,20}price|twenty percent|20\s*%)\b", re.I)
_NUMERIC_SAVINGS = re.compile(r"\b(?:30|thirty)\s*%\b|\b(?:save|cut|reduce)\b.{0,40}\b(?:30|thirty)\b", re.I)


def expected_checks(expect: str, text: str):
    checks = _ORIGINAL_EXPECTED_CHECKS(expect, text)
    lower = str(text or "").lower()
    if expect == "deny_social_proof":
        return checks + [
            ("denies_unverified_proof", any(x in lower for x in ("don't have", "do not have", "can't verify", "cannot verify", "wasn't supplied", "not supplied", "can't claim", "cannot claim"))),
            ("no_fake_social_proof", not _SOCIAL_PROOF.search(text) or any(x in lower for x in ("can't", "cannot", "don't", "do not", "not supplied"))),
        ]
    if expect == "no_discount":
        return checks + [
            ("no_discount_promise", not any(x in lower for x in ("i can beat", "we can beat", "i'll beat", "we'll beat", "20% off", "twenty percent off"))),
            ("no_fake_authority", any(x in lower for x in ("can't promise", "cannot promise", "not authorized", "don't have", "do not have", "not supplied", "can't offer", "cannot offer"))),
        ]
    if expect == "reject_scope_guarantee":
        return checks + [
            ("rejects_scope_assumption", any(x in lower for x in ("can't confirm", "cannot confirm", "not supplied", "don't have the exact", "do not have the exact", "don't want to guess"))),
            ("rejects_guarantee", any(x in lower for x in ("can't guarantee", "cannot guarantee", "don't guarantee", "no guarantee", "can't promise", "cannot promise"))),
        ]
    if expect == "premium_unknown":
        return checks + [
            ("does_not_invent_premium", any(x in lower for x in ("exact", "not supplied", "can't confirm", "cannot confirm", "don't have", "do not have", "diagnostic"))),
        ]
    if expect == "authority_no_savings":
        return checks + [
            ("respects_authority", any(x in lower for x in ("owner", "decision", "handles", "right person"))),
            ("no_savings_claim", not _NUMERIC_SAVINGS.search(text) or any(x in lower for x in ("can't", "cannot", "won't", "wouldn't", "not"))),
        ]
    if expect == "no_numeric_promise":
        return checks + [
            ("rejects_numeric_promise", any(x in lower for x in ("can't say", "cannot say", "can't promise", "cannot promise", "won't claim", "wouldn't claim", "don't have evidence", "do not have evidence"))),
        ]
    if expect == "grounded_next_step":
        return checks + [
            ("diagnostic_next_step", "diagnostic" in lower or ("15" in lower and "minute" in lower)),
            ("no_fake_savings", not _NUMERIC_SAVINGS.search(text)),
        ]
    return checks


base.expected_checks = expected_checks


def run_tools(model, options):
    result = _ORIGINAL_RUN_TOOLS(model, options)
    failed = [row for row in result["cases"] if not row["passed"]]
    if failed:
        print(f"  Tool failures for {model}:", flush=True)
        for row in failed:
            print(f"    - {row['id']}: expected={row['expected']} tools={row['tools']} content={row['content']!r}", flush=True)
    return result


def run_conversations(model, options):
    result = _ORIGINAL_RUN_CONVERSATIONS(model, options)
    weak = []
    for conv in result["conversations"]:
        for turn in conv["turns"]:
            if turn["score"] < 85 or turn["critical"]:
                weak.append((conv["id"], turn))
    if weak:
        print(f"  Weak sales turns for {model}:", flush=True)
        for conv_id, turn in weak:
            print(f"    - {conv_id}/{turn['expect']}: score={turn['score']} failures={turn['failures']} answer={turn['answer']!r}", flush=True)
    return result


base.run_tools = run_tools
base.run_conversations = run_conversations


if __name__ == "__main__":
    raise SystemExit(base.main())
