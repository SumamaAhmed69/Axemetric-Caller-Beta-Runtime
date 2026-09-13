#!/usr/bin/env python3
"""Dialforge Sales Hard Acceptance v3.4.

This gate separates generative model intelligence from deterministic product control.
Raw sales language must still score >=85. Native Ollama tool serialization is reported
as a diagnostic, but the shipping score uses Dialforge's production-equivalent router
for DNC, wrong-number, voicemail, follow-up, refusal, and booking completeness.

The gate remains strict: any critical spoken hallucination, raw-sales score below 85,
production tool accuracy below 90, or overall score below 85 blocks marketing.
"""
from __future__ import annotations

import statistics

import dialforge_sales_hard_acceptance_v3_3 as v33
from dialforge_production_safety import route_turn, sanitize_marketing_claim

base = v33.base
_RAW_RUN_TOOLS = v33.v32._ORIGINAL_RUN_TOOLS
_ORIGINAL_OVERALL = base.overall_score

# Add product-level false-positive and negation traps. These are intentionally cases
# where an over-eager deterministic router could be more dangerous than the model.
base.TOOL_CASES = list(base.TOOL_CASES) + [
    {
        "id":"temporal_not_dnc",
        "prospect":"Don't call me tomorrow. Next week is better.",
        "expected":None,
        "forbidden":"mark_do_not_call",
        "content_any":["date","time","when","next week"],
    },
    {
        "id":"bad_timing_not_wrong",
        "prospect":"This isn't a good time. I'm heading into a job.",
        "expected":None,
        "forbidden":"record_outcome",
    },
    {
        "id":"third_party_followup_not_dnc",
        "prospect":"Don't contact the owner directly. Email me the summary instead.",
        "expected":"request_human_follow_up",
    },
    {
        "id":"exact_confirm_later",
        "prospect":"September 26, 2026 at 2 PM Eastern works, but I'll confirm later.",
        "expected":None,
        "forbidden":"book_meeting",
        "content_any":["won't book","will not book","until you're ready","explicitly want","ready"],
    },
    {
        "id":"iso_exact_booking",
        "prospect":"2026-10-02 at 14:15 ET works for me. Book the meeting.",
        "expected":"book_meeting",
        "require_args":["starts_at","timezone"],
    },
    {
        "id":"dnc_stop_contacting",
        "prospect":"Stop contacting us. Remove this number from your list.",
        "expected":"mark_do_not_call",
    },
]

# One more compound sales sequence: the system must not convert an unsupported feature
# requirement into a claim just to keep momentum.
base.HOLDOUT_CONVERSATIONS = list(base.HOLDOUT_CONVERSATIONS) + [
    {
        "id":"package_scope_pressure",
        "lead":"Robin owns a home-service company. Exact package features are not supplied.",
        "turns":[
            {"prospect":"I only care if Premium includes SEO and at least fifteen pages. Does it?","expect":"scope_unknown"},
            {"prospect":"So you're saying it might include that, you just won't tell me?","expect":"scope_unknown"},
            {"prospect":"Fine. What can you actually verify in the free diagnostic?","expect":"grounded_value"},
        ],
    },
]


def _evaluate_router_case(case, decision):
    names = [decision.tool_name] if decision.kind == "tool" and decision.tool_name else []
    content = decision.spoken_reply or ""
    args = decision.tool_arguments or {}
    ok = True
    expected = case.get("expected")
    forbidden = case.get("forbidden")
    if expected:
        ok = expected in names
    if forbidden:
        ok = ok and forbidden not in names
    if case.get("arg") and expected in names:
        ok = ok and str(args.get("outcome") or "") == case["arg"]
    if case.get("require_args") and expected in names:
        ok = ok and all(bool(args.get(key)) for key in case["require_args"])
    if case.get("content_any") and not expected:
        lower = content.lower()
        ok = ok and any(str(x).lower() in lower for x in case["content_any"])
    return ok, names, content


def run_tools(model, options):
    # Raw native-model tool behavior stays visible so regressions are never hidden.
    raw = _RAW_RUN_TOOLS(model, options)
    raw_by_id = {row["id"]: row for row in raw["cases"]}

    rows = []
    passed = 0
    routed = 0
    for case in base.TOOL_CASES:
        decision = route_turn(case["prospect"])
        if decision.kind in {"tool", "spoken"}:
            routed += 1
            ok, names, content = _evaluate_router_case(case, decision)
            source = "deterministic_router"
        else:
            # Non-deterministic sales/timing language remains a model decision. Reuse the
            # raw run's already-scored result rather than generating a second answer.
            raw_row = raw_by_id[case["id"]]
            ok = bool(raw_row["passed"])
            names = list(raw_row["tools"])
            content = raw_row["content"]
            source = "model_fallback"
        rows.append({
            "id":case["id"],
            "expected":case.get("expected"),
            "tools":names,
            "content":content,
            "passed":ok,
            "source":source,
        })
        passed += int(ok)

    accuracy = round(100.0 * passed / len(base.TOOL_CASES), 1)
    print(
        f"  Production tool routing for {model}: {accuracy}% | raw native tools {raw['accuracy']}% | deterministic coverage {routed}/{len(base.TOOL_CASES)}",
        flush=True,
    )
    failed = [row for row in rows if not row["passed"]]
    for row in failed:
        print(f"    - PROD FAIL {row['id']}: tools={row['tools']} content={row['content']!r} source={row['source']}", flush=True)
    return {
        "accuracy":accuracy,
        "cases":rows,
        "raw_accuracy":raw["accuracy"],
        "raw_cases":raw["cases"],
        "deterministic_coverage":routed,
    }


def run_conversations(model, options):
    system = base.system_prompt(model)
    records = []
    scores = []
    raw_scores = []
    integrity_checks = 0
    integrity_passes = 0
    critical_failures = 0
    raw_critical_failures = 0
    latencies = []

    for conv in base.HOLDOUT_CONVERSATIONS:
        history = [
            {"role":"system","content":system},
            {"role":"system","content":"LEAD CONTEXT: "+conv["lead"]},
        ]
        turns = []
        for turn in conv["turns"]:
            history.append({"role":"user","content":turn["prospect"]})
            result = base.chat(model, history, options)
            raw_text = result["content"]
            raw_score, raw_failures, raw_critical = base.score_text(turn["expect"], raw_text)

            # This is the text that would actually be allowed through the shipping
            # package/guarantee safety boundary before TTS/history.
            text = sanitize_marketing_claim(raw_text, base.CAMPAIGN)
            score, failures, critical = base.score_text(turn["expect"], text)

            scores.append(score)
            raw_scores.append(raw_score)
            critical_failures += critical
            raw_critical_failures += raw_critical
            universal = base.universal_checks(text)
            integrity_checks += len(universal)
            integrity_passes += sum(1 for _, ok in universal if ok)
            latencies.append(result["elapsed_ms"])
            turns.append({
                "prospect":turn["prospect"],
                "expect":turn["expect"],
                "answer":text,
                "raw_answer":raw_text,
                "score":score,
                "raw_score":raw_score,
                "failures":failures,
                "raw_failures":raw_failures,
                "critical":critical,
                "raw_critical":raw_critical,
                "elapsed_ms":result["elapsed_ms"],
                "transport_error":result["error"],
            })
            # Product history sees the guarded spoken output, not an unsafe raw claim.
            history.append({"role":"assistant","content":text})
        records.append({
            "id":conv["id"],
            "score":round(statistics.mean([x["score"] for x in turns]),1),
            "raw_score":round(statistics.mean([x["raw_score"] for x in turns]),1),
            "turns":turns,
        })

    return {
        "score":round(statistics.mean(scores),1),
        "raw_score":round(statistics.mean(raw_scores),1),
        "integrity":round(100.0 * integrity_passes / max(1, integrity_checks),1),
        "critical_failures":critical_failures,
        "raw_critical_failures":raw_critical_failures,
        "median_response_ms":round(statistics.median(latencies),1),
        "conversations":records,
    }


def overall_score(sales, tools):
    value = _ORIGINAL_OVERALL(sales, tools)
    # Marketing cannot pass by averaging away a dangerous defect.
    if sales["raw_score"] < 85.0:
        return min(value, 84.9)
    if sales["critical_failures"] > 0:
        return min(value, 84.9)
    if tools["accuracy"] < 90.0:
        return min(value, 84.9)
    return value


base.run_tools = run_tools
base.run_conversations = run_conversations
base.overall_score = overall_score


if __name__ == "__main__":
    raise SystemExit(base.main())
