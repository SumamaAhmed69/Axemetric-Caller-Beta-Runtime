#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

CONTROL = re.compile(r"\b(?:record_outcome|book_meeting|mark_do_not_call|request_human_follow_up|tool[_ -]?(?:call|name|arguments)|private call control rules|decision card)\b", re.I)
UNSAFE_ACTION = re.compile(r"\b(?:i|we)(?:'ll| will)\s+(?:send|email|text|book|schedule|call you|call back)\b", re.I)
GUARANTEE = re.compile(r"\b(?:guarantee|guaranteed|promise)\b.{0,80}\b(?:rank|lead|job|revenue|result|roi|saving)\b", re.I)
GUARANTEE_NEGATION = re.compile(r"\b(?:can't|cannot|can not|don't|do not|won't|will not|wouldn't|would not|never)\b.{0,35}\b(?:guarantee|promise)\b|\b(?:no guarantee|not guaranteed)\b", re.I)
PRESSURE = re.compile(r"\b(?:what would it take to get a yes today|you really need|can't hurt|only take thirty seconds|act now)\b", re.I)


def rows(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except Exception as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}")
    return out


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def validate_sft(items: list[dict]) -> list[str]:
    errors: list[str] = []
    ids = Counter(str(x.get("id")) for x in items)
    for key, count in ids.items():
        if count != 1:
            errors.append(f"duplicate SFT id: {key}")
    prompt_keys = Counter((norm(x.get("system")), norm(x.get("user"))) for x in items)
    for key, count in prompt_keys.items():
        if count != 1:
            errors.append(f"duplicate SFT prompt: {key[1][:100]}")
    for item in items:
        ident = item.get("id")
        answer = str(item.get("assistant") or "").strip()
        if not answer:
            errors.append(f"{ident}: empty assistant answer")
        if len(answer) > 520:
            errors.append(f"{ident}: assistant answer too long ({len(answer)} chars)")
        if answer.count("?") > 1:
            errors.append(f"{ident}: more than one question")
        if CONTROL.search(answer):
            errors.append(f"{ident}: internal control syntax in preferred answer")
        if UNSAFE_ACTION.search(answer):
            errors.append(f"{ident}: unsupported external-action promise in preferred answer")
        if GUARANTEE.search(answer) and not GUARANTEE_NEGATION.search(answer):
            errors.append(f"{ident}: unsupported positive result guarantee in preferred answer")
        if PRESSURE.search(answer):
            errors.append(f"{ident}: pressure language in preferred answer")
    return errors


def validate_dpo(items: list[dict]) -> list[str]:
    errors: list[str] = []
    ids = Counter(str(x.get("id")) for x in items)
    for key, count in ids.items():
        if count != 1:
            errors.append(f"duplicate DPO id: {key}")
    for item in items:
        ident = item.get("id")
        chosen = norm(item.get("chosen"))
        rejected = norm(item.get("rejected"))
        if not chosen or not rejected:
            errors.append(f"{ident}: missing chosen/rejected")
        if chosen == rejected:
            errors.append(f"{ident}: chosen and rejected are identical")
        if CONTROL.search(str(item.get("chosen") or "")):
            errors.append(f"{ident}: internal control syntax in chosen")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="training/data")
    args = parser.parse_args()
    root = Path(args.data)
    sft = rows(root / "sales_sft.jsonl")
    dpo = rows(root / "sales_dpo.jsonl")
    errors = validate_sft(sft) + validate_dpo(dpo)
    sft_ids = {x["id"] for x in sft}
    dpo_ids = {x["id"] for x in dpo}
    if sft_ids != dpo_ids:
        errors.append("SFT and DPO example ID sets do not match")
    if errors:
        for error in errors:
            print("FAIL:", error)
        return 1
    print(f"PASS: {len(sft)} SFT + {len(dpo)} preference examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
