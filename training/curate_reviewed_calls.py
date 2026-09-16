#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d .()\-]{7,}\d)(?!\d)")
URL = re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.I)
CONTROL = re.compile(r"\b(?:record_outcome|book_meeting|mark_do_not_call|request_human_follow_up|tool[_ -]?(?:call|name|arguments))\b", re.I)


def redact(text: str) -> str:
    value = EMAIL.sub("[EMAIL]", str(text or ""))
    value = PHONE.sub("[PHONE]", value)
    value = URL.sub("[URL]", value)
    return re.sub(r"\s+", " ", value).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert explicitly reviewed beta-call examples into private training rows.")
    parser.add_argument("input", help="JSONL with approved, system, prospect, assistant, human_revision, rating")
    parser.add_argument("--sft-out", default="reviewed_sales_sft.jsonl")
    parser.add_argument("--dpo-out", default="reviewed_sales_dpo.jsonl")
    parser.add_argument("--min-rating", type=int, default=4)
    args = parser.parse_args()

    sft, dpo = [], []
    with Path(args.input).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("approved") is not True or int(row.get("rating") or 0) < args.min_rating:
                continue
            system = redact(row.get("system"))
            user = redact(row.get("prospect"))
            rejected = redact(row.get("assistant"))
            chosen = redact(row.get("human_revision"))
            if not system or not user or not chosen or chosen == rejected:
                continue
            if CONTROL.search(chosen):
                raise SystemExit(f"line {line_no}: approved revision contains internal control syntax")
            ident = str(row.get("id") or f"reviewed-{line_no}")
            sft.append({"id": ident, "category": "reviewed_real_call", "system": system, "user": user, "assistant": chosen})
            if rejected:
                dpo.append({"id": ident, "category": "reviewed_real_call", "system": system, "user": user, "chosen": chosen, "rejected": rejected})

    for path, rows in ((Path(args.sft_out), sft), (Path(args.dpo_out), dpo)):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(sft)} approved SFT examples and {len(dpo)} preference pairs")
    print("Keep reviewed-call datasets private; do not commit customer transcripts to this public repository.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
