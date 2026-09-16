#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--adapter", default="training/output/lineborn-sales-dpo/adapter")
    parser.add_argument("--output", default="training/output/lineborn-sales-merged")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    merged = model.merge_and_unload(safe_merge=True)
    merged.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(output)
    print(f"merged model saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
