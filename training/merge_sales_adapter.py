#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    return requested


def resolve_dtype(requested: str, device: str):
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if requested == "float32":
        return torch.float32
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--adapter", default="training/output/lineborn-sales-dpo/adapter")
    parser.add_argument("--output", default="training/output/lineborn-sales-merged")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    device_map = {"": 0} if device == "cuda" else "cpu"

    print(f"Loading {args.base_model} on {device} as {dtype}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    print(f"Loading adapter {args.adapter}", flush=True)
    model = PeftModel.from_pretrained(base, args.adapter)
    print("Merging LoRA into base model...", flush=True)
    merged = model.merge_and_unload(safe_merge=True)
    print(f"Saving merged model to {output}", flush=True)
    merged.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(output)
    print(f"merged model saved to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
