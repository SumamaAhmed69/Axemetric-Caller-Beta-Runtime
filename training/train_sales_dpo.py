#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import DPOConfig, DPOTrainer

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
BASE_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


def compatible_init_kwargs(cls, values: dict[str, Any], aliases: dict[str, str] | None = None, label: str | None = None) -> dict[str, Any]:
    aliases = aliases or {}
    params = inspect.signature(cls.__init__).parameters
    out: dict[str, Any] = {}
    prefix = label or cls.__name__
    for key, value in values.items():
        if key in params:
            out[key] = value
            continue
        alias = aliases.get(key)
        if alias and alias in params:
            out[alias] = value
            print(f"{prefix} compatibility: {key} -> {alias}")
            continue
        print(f"{prefix} compatibility: omitting unsupported option {key}={value!r}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="training/data/sales_dpo.jsonl")
    parser.add_argument("--adapter", default="training/output/lineborn-sales-sft/adapter")
    parser.add_argument("--output", default="training/output/lineborn-sales-dpo")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--base-revision", default=BASE_REVISION)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=args.base_revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=args.base_revision,
        quantization_config=quant,
        device_map={"": 0},
        dtype=compute_dtype,
    )
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True)

    dataset = load_dataset("json", data_files=str(Path(args.data)), split="train")  # nosec B615 - local JSON builder + explicit local file

    def format_row(row):
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": str(row["system"])},
                {"role": "user", "content": str(row["user"])},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {"prompt": prompt, "chosen": str(row["chosen"]), "rejected": str(row["rejected"])}

    dataset = dataset.map(format_row, remove_columns=dataset.column_names)
    split = dataset.train_test_split(test_size=0.08, seed=args.seed)
    if not len(split["train"]) or not len(split["test"]):
        raise SystemExit(f"DPO split is empty: train={len(split['train'])} eval={len(split['test'])}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    optimizer_steps_per_epoch = max(1, math.ceil(len(split["train"]) / max(1, args.grad_accum)))
    total_optimizer_steps = max(1, math.ceil(optimizer_steps_per_epoch * args.epochs))
    warmup_steps = max(1, round(total_optimizer_steps * 0.05))

    config_values = {
        "output_dir": str(output),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "beta": args.beta,
        "max_length": args.max_length,
        "max_prompt_length": min(1408, args.max_length - 128),
        "logging_steps": 5,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "gradient_checkpointing": True,
        "fp16": compute_dtype == torch.float16,
        "bf16": compute_dtype == torch.bfloat16,
        "optim": "paged_adamw_8bit",
        "warmup_steps": warmup_steps,
        "lr_scheduler_type": "cosine",
        "report_to": "none",
        "seed": args.seed,
    }
    config = DPOConfig(
        **compatible_init_kwargs(
            DPOConfig,
            config_values,
            aliases={"eval_strategy": "evaluation_strategy"},
            label="DPOConfig",
        )
    )
    print(
        f"DPO schedule: {len(split['train'])} train / {len(split['test'])} eval, "
        f"~{total_optimizer_steps} optimizer updates, {warmup_steps} warmup steps",
        flush=True,
    )

    trainer_values: dict[str, Any] = {
        "model": model,
        "ref_model": None,
        "args": config,
        "train_dataset": split["train"],
        "eval_dataset": split["test"],
        "processing_class": tokenizer,
    }
    trainer = DPOTrainer(
        **compatible_init_kwargs(
            DPOTrainer,
            trainer_values,
            aliases={"processing_class": "tokenizer"},
            label="DPOTrainer",
        )
    )
    train_result = trainer.train()
    eval_result = trainer.evaluate()
    adapter_dir = output / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    adapter_weights = adapter_dir / "adapter_model.safetensors"
    if not adapter_weights.is_file() or adapter_weights.stat().st_size <= 1024 * 1024:
        raise SystemExit("DPO training completed but a valid adapter_model.safetensors was not produced")

    metrics = {
        "base_model": args.base_model,
        "sft_adapter": str(args.adapter),
        "train_examples": len(split["train"]),
        "eval_examples": len(split["test"]),
        "train_loss": train_result.metrics.get("train_loss"),
        "eval_loss": eval_result.get("eval_loss"),
        "beta": args.beta,
        "warmup_steps": warmup_steps,
        "seed": args.seed,
        "adapter_bytes": adapter_weights.stat().st_size,
    }
    (output / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
