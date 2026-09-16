#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def common_prefix(a: list[int], b: list[int]) -> int:
    size = min(len(a), len(b))
    index = 0
    while index < size and a[index] == b[index]:
        index += 1
    return index


def tokenize_example(example: dict[str, Any], tokenizer, max_length: int) -> dict[str, Any]:
    prompt_messages = [
        {"role": "system", "content": str(example["system"])},
        {"role": "user", "content": str(example["user"])},
    ]
    full_messages = prompt_messages + [{"role": "assistant", "content": str(example["assistant"])}]
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    cutoff = common_prefix(prompt_ids, full_ids)
    full_ids = full_ids[:max_length]
    cutoff = min(cutoff, len(full_ids))
    labels = [-100] * cutoff + full_ids[cutoff:]
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "supervised_tokens": sum(1 for x in labels if x != -100),
    }


@dataclass
class Collator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in features)
        ids, masks, labels = [], [], []
        for row in features:
            pad = max_len - len(row["input_ids"])
            ids.append(row["input_ids"] + [self.pad_token_id] * pad)
            masks.append(row["attention_mask"] + [0] * pad)
            labels.append(row["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="training/data/sales_sft.jsonl")
    parser.add_argument("--output", default="training/output/lineborn-sales-sft")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for this QLoRA recipe")
    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant,
        device_map={"": 0},
        dtype=compute_dtype,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES,
        ),
    )
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=str(Path(args.data)), split="train")
    split = dataset.train_test_split(test_size=0.08, seed=args.seed)

    def mapper(row):
        return tokenize_example(row, tokenizer, args.max_length)

    train = split["train"].map(mapper, remove_columns=split["train"].column_names)
    valid = split["test"].map(mapper, remove_columns=split["test"].column_names)
    train = train.filter(lambda x: int(x["supervised_tokens"]) >= 4).remove_columns(["supervised_tokens"])
    valid = valid.filter(lambda x: int(x["supervised_tokens"]) >= 4).remove_columns(["supervised_tokens"])
    if not len(train) or not len(valid):
        raise SystemExit(f"tokenization produced an empty split: train={len(train)} eval={len(valid)}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        gradient_checkpointing=True,
        fp16=compute_dtype == torch.float16,
        bf16=compute_dtype == torch.bfloat16,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=1.0,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=valid,
        data_collator=Collator(tokenizer.pad_token_id),
    )
    train_result = trainer.train()
    eval_result = trainer.evaluate()
    model.save_pretrained(output / "adapter")
    tokenizer.save_pretrained(output / "adapter")
    metrics = {
        "base_model": args.base_model,
        "train_examples": len(train),
        "eval_examples": len(valid),
        "train_loss": train_result.metrics.get("train_loss"),
        "eval_loss": eval_result.get("eval_loss"),
        "eval_perplexity": math.exp(eval_result["eval_loss"]) if eval_result.get("eval_loss") is not None and eval_result["eval_loss"] < 20 else None,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_length": args.max_length,
        "seed": args.seed,
    }
    (output / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
