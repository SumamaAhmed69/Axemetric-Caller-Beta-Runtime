# Lineborn Sales LoRA

This directory contains the candidate fine-tuning pipeline for the Lineborn sales model.

## Objective

Train **Qwen/Qwen3-4B-Instruct-2507** as a stronger outbound-sales conversation model without teaching it carrier state, DNC persistence, booking persistence, or other deterministic product control. Those remain owned by Lineborn code.

The adapter is trained to improve conversational judgment: discovery, impact questions, relevance, objection resolution, trust, pricing answers, provider respect, authority, problem shifts, concise value bridges, buying-signal recognition, and clean closes.

## Guardrails

- Never train on the frozen release benchmark corpus. The benchmark remains holdout-only.
- Never put internal function names or router syntax into preferred assistant answers.
- Never reward invented pricing, package scope, guarantees, audits, results, urgency, social proof, or external actions.
- Do not train the model to override DNC, booking, voicemail, wrong-number, or call-lifecycle logic.
- A trained adapter is a release candidate only after it beats the unmodified 4B model on the frozen strict benchmark with zero critical failures and zero control leaks.

## Pipeline

1. `build_sales_corpus.py` creates a clean SFT corpus and a preference corpus from hand-authored sales situations. It deliberately uses prompts distinct from the release benchmark.
2. `validate_sales_corpus.py` blocks unsafe examples, duplicate prompts, control-language leakage, oversized answers, and chosen/rejected collisions.
3. `train_sales_lora.py` performs 4-bit QLoRA on Qwen3-4B-Instruct-2507. Only assistant tokens receive supervised loss.
4. `train_sales_dpo.py` optionally preference-tunes the SFT adapter so concise, consultative, truthful sales behavior wins over pitchy or hallucinated alternatives.
5. `merge_sales_adapter.py` merges the adapter into the base model for export/quantization.
6. Run the existing strict Lineborn benchmark against the merged model before changing the shipping runtime.

## Recommended Colab/T4 settings

SFT defaults are intentionally conservative for a 16 GB T4: NF4 4-bit base weights, LoRA rank 32, batch size 1, gradient accumulation 16, 2 epochs, 2048-token cap, gradient checkpointing, and assistant-only loss. DPO is optional and should be run only after SFT is stable.

The target shipping strategy is one 4B model family with different GGUF quantizations (for example Q4_K_M, Q5_K_M/Q6_K, and Q8_0), not separate 1.7B and 4B behavioral models.
