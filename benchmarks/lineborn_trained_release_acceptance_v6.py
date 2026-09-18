#!/usr/bin/env python3
"""Strict Lineborn release acceptance for the trained v6 4B model family.

This wrapper preserves the frozen v5 sales/safety corpus and changes only the
shipping candidate model tags/options. Balanced and performance are the same
merged Lineborn v6 brain at different GGUF quantizations.
"""
from __future__ import annotations

import subprocess

import lineborn_release_acceptance_v5 as v5


gate = v5.gate

MODELS = (
    "lineborn-v6-balanced",
    "lineborn-v6-performance",
)

# Keep the established 4B production decoding profiles so this benchmark measures
# the trained model/quantization change rather than tuning new inference settings.
PRODUCTION_OPTIONS = {
    "lineborn-v6-balanced": {"temperature": 0.05, "top_p": 0.72, "num_predict": 68},
    "lineborn-v6-performance": {"temperature": 0.08, "top_p": 0.76, "num_predict": 72},
}

gate.MODELS = MODELS
gate.PRODUCTION_OPTIONS = PRODUCTION_OPTIONS

def require_local_model(model: str) -> None:
    """Release candidates are local Lineborn GGUF imports, not registry models."""
    result = subprocess.run(
        ["ollama", "show", model],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Local Ollama model {model!r} is missing. Import the Lineborn GGUF before running v6 acceptance."
        )

gate.pull = require_local_model

if __name__ == "__main__":
    raise SystemExit(gate.main())
