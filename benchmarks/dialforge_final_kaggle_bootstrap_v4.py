#!/usr/bin/env python3
"""Kaggle bootstrap for Dialforge Final Release Acceptance v4."""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

import dialforge_final_colab_bootstrap_v3 as legacy

ROOT = pathlib.Path("/kaggle/working") if pathlib.Path("/kaggle/working").exists() else pathlib.Path.cwd()
BENCH = pathlib.Path(__file__).resolve().parent
REPO = BENCH.parent
OUT = ROOT / "dialforge-final-release-v4"
ZIP = ROOT / "dialforge-final-release-v4-results.zip"

# Reuse the heavily tested Python/Ollama installer from Final Acceptance v3, but point
# all mutable state at Kaggle's persistent working directory for this session.
legacy.ROOT = ROOT
legacy.VENV = ROOT / "dialforge-final-py311-v1"
legacy.CACHE = ROOT / "dialforge-final-cache"
legacy.OLLAMA_ARCHIVE = legacy.CACHE / "ollama-linux-amd64.tar.zst"
legacy.OLLAMA_LOG = ROOT / "dialforge-final-ollama.log"


def main() -> int:
    print("=== DIALFORGE FINAL RELEASE v4 KAGGLE BOOTSTRAP ===", flush=True)
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        text=True, capture_output=True,
    )
    if gpu.returncode != 0 or not gpu.stdout.strip():
        raise RuntimeError("No NVIDIA GPU detected. Enable a Kaggle GPU accelerator first.")
    print("GPU(s):\n" + gpu.stdout.strip(), flush=True)

    legacy.apt_setup()
    legacy.install_ollama()
    python = legacy.ensure_py311()
    legacy.setup_python_stack(python)
    legacy.validate_stack(python)
    legacy.start_ollama()

    runner = BENCH / "dialforge_final_release_acceptance_v4.py"
    if not runner.exists():
        raise RuntimeError(f"Final runner missing: {runner}")

    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)
    env = legacy.benchmark_env()
    env.update({
        "PYTHONPATH": str(BENCH),
        "OLLAMA_KEEP_ALIVE": "30m",
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_MAX_LOADED_MODELS": "1",
    })
    result = subprocess.run(
        [str(python), str(runner), "--output-dir", str(OUT)],
        cwd=str(BENCH), env=env,
    )

    if ZIP.exists():
        ZIP.unlink()
    shutil.make_archive(str(ZIP.with_suffix("")), "zip", root_dir=str(OUT))
    print(f"\nResults folder: {OUT}", flush=True)
    print(f"Results ZIP: {ZIP}", flush=True)
    report = OUT / "dialforge-final-release-v4.json"
    if report.exists():
        print(f"Report: {report}", flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
