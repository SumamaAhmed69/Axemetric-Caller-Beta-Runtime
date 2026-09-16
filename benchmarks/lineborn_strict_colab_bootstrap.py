#!/usr/bin/env python3
"""Direct Google Colab strict pre-release benchmark for Lineborn.

This benchmark runs entirely inside the temporary Colab VM. It does not require the
Lineborn desktop app, a Cloudflare tunnel, SIP credentials, or a manually uploaded
voice reference. It reuses the frozen Final Release Acceptance v4 corpus and scoring
so results remain comparable with prior release-gate runs.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import time

import dialforge_final_colab_bootstrap_v3 as legacy

ROOT = pathlib.Path("/content") if pathlib.Path("/content").exists() else pathlib.Path.cwd()
BENCH = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "lineborn-strict-prerelease"
REPORT = OUT / "lineborn-strict-prerelease.json"
RAW_REPORT = OUT / "dialforge-final-release-v4.json"
ZIP = ROOT / "lineborn-strict-prerelease-results.zip"
MODELS = (
    "qwen3:1.7b",
    "qwen3:4b-instruct",
    "qwen3:4b-instruct-2507-q8_0",
)

# Reuse the already-tested Python 3.11 / CUDA / Ollama / Whisper / Chatterbox installer.
legacy.ROOT = ROOT
legacy.VENV = ROOT / "lineborn-strict-py311-v1"
legacy.CACHE = ROOT / "lineborn-strict-cache"
legacy.OLLAMA_ARCHIVE = legacy.CACHE / "ollama-linux-amd64.tar.zst"
legacy.OLLAMA_LOG = ROOT / "lineborn-strict-ollama.log"


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(BENCH.parent), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def gpu_info() -> tuple[str, int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(
            "No NVIDIA GPU detected. In Colab choose Runtime > Change runtime type > GPU."
        )
    first = result.stdout.strip().splitlines()[0]
    name, memory = [part.strip() for part in first.rsplit(",", 1)]
    return name, int(memory)


def selected_models(requested: list[str] | None) -> list[str]:
    if not requested or requested == ["all"]:
        return list(MODELS)
    invalid = [m for m in requested if m not in MODELS]
    if invalid:
        raise ValueError(f"Unsupported model(s): {', '.join(invalid)}")
    return list(dict.fromkeys(requested))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="all or one/more Lineborn model IDs",
    )
    args = parser.parse_args()
    models = selected_models(args.models)

    print("=" * 80, flush=True)
    print("LINEBORN STRICT PRE-RELEASE BENCHMARK · DIRECT COLAB", flush=True)
    print("=" * 80, flush=True)
    gpu_name, vram_mb = gpu_info()
    print(f"GPU: {gpu_name} ({vram_mb} MiB)", flush=True)
    print("Models:", ", ".join(models), flush=True)
    print("Source:", git_sha(), flush=True)
    print("Mode: direct Colab; no desktop app, tunnel, SIP, or remote pairing", flush=True)

    legacy.apt_setup()
    legacy.install_ollama()
    python = legacy.ensure_py311()
    legacy.setup_python_stack(python)
    legacy.validate_stack(python)
    legacy.start_ollama()

    runner = BENCH / "dialforge_final_release_acceptance_v4_kaggle.py"
    if not runner.exists():
        raise RuntimeError(f"Frozen release-gate runner missing: {runner}")

    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)

    env = legacy.benchmark_env()
    env.update(
        {
            "PYTHONPATH": str(BENCH),
            "OLLAMA_KEEP_ALIVE": "30m",
            "OLLAMA_NUM_PARALLEL": "1",
            "OLLAMA_MAX_LOADED_MODELS": "1",
        }
    )

    started = time.time()
    command = [str(python), str(runner), "--output-dir", str(OUT), "--models", *models]
    result = subprocess.run(command, cwd=str(BENCH), env=env)
    elapsed = time.time() - started

    if not RAW_REPORT.exists():
        raise RuntimeError(
            f"Benchmark exited with code {result.returncode} without producing {RAW_REPORT.name}."
        )

    report = json.loads(RAW_REPORT.read_text("utf-8"))
    report["product"] = "Lineborn"
    report["benchmark"] = "lineborn-strict-prerelease-direct-colab-v1"
    report["benchmark_mode"] = "direct-colab-no-desktop"
    report["source_commit"] = git_sha()
    report["benchmark_wall_seconds"] = round(elapsed, 2)
    report["requested_models"] = models
    report["colab_gpu"] = {"name": gpu_name, "vram_mb": vram_mb}
    report["scope_note"] = (
        "Strict AI/product-path gate only. Direct SIP/PSTN, carrier/NAT behavior, installer, "
        "desktop integration, and long campaign soak testing remain separate release gates."
    )
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Keep the raw report for auditability, but provide Lineborn-branded canonical output.
    if ZIP.exists():
        ZIP.unlink()
    shutil.make_archive(str(ZIP.with_suffix("")), "zip", root_dir=str(OUT))

    strict_pass = bool(report.get("release_gate_pass"))
    print("\n" + "=" * 80, flush=True)
    print("STRICT PRE-RELEASE RESULT:", "PASS" if strict_pass else "BLOCKED", flush=True)
    print("=" * 80, flush=True)
    for model, data in (report.get("models") or {}).items():
        sales = data.get("sales") or {}
        tools = data.get("tools") or {}
        stream = data.get("stream_speed") or {}
        voice = data.get("voice_speed") or {}
        print(
            f"{model}: {'PASS' if data.get('passed') else 'FAIL'} | "
            f"sales {sales.get('product_score')} | tools {tools.get('accuracy')} | "
            f"TTFT {stream.get('ttft_median_ms')} ms | "
            f"voice median {voice.get('voice_start_median_ms')} ms | "
            f"voice p95 {voice.get('voice_start_p95_ms')} ms",
            flush=True,
        )
        failed = [k for k, v in (data.get("gate_checks") or {}).items() if not v]
        if failed:
            print("  failed gates:", ", ".join(failed), flush=True)

    print(f"\nCanonical JSON: {REPORT}", flush=True)
    print(f"Full results ZIP: {ZIP}", flush=True)
    print(f"Wall time: {elapsed:.1f}s", flush=True)
    return 0 if strict_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
