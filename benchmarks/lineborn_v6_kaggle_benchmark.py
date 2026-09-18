#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import time

import dialforge_final_colab_bootstrap_v3 as legacy

ROOT = pathlib.Path("/kaggle/working")
SCRATCH = pathlib.Path("/kaggle/tmp/lineborn-v6-benchmark")
BENCH = pathlib.Path(__file__).resolve().parent
MODEL_DIR = ROOT / "lineborn-v6-export"
OUT = ROOT / "lineborn-v6-release-acceptance"
RAW_REPORT = OUT / "dialforge-final-release-v4.json"
REPORT = OUT / "lineborn-v6-release-acceptance.json"
ZIP = ROOT / "lineborn-v6-release-acceptance-results.zip"

BALANCED = MODEL_DIR / "lineborn-v6-balanced-q4_k_m.gguf"
PERFORMANCE = MODEL_DIR / "lineborn-v6-performance-q8_0.gguf"
MODELS = ("lineborn-v6-balanced", "lineborn-v6-performance")

legacy.ROOT = SCRATCH
legacy.VENV = SCRATCH / "py311"
legacy.CACHE = SCRATCH / "cache"
legacy.OLLAMA_ARCHIVE = legacy.CACHE / "ollama-linux-amd64.tar.zst"
legacy.OLLAMA_LOG = SCRATCH / "ollama.log"


def run(cmd, *, cwd=None, env=None, check=True):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, cwd=cwd, env=env, check=check)


def ensure_inputs() -> None:
    missing = [p for p in (BALANCED, PERFORMANCE, MODEL_DIR / "Modelfile.balanced", MODEL_DIR / "Modelfile.performance") if not p.is_file()]
    if missing:
        raise RuntimeError("Missing trained Lineborn output(s): " + ", ".join(str(p) for p in missing))
    print("Trained GGUF inputs found:", flush=True)
    print(f"  Balanced:    {BALANCED.stat().st_size / 1024**3:.3f} GiB", flush=True)
    print(f"  Performance: {PERFORMANCE.stat().st_size / 1024**3:.3f} GiB", flush=True)


def import_ollama_models() -> None:
    for tag, modelfile in (
        ("lineborn-v6-balanced", MODEL_DIR / "Modelfile.balanced"),
        ("lineborn-v6-performance", MODEL_DIR / "Modelfile.performance"),
    ):
        print(f"Importing {tag} into local Ollama...", flush=True)
        run(["ollama", "create", tag, "-f", str(modelfile)], cwd=str(MODEL_DIR))


def main() -> int:
    ensure_inputs()
    SCRATCH.mkdir(parents=True, exist_ok=True)

    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        text=True,
    ).strip().splitlines()
    if len(gpu) < 2:
        raise RuntimeError("This benchmark expects Kaggle T4 x2 so Ollama and the voice stack can use separate GPUs.")
    print("GPU(s):", *gpu, sep="\n  ", flush=True)

    legacy.apt_setup()
    legacy.install_ollama()
    python = legacy.ensure_py311()
    legacy.setup_python_stack(python)
    legacy.validate_stack(python)

    # Keep Ollama on the second T4. The benchmark Python process uses GPU 0 for
    # Whisper/Chatterbox, avoiding VRAM contention.
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["OLLAMA_MODELS"] = str(SCRATCH / "ollama-models")
    legacy.start_ollama()
    import_ollama_models()

    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)

    env = legacy.benchmark_env()  # sets CUDA_VISIBLE_DEVICES=0 for voice/STT/TTS
    env.update({
        "PYTHONPATH": str(BENCH),
        "OLLAMA_KEEP_ALIVE": "30m",
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_MAX_LOADED_MODELS": "1",
        "OLLAMA_MODELS": str(SCRATCH / "ollama-models"),
    })

    runner = BENCH / "lineborn_trained_release_acceptance_v6.py"
    command = [
        str(python),
        str(runner),
        "--output-dir", str(OUT),
        "--models", *MODELS,
    ]
    started = time.time()
    result = subprocess.run(command, cwd=str(BENCH), env=env)
    elapsed = time.time() - started

    if not RAW_REPORT.is_file():
        raise RuntimeError(
            f"v6 benchmark exited with code {result.returncode} without producing {RAW_REPORT}"
        )

    report = json.loads(RAW_REPORT.read_text("utf-8"))
    report["product"] = "Lineborn"
    report["benchmark"] = "lineborn-trained-release-acceptance-v6-kaggle"
    report["benchmark_wall_seconds"] = round(elapsed, 2)
    report["model_artifacts"] = {
        "lineborn-v6-balanced": str(BALANCED),
        "lineborn-v6-performance": str(PERFORMANCE),
    }
    report["scope_note"] = (
        "Strict trained-model AI/product/voice gate. Real Direct SIP/PSTN and clean Windows installer "
        "acceptance remain separate release gates."
    )
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if ZIP.exists():
        ZIP.unlink()
    shutil.make_archive(str(ZIP.with_suffix("")), "zip", root_dir=str(OUT))

    print("\n" + "=" * 80, flush=True)
    print("LINEBORN V6 TRAINED RELEASE GATE:", "PASS" if report.get("release_gate_pass") else "BLOCKED", flush=True)
    print("=" * 80, flush=True)
    for model, data in (report.get("models") or {}).items():
        sales = data.get("sales") or {}
        tools = data.get("tools") or {}
        voice = data.get("voice_speed") or {}
        failed = [k for k, v in (data.get("gate_checks") or {}).items() if not v]
        print(
            f"{model}: {'PASS' if data.get('passed') else 'FAIL'} | "
            f"sales {sales.get('product_score')} | raw {sales.get('raw_score')} | "
            f"tools {tools.get('accuracy')} | critical {sales.get('product_critical')} | "
            f"voice median {voice.get('voice_start_median_ms')} ms | voice p95 {voice.get('voice_start_p95_ms')} ms",
            flush=True,
        )
        if failed:
            print("  failed:", ", ".join(failed), flush=True)

    print("\nCanonical report:", REPORT, flush=True)
    print("Results ZIP:", ZIP, flush=True)
    return 0 if report.get("release_gate_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
