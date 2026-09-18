#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

EXPECTED_ARCHIVE_BYTES = 1_800_343_183
EXPECTED_ARCHIVE_SHA256 = "ea657b97036665d0e9cbd95b09523f0a224ce40485b3fc9e02361b816e09f406"
EXPECTED_PART_BYTES = {i: 225_042_898 for i in range(1, 8)} | {8: 225_042_897}
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working")
SCRATCH_ROOT = Path("/kaggle/tmp/lineborn-v6-build")
ARCHIVE = SCRATCH_ROOT / "lineborn-sales-adapters-v6.zip"
EXTRACT_DIR = SCRATCH_ROOT / "lineborn-v6-adapters"
EXPORT_DIR = SCRATCH_ROOT / "lineborn-v6-export"
FINAL_DIR = WORK_ROOT / "lineborn-v6-export"

def run(cmd, **kwargs):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, check=True, **kwargs)

def preflight() -> None:
    print("=== Lineborn v6 Kaggle deployment ===", flush=True)
    print("Python:", sys.version.split()[0], flush=True)
    try:
        run(["nvidia-smi"])
    except Exception as exc:
        raise SystemExit(
            "No NVIDIA GPU detected. In Kaggle Notebook settings choose Accelerator > GPU T4 x2."
        ) from exc
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    _total, _used, free = shutil.disk_usage(SCRATCH_ROOT)
    print(f"Scratch disk free: {free / 1024**3:.1f} GiB", flush=True)
    if free < 30 * 1024**3:
        raise SystemExit(
            "Kaggle scratch space is below 30 GiB in this session. Restart the Kaggle session and rerun."
        )
    _wt, _wu, working_free = shutil.disk_usage(WORK_ROOT)
    print(f"/kaggle/working persistent space free: {working_free / 1024**3:.1f} GiB", flush=True)
    if working_free < 8 * 1024**3:
        raise SystemExit("Need at least ~8 GiB free in /kaggle/working for the final Q4 + Q8 outputs.")
    if not INPUT_ROOT.exists():
        raise SystemExit("/kaggle/input is missing. Attach the Dataset containing all 8 adapter parts.")

def find_parts() -> list[Path]:
    parts: list[Path] = []
    print("Searching attached Kaggle inputs for the 8 Lineborn adapter parts...", flush=True)
    for i in range(1, 9):
        name = f"lineborn-sales-adapters-v6.zip.part{i}"
        matches = [
            p for p in INPUT_ROOT.rglob(name)
            if p.is_file() and p.stat().st_size == EXPECTED_PART_BYTES[i]
        ]
        if not matches:
            raise FileNotFoundError(
                f"Missing {name} ({EXPECTED_PART_BYTES[i]:,} bytes). "
                "Upload all eight parts as one Kaggle Dataset and attach it to this notebook."
            )
        if len(matches) > 1:
            print(f"Warning: found {len(matches)} copies of {name}; using {matches[0]}", flush=True)
        parts.append(matches[0])
    print("Found all 8 parts:", flush=True)
    for p in parts:
        print(f"  {p}: {p.stat().st_size:,} bytes", flush=True)
    return parts

def reconstruct_and_verify(parts: list[Path]) -> None:
    if ARCHIVE.exists():
        if ARCHIVE.stat().st_size == EXPECTED_ARCHIVE_BYTES:
            h = hashlib.sha256()
            with ARCHIVE.open("rb") as f:
                for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
                    h.update(block)
            if h.hexdigest() == EXPECTED_ARCHIVE_SHA256:
                print("✅ Existing reconstructed archive already verified; reusing it.", flush=True)
                return
        ARCHIVE.unlink()

    h = hashlib.sha256()
    written = 0
    started = time.time()
    with ARCHIVE.open("wb") as dst:
        for idx, part in enumerate(parts, 1):
            print(f"Joining part {idx}/8: {part.name}", flush=True)
            with part.open("rb") as src:
                while True:
                    chunk = src.read(16 * 1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
                    h.update(chunk)
                    written += len(chunk)
            print(f"  total written: {written / 1024**3:.3f} GiB", flush=True)

    sha = h.hexdigest()
    print("Archive bytes:", written, flush=True)
    print("Archive SHA256:", sha, flush=True)
    print(f"Join+hash elapsed: {time.time() - started:.1f}s", flush=True)
    if written != EXPECTED_ARCHIVE_BYTES:
        raise RuntimeError(f"Archive size mismatch: {written} != {EXPECTED_ARCHIVE_BYTES}")
    if sha != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError(f"Archive SHA mismatch: {sha} != {EXPECTED_ARCHIVE_SHA256}")
    print("✅ Exact Lineborn v6 adapter archive verified.", flush=True)

def prepare_environment() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    # Keep large HF cache outside /kaggle/working so notebook output stays compact.
    os.environ.setdefault("HF_HOME", "/kaggle/tmp/lineborn-hf-cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/kaggle/tmp/lineborn-hf-cache")
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "training/requirements-colab.txt"])

    # Kaggle currently preinstalls an old torchao build that PEFT 0.21+ detects
    # and rejects before a normal BF16 LoRA adapter can even be injected.
    # Lineborn does not use TorchAO for this merge; GGUF quantization is handled
    # later by llama.cpp. Remove the optional stale package so PEFT follows its
    # standard torch.nn.Linear LoRA path.
    try:
        from importlib.metadata import PackageNotFoundError, version
        torchao_version = version("torchao")
        print(f"Removing Kaggle preinstalled torchao {torchao_version}; not needed for Lineborn BF16 merge.", flush=True)
        run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])
    except PackageNotFoundError:
        print("torchao is not installed; continuing.", flush=True)

    missing = [tool for tool in ("cmake", "g++", "git") if shutil.which(tool) is None]
    if missing:
        print("Installing missing system tools:", ", ".join(missing), flush=True)
        run(["apt-get", "update", "-qq"])
        run(["apt-get", "install", "-y", "-qq", "build-essential", "cmake", "ninja-build", "git"])

def verify_extract() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    adapter = EXTRACT_DIR / "lineborn-sales-dpo" / "adapter"
    if (adapter / "adapter_config.json").is_file() and (adapter / "adapter_model.safetensors").is_file():
        print("✅ Adapter already extracted; reusing:", adapter, flush=True)
        return adapter
    if EXTRACT_DIR.exists():
        shutil.rmtree(EXTRACT_DIR)
    run([
        sys.executable,
        str(repo_root / "training/verify_lineborn_v6_export.py"),
        str(ARCHIVE),
        "--extract-dir", str(EXTRACT_DIR),
    ])
    if not (adapter / "adapter_config.json").is_file():
        raise RuntimeError("DPO adapter_config.json missing after extraction")
    if not (adapter / "adapter_model.safetensors").is_file():
        raise RuntimeError("DPO adapter_model.safetensors missing after extraction")
    print("✅ Final DPO adapter:", adapter, flush=True)
    return adapter

def build_candidates(adapter: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LINEBORN_BASE_MODEL"] = BASE_MODEL
    env["LINEBORN_BALANCED_QUANT"] = "Q4_K_M"
    env["LINEBORN_PERFORMANCE_QUANT"] = "Q8_0"
    env["LINEBORN_KEEP_F16"] = "0"
    env["LINEBORN_PYTHON"] = sys.executable
    env["HF_HOME"] = "/kaggle/tmp/lineborn-hf-cache"
    env["TRANSFORMERS_CACHE"] = "/kaggle/tmp/lineborn-hf-cache"
    # One T4 has enough memory for the 4B FP16 merge; leave the second T4 free.
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    run([
        "bash",
        str(repo_root / "training/export_lineborn_v6_gguf.sh"),
        str(adapter),
        str(EXPORT_DIR),
    ], env=env)

def validate_outputs() -> list[str]:
    required = [
        "lineborn-v6-balanced-q4_k_m.gguf",
        "lineborn-v6-performance-q8_0.gguf",
        "Modelfile.balanced",
        "Modelfile.performance",
        "lineborn-v6-deployment-manifest.json",
    ]
    manifest_path = EXPORT_DIR / "lineborn-v6-deployment-manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Deployment manifest was not produced.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("base_model") != BASE_MODEL:
        raise RuntimeError(f"Unexpected base model in manifest: {manifest.get('base_model')}")
    for name in required:
        p = EXPORT_DIR / name
        if not p.is_file() or p.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty output: {p}")
        if p.suffix == ".gguf":
            print(f"{name}: {p.stat().st_size / 1024**3:.3f} GiB", flush=True)
        else:
            print(f"{name}: {p.stat().st_size:,} bytes", flush=True)
    print("\nDeployment manifest:\n", json.dumps(manifest, indent=2), flush=True)
    return required

def persist_outputs_and_cleanup(required: list[str]) -> None:
    # Build entirely on Kaggle scratch disk, then copy only final deployable artifacts
    # into /kaggle/working so they are eligible for notebook output persistence.
    print("\nCopying final Lineborn artifacts into persistent /kaggle/working...", flush=True)
    if FINAL_DIR.exists():
        shutil.rmtree(FINAL_DIR)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    for name in required:
        src = EXPORT_DIR / name
        dst = FINAL_DIR / name
        shutil.copy2(src, dst)
        if src.stat().st_size != dst.stat().st_size:
            raise RuntimeError(f"Persistent copy size mismatch for {name}")
        print("  saved:", dst, flush=True)

    total = sum((FINAL_DIR / name).stat().st_size for name in required)
    print(f"Final persistent output footprint: {total / 1024**3:.3f} GiB", flush=True)

    print("Cleaning scratch build data...", flush=True)
    shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)
    shutil.rmtree(Path("/kaggle/tmp/lineborn-hf-cache"), ignore_errors=True)
    print("✅ Final files are in:", FINAL_DIR, flush=True)

def main() -> int:
    preflight()
    parts = find_parts()
    reconstruct_and_verify(parts)
    prepare_environment()
    adapter = verify_extract()
    build_candidates(adapter)
    required = validate_outputs()
    persist_outputs_and_cleanup(required)
    print("\n✅ Lineborn v6 Kaggle candidate build complete.", flush=True)
    print("Save/Commit the notebook version so Kaggle persists /kaggle/working/lineborn-v6-export.", flush=True)
    print("Next gate: benchmarks/lineborn_trained_release_acceptance_v6.py", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
