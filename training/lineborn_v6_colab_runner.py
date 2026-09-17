#!/usr/bin/env python3
from __future__ import annotations

import gc
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

ROOT = Path("/content")
DRIVE_ROOT = Path("/content/drive/MyDrive")
UPLOAD_DIR = ROOT / "lineborn-upload-parts"
ARCHIVE = ROOT / "lineborn-sales-adapters-v6.zip"
EXTRACT_DIR = ROOT / "lineborn-v6-adapters"
EXPORT_DIR = ROOT / "lineborn-v6-export"
OUTPUT_DRIVE = DRIVE_ROOT / "Lineborn-v6-export"


def run(cmd, **kwargs):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, check=True, **kwargs)


def preflight() -> None:
    print("=== Lineborn v6 Colab deployment ===", flush=True)
    print("Python:", sys.version.split()[0], flush=True)
    try:
        run(["nvidia-smi"])
    except Exception as exc:
        raise SystemExit(
            "GPU runtime not detected. In Colab choose Runtime > Change runtime type > T4 GPU, then Run all."
        ) from exc
    _total, _used, free = shutil.disk_usage(ROOT)
    print(f"Local disk free: {free / 1024**3:.1f} GiB", flush=True)
    if free < 30 * 1024**3:
        raise SystemExit("At least ~30 GiB of free Colab disk is recommended for merge + GGUF quantization.")


def try_mount_drive() -> bool:
    try:
        from google.colab import drive

        print("Attempting Google Drive mount...", flush=True)
        drive.mount("/content/drive", force_remount=False)
        ok = DRIVE_ROOT.exists()
        print("Drive mounted." if ok else "Drive mount returned but MyDrive was not found.", flush=True)
        return ok
    except Exception as exc:
        print("\nWARNING: Google Drive mount failed.", flush=True)
        print(f"Colab auth error: {exc!r}", flush=True)
        print("Continuing in manual-upload mode. This bypasses Drive credential propagation entirely.\n", flush=True)
        return False


def find_parts_on_drive() -> list[Path]:
    parts: list[Path] = []
    for i in range(1, 9):
        name = f"lineborn-sales-adapters-v6.zip.part{i}"
        matches = [
            p
            for p in DRIVE_ROOT.rglob(name)
            if p.is_file() and p.stat().st_size == EXPECTED_PART_BYTES[i]
        ]
        if not matches:
            raise FileNotFoundError(
                f"Could not find {name} with expected size {EXPECTED_PART_BYTES[i]:,} bytes under {DRIVE_ROOT}"
            )
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        parts.append(matches[0])
    return parts


def upload_parts_manually() -> list[Path]:
    from google.colab import files

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    print("Upload each requested file when prompted. Select exactly ONE file each time.", flush=True)
    print("The notebook does this one part at a time to avoid holding the entire 1.8 GB archive in RAM.\n", flush=True)

    for i in range(1, 9):
        expected_name = f"lineborn-sales-adapters-v6.zip.part{i}"
        expected_size = EXPECTED_PART_BYTES[i]
        target = UPLOAD_DIR / expected_name

        if target.exists() and target.stat().st_size == expected_size:
            print(f"Already present: {expected_name}", flush=True)
            parts.append(target)
            continue

        print(f"\nUpload {expected_name} ({expected_size:,} bytes)", flush=True)
        uploaded = files.upload()
        if len(uploaded) != 1:
            raise RuntimeError(f"Please upload exactly one file for part {i}.")
        uploaded_name, uploaded_bytes = next(iter(uploaded.items()))
        if uploaded_name != expected_name:
            raise RuntimeError(f"Expected {expected_name}, got {uploaded_name}")
        if len(uploaded_bytes) != expected_size:
            raise RuntimeError(
                f"{uploaded_name} size mismatch: {len(uploaded_bytes):,} != {expected_size:,}"
            )
        target.write_bytes(uploaded_bytes)
        del uploaded, uploaded_bytes
        gc.collect()
        print("Saved:", target, flush=True)
        parts.append(target)
    return parts


def reconstruct_and_verify(parts: list[Path]) -> None:
    if ARCHIVE.exists():
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
    print("Exact Lineborn v6 adapter archive verified.\n", flush=True)


def prepare_build_environment() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "training/requirements-colab.txt"])
    run(["apt-get", "update", "-qq"])
    run(["apt-get", "install", "-y", "-qq", "build-essential", "cmake", "ninja-build"])


def verify_extract() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    if EXTRACT_DIR.exists():
        shutil.rmtree(EXTRACT_DIR)
    run(
        [
            sys.executable,
            str(repo_root / "training/verify_lineborn_v6_export.py"),
            str(ARCHIVE),
            "--extract-dir",
            str(EXTRACT_DIR),
        ]
    )
    adapter = EXTRACT_DIR / "lineborn-sales-dpo" / "adapter"
    if not (adapter / "adapter_config.json").is_file():
        raise RuntimeError("DPO adapter_config.json missing after extraction")
    if not (adapter / "adapter_model.safetensors").is_file():
        raise RuntimeError("DPO adapter_model.safetensors missing after extraction")
    print("Final DPO adapter:", adapter, flush=True)
    return adapter


def build_candidates(adapter: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)
    env = os.environ.copy()
    env["LINEBORN_BASE_MODEL"] = BASE_MODEL
    env["LINEBORN_BALANCED_QUANT"] = "Q4_K_M"
    env["LINEBORN_PERFORMANCE_QUANT"] = "Q8_0"
    env["LINEBORN_KEEP_F16"] = "0"
    run(
        [
            "bash",
            str(repo_root / "training/export_lineborn_v6_gguf.sh"),
            str(adapter),
            str(EXPORT_DIR),
        ],
        env=env,
    )


def validate_outputs() -> list[str]:
    required = [
        "lineborn-v6-balanced-q4_k_m.gguf",
        "lineborn-v6-performance-q8_0.gguf",
        "Modelfile.balanced",
        "Modelfile.performance",
        "lineborn-v6-deployment-manifest.json",
    ]
    manifest_path = EXPORT_DIR / "lineborn-v6-deployment-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("base_model") != BASE_MODEL:
        raise RuntimeError(f"Unexpected base model in deployment manifest: {manifest.get('base_model')}")
    for name in required:
        path = EXPORT_DIR / name
        if not path.is_file():
            raise RuntimeError(f"Missing output: {path}")
        if path.suffix == ".gguf":
            print(f"{name}: {path.stat().st_size / 1024**3:.3f} GiB", flush=True)
        else:
            print(f"{name}: {path.stat().st_size:,} bytes", flush=True)
    print("\nDeployment manifest:\n", json.dumps(manifest, indent=2), flush=True)
    return required


def save_outputs_to_drive(required: list[str], drive_available: bool) -> None:
    if not drive_available:
        print("\nDrive authentication was unavailable, so outputs remain in the Colab runtime:", flush=True)
        for name in required:
            print(" ", EXPORT_DIR / name, flush=True)
        print(
            "\nOpen the Colab Files sidebar (folder icon), browse to /content/lineborn-v6-export, "
            "and download the GGUFs, Modelfiles, and manifest.",
            flush=True,
        )
        return

    OUTPUT_DRIVE.mkdir(parents=True, exist_ok=True)
    for name in required:
        src = EXPORT_DIR / name
        dst = OUTPUT_DRIVE / name
        print("Copying", name, "to Drive...", flush=True)
        shutil.copy2(src, dst)
        if dst.stat().st_size != src.stat().st_size:
            raise RuntimeError(f"Drive copy size mismatch for {name}")

    (OUTPUT_DRIVE / "lineborn-v6-source.txt").write_text(
        f"base_model={BASE_MODEL}\n"
        f"adapter_archive_sha256={EXPECTED_ARCHIVE_SHA256}\n"
        f"adapter_archive_bytes={EXPECTED_ARCHIVE_BYTES}\n"
        "repo_branch=lineborn-v6-deployment\n",
        encoding="utf-8",
    )
    print("\nFinished. Saved to:", OUTPUT_DRIVE, flush=True)


def main() -> int:
    preflight()
    drive_available = try_mount_drive()
    parts = find_parts_on_drive() if drive_available else upload_parts_manually()

    print("\nInput parts:", flush=True)
    for part in parts:
        print(f"  {part.name}: {part.stat().st_size:,} bytes", flush=True)

    reconstruct_and_verify(parts)
    prepare_build_environment()
    adapter = verify_extract()
    build_candidates(adapter)
    required = validate_outputs()
    save_outputs_to_drive(required, drive_available)

    print("\nLineborn v6 candidate build complete.", flush=True)
    print("Next gate: benchmarks/lineborn_trained_release_acceptance_v6.py", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
