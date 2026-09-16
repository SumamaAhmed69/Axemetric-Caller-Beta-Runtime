#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
)
OPTIONAL_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_adapter_dir(root: Path) -> dict:
    adapter = root / "adapter"
    missing = [name for name in REQUIRED_ADAPTER_FILES if not (adapter / name).is_file()]
    if missing:
        raise RuntimeError(f"{root}: training output is incomplete; missing {', '.join(missing)}")
    model_size = (adapter / "adapter_model.safetensors").stat().st_size
    if model_size < 1024:
        raise RuntimeError(f"{root}: adapter_model.safetensors is suspiciously small ({model_size} bytes)")
    metrics_path = root / "training_metrics.json"
    metrics = json.loads(metrics_path.read_text("utf-8")) if metrics_path.is_file() else None
    files = []
    for path in sorted(adapter.rglob("*")):
        if path.is_file():
            files.append({
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    return {
        "root": str(root),
        "adapter_bytes": model_size,
        "metrics": metrics,
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="training/output")
    parser.add_argument("--archive", default="/content/lineborn-sales-adapters.zip")
    parser.add_argument("--require-dpo", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    sft = output_root / "lineborn-sales-sft"
    dpo = output_root / "lineborn-sales-dpo"

    stages = {"sft": validate_adapter_dir(sft)}
    if dpo.exists():
        stages["dpo"] = validate_adapter_dir(dpo)
    elif args.require_dpo:
        raise RuntimeError("DPO output is required but training/output/lineborn-sales-dpo does not exist")

    archive = Path(args.archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        archive.unlink()

    manifest = {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stages": stages,
        "recommended_adapter": "lineborn-sales-dpo/adapter" if "dpo" in stages else "lineborn-sales-sft/adapter",
    }
    manifest_path = output_root / "lineborn-sales-export-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for stage_dir in (sft, dpo):
            if not stage_dir.exists():
                continue
            for path in sorted(stage_dir.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(output_root))
        zf.write(manifest_path, manifest_path.name)

    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        required = {
            "lineborn-sales-sft/adapter/adapter_config.json",
            "lineborn-sales-sft/adapter/adapter_model.safetensors",
            "lineborn-sales-export-manifest.json",
        }
        missing = sorted(required - names)
        if missing:
            raise RuntimeError(f"archive verification failed; missing {', '.join(missing)}")

    size = archive.stat().st_size
    if size < 1024:
        raise RuntimeError(f"archive is suspiciously small ({size} bytes)")
    print(json.dumps({"archive": str(archive), "bytes": size, "sha256": sha256(archive), **manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
