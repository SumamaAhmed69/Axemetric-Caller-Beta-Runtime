#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

EXPECTED_BASE = "Qwen/Qwen3-4B-Instruct-2507"
EXPECTED_ARCHIVE_SHA256 = "ea657b97036665d0e9cbd95b09523f0a224ce40485b3fc9e02361b816e09f406"
MANIFEST_NAME = "lineborn-sales-export-manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_stream(handle) -> str:
    h = hashlib.sha256()
    for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
        h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify and optionally extract the Lineborn v6 sales adapter export.")
    ap.add_argument("archive", type=Path)
    ap.add_argument("--extract-dir", type=Path)
    ap.add_argument("--expected-sha256", default=EXPECTED_ARCHIVE_SHA256)
    ap.add_argument("--skip-archive-sha", action="store_true")
    ap.add_argument("--skip-crc", action="store_true")
    args = ap.parse_args()

    archive = args.archive.expanduser().resolve()
    if not archive.is_file():
        raise SystemExit(f"archive not found: {archive}")

    archive_sha = None
    if not args.skip_archive_sha:
        archive_sha = sha256_file(archive)
        if args.expected_sha256 and archive_sha.lower() != args.expected_sha256.lower():
            raise SystemExit(f"archive SHA-256 mismatch: {archive_sha} != {args.expected_sha256}")

    with zipfile.ZipFile(archive) as zf:
        if not args.skip_crc:
            broken = zf.testzip()
            if broken:
                raise SystemExit(f"ZIP CRC failure: {broken}")
        names = set(zf.namelist())
        if MANIFEST_NAME not in names:
            raise SystemExit(f"missing {MANIFEST_NAME}")
        manifest = json.loads(zf.read(MANIFEST_NAME).decode("utf-8"))

        recommended = manifest.get("recommended_adapter")
        if recommended != "lineborn-sales-dpo/adapter":
            raise SystemExit(f"unexpected recommended adapter: {recommended!r}")

        verified = []
        for stage_name, stage in (manifest.get("stages") or {}).items():
            metrics = stage.get("metrics") or {}
            base_model = metrics.get("base_model")
            if base_model != EXPECTED_BASE:
                raise SystemExit(f"{stage_name}: wrong base model {base_model!r}; expected {EXPECTED_BASE}")
            root_name = Path(str(stage.get("root") or "")).name
            if not root_name:
                raise SystemExit(f"{stage_name}: invalid stage root")
            for item in stage.get("files") or []:
                member = f"{root_name}/{item['path']}"
                if member not in names:
                    raise SystemExit(f"manifest member missing from ZIP: {member}")
                with zf.open(member) as handle:
                    digest = sha256_stream(handle)
                if digest.lower() != str(item["sha256"]).lower():
                    raise SystemExit(f"SHA-256 mismatch for {member}: {digest} != {item['sha256']}")
                verified.append(member)

        if args.extract_dir:
            extract_dir = args.extract_dir.expanduser().resolve()
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            extract_dir.mkdir(parents=True, exist_ok=True)
            zf.extractall(extract_dir)
        else:
            extract_dir = None

    report = {
        "archive": str(archive),
        "archive_sha256": archive_sha,
        "expected_base_model": EXPECTED_BASE,
        "recommended_adapter": manifest["recommended_adapter"],
        "manifest_files_verified": len(verified),
        "extract_dir": str(extract_dir) if extract_dir else None,
        "sft_metrics": manifest["stages"]["sft"].get("metrics"),
        "dpo_metrics": manifest["stages"]["dpo"].get("metrics"),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
