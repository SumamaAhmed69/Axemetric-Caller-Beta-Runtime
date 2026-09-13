#!/usr/bin/env python3
"""Dedicated Dialforge Sales Acceptance v2.2 Colab bootstrap."""
from __future__ import annotations

import base64
import importlib.util
import json
import os
import pathlib
import sys
import time
import urllib.request

ROOT = pathlib.Path("/content") if pathlib.Path("/content").exists() else pathlib.Path.cwd()
OWNER = "SumamaAhmed69"
REPO = "Axemetric-Caller-Beta-Runtime"
BASE_PATH = "benchmarks/dialforge_final_colab_bootstrap_v3.py"
BASE_LOCAL = ROOT / "dialforge_final_colab_bootstrap_v3_imported.py"


def api_json(url: str):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Dialforge-Sales-v2.2-Bootstrap", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def pinned_sha() -> str:
    value = os.environ.get("DIALFORGE_SOURCE_SHA", "").strip()
    if value:
        return value
    return api_json(
        f"https://api.github.com/repos/{OWNER}/{REPO}/commits/main?x={time.time_ns()}"
    )["sha"]


def load_base(sha: str):
    print("Fetching proven Dialforge environment bootstrap...", flush=True)
    item = api_json(
        f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{BASE_PATH}?ref={sha}&x={time.time_ns()}"
    )
    if item.get("encoding") != "base64" or not item.get("content"):
        raise RuntimeError("GitHub did not return the base bootstrap content.")
    payload = base64.b64decode(item["content"])
    text = payload.decode("utf-8")
    compile(text, str(BASE_LOCAL), "exec")
    BASE_LOCAL.write_bytes(payload)
    spec = importlib.util.spec_from_file_location("dialforge_final_base", BASE_LOCAL)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the proven Dialforge base bootstrap.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    sha = pinned_sha()
    print("=== DIALFORGE SALES COLAB ACCEPTANCE v2.2 ===", flush=True)
    print("Pinned source SHA:", sha, flush=True)
    base = load_base(sha)

    base.RUNNER = ROOT / "dialforge_sales_acceptance_v2_2.py"
    base.REPORT_DIR = ROOT / "dialforge-sales-acceptance"
    base.REPORT_HTML = base.REPORT_DIR / "dialforge-sales-acceptance.html"
    base.REPORT_JSON = base.REPORT_DIR / "dialforge-sales-acceptance.json"
    base.REPORT_ZIP = ROOT / "dialforge-sales-acceptance-results.zip"
    base.RUNNER_PATH = "benchmarks/dialforge_sales_acceptance_v2_2.py"

    print("Sales runner:", base.RUNNER_PATH, flush=True)
    print("Report directory:", base.REPORT_DIR, flush=True)
    os.environ["DIALFORGE_SOURCE_SHA"] = sha

    code = base.main()
    if code not in (None, 0):
        raise RuntimeError(f"Sales bootstrap returned non-zero code: {code}")
    if not base.REPORT_HTML.exists() or not base.REPORT_JSON.exists() or not base.REPORT_ZIP.exists():
        raise RuntimeError("Sales bootstrap completed but one or more report files are missing.")

    print("\n=== DIALFORGE SALES ACCEPTANCE v2.2 COMPLETE ===", flush=True)
    print("HTML:", base.REPORT_HTML, flush=True)
    print("JSON:", base.REPORT_JSON, flush=True)
    print("ZIP :", base.REPORT_ZIP, flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nSales acceptance interrupted by user.", flush=True)
        raise
    except Exception as exc:
        print("\n=== SALES ACCEPTANCE v2.2 STOPPED SAFELY ===", flush=True)
        print(f"{type(exc).__name__}: {exc}", flush=True)
        print("The failure occurred only inside the temporary Colab VM.", flush=True)
        raise SystemExit(1)
