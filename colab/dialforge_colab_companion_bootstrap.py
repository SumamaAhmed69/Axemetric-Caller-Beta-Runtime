#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = pathlib.Path("/content") if pathlib.Path("/content").exists() else pathlib.Path.cwd()
REPO = pathlib.Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmarks"
sys.path.insert(0, str(BENCH))

import dialforge_final_colab_bootstrap_v3 as legacy

VENV = ROOT / "dialforge-companion-py311-v1"
CACHE = ROOT / "dialforge-final-cache"
WORKER = pathlib.Path(__file__).resolve().parent / "dialforge_colab_companion.py"
CLOUDFLARED = ROOT / "dialforge-cloudflared"
PAIRING_FILE = ROOT / "dialforge-colab-pairing.json"
PORT = 8766

legacy.ROOT = ROOT
legacy.VENV = VENV
legacy.CACHE = CACHE
legacy.OLLAMA_ARCHIVE = CACHE / "ollama-linux-amd64.tar.zst"
legacy.OLLAMA_LOG = ROOT / "dialforge-companion-ollama.log"


def run(cmd, *, check=True, env=None):
    print("\n> " + " ".join(str(x) for x in cmd), flush=True)
    return subprocess.run(cmd, check=check, env=env)


def gpu_info() -> tuple[str, int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError("No NVIDIA GPU detected. In Colab choose Runtime > Change runtime type > T4 GPU.")
    first = result.stdout.strip().splitlines()[0]
    name, memory = [part.strip() for part in first.rsplit(",", 1)]
    return name, int(memory)


def choose_model(vram_mb: int) -> str:
    if vram_mb >= 11776:
        return "qwen3:4b-instruct-2507-q8_0"
    if vram_mb >= 7680:
        return "qwen3:4b-instruct"
    return "qwen3:1.7b"


def install_server_deps(python: pathlib.Path) -> None:
    run([
        str(python), "-m", "pip", "install", "--upgrade",
        "fastapi==0.116.1", "uvicorn==0.35.0", "python-multipart==0.0.20", "httpx==0.28.1",
    ])


def install_cloudflared() -> None:
    if CLOUDFLARED.exists():
        CLOUDFLARED.chmod(0o755)
        return
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    run([
        "curl", "-fL", "--connect-timeout", "20", "--retry", "5", "--retry-all-errors",
        url, "-o", str(CLOUDFLARED),
    ])
    CLOUDFLARED.chmod(0o755)


def wait_local_health(token: str, timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{PORT}/health"
    last = None
    while time.monotonic() < deadline:
        try:
            request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
            with urllib.request.urlopen(request, timeout=3) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            last = exc
        time.sleep(1)
    raise RuntimeError(f"Colab Companion service did not become healthy: {last}")


def drain(prefix: str, pipe) -> None:
    try:
        for line in iter(pipe.readline, ""):
            if line:
                print(f"[{prefix}] {line}", end="", flush=True)
    except Exception:
        pass


def start_tunnel() -> tuple[subprocess.Popen, str]:
    process = subprocess.Popen(
        [str(CLOUDFLARED), "tunnel", "--url", f"http://127.0.0.1:{PORT}", "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    pattern = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    deadline = time.monotonic() + 90
    collected: list[str] = []
    url = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        line = process.stdout.readline() if process.stdout else ""
        if line:
            collected.append(line)
            print("[tunnel] " + line, end="", flush=True)
            match = pattern.search(line)
            if match:
                url = match.group(0)
                break
        else:
            time.sleep(0.1)
    if not url:
        process.terminate()
        raise RuntimeError("Cloudflare quick tunnel did not produce a public HTTPS URL.\n" + "".join(collected[-40:]))
    if process.stdout:
        threading.Thread(target=drain, args=("tunnel", process.stdout), daemon=True).start()
    return process, url


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="auto")
    args = parser.parse_args()

    print("=== DIALFORGE COLAB COMPANION ===", flush=True)
    gpu_name, vram_mb = gpu_info()
    model = choose_model(vram_mb) if args.model == "auto" else args.model.strip()
    print(f"GPU: {gpu_name} ({vram_mb} MiB)", flush=True)
    print(f"Dialforge model tier: {model}", flush=True)

    legacy.apt_setup()
    legacy.install_ollama()
    python = legacy.ensure_py311()
    legacy.setup_python_stack(python)
    install_server_deps(python)
    legacy.validate_stack(python)
    legacy.start_ollama()
    install_cloudflared()

    run(["ollama", "pull", model])

    token = secrets.token_urlsafe(32)
    env = os.environ.copy()
    env.update({
        "DIALFORGE_COMPANION_TOKEN": token,
        "DIALFORGE_COMPANION_MODEL": model,
        "DIALFORGE_COMPANION_ROOT": str(ROOT / "dialforge-colab-companion"),
        "OLLAMA_BASE": "http://127.0.0.1:11434",
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HOME": str(CACHE / "huggingface"),
        "HF_HUB_CACHE": str(CACHE / "huggingface" / "hub"),
        "TOKENIZERS_PARALLELISM": "false",
    })

    worker = subprocess.Popen(
        [str(python), "-m", "uvicorn", "dialforge_colab_companion:app", "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=str(WORKER.parent),
        env=env,
    )
    tunnel = None
    try:
        wait_local_health(token)
        tunnel, public_url = start_tunnel()
        pairing = {
            "service": "Dialforge Colab Companion",
            "url": public_url,
            "token": token,
            "model": model,
            "gpu": gpu_name,
            "vram_mb": vram_mb,
            "started_at": time.time(),
        }
        PAIRING_FILE.write_text(json.dumps(pairing, indent=2), encoding="utf-8")

        print("\n" + "=" * 78, flush=True)
        print("DIALFORGE COLAB COMPANION READY", flush=True)
        print("=" * 78, flush=True)
        print(f"Companion URL : {public_url}", flush=True)
        print(f"Pairing token : {token}", flush=True)
        print(f"Model         : {model}", flush=True)
        print(f"GPU           : {gpu_name} ({vram_mb} MiB)", flush=True)
        print("\nIn Dialforge: Settings > AI Compute > Google Colab Companion.", flush=True)
        print("Paste the URL and token, save, test, then Start AI.", flush=True)
        print("Keep this Colab cell running while you make calls.", flush=True)
        print("=" * 78 + "\n", flush=True)

        while worker.poll() is None and tunnel.poll() is None:
            time.sleep(2)
        if worker.poll() is not None:
            raise RuntimeError(f"Dialforge Companion worker exited with code {worker.returncode}")
        raise RuntimeError(f"Cloudflare tunnel exited with code {tunnel.returncode}")
    except KeyboardInterrupt:
        print("\nStopping Dialforge Colab Companion...", flush=True)
        return 0
    finally:
        if tunnel is not None and tunnel.poll() is None:
            tunnel.terminate()
        if worker.poll() is None:
            worker.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
