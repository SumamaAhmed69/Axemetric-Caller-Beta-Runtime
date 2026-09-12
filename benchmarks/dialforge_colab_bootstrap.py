#!/usr/bin/env python3
"""Resilient one-click bootstrap for the Dialforge Google Colab benchmark.

This bootstrap deliberately does not inherit Colab's system Python. Current Colab images can
ship Python 3.13, while the Dialforge speech stack is benchmarked on Python 3.11. We provision
a managed Python 3.11 interpreter with uv, create an isolated environment from that exact
interpreter, validate the stack component by component, then run the benchmark.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path("/content") if pathlib.Path("/content").exists() else pathlib.Path.cwd()
VENV = ROOT / "dialforge-benchmark-py311-v5"
OLLAMA_ARCHIVE = ROOT / "ollama-linux-amd64.tar.zst"
OLLAMA_URL = "https://ollama.com/download/ollama-linux-amd64.tar.zst"
OLLAMA_LOG = ROOT / "ollama.log"
RUNNER = ROOT / "dialforge_colab_benchmark.py"
REPORT_DIR = ROOT / "dialforge-benchmark"
REPORT_HTML = REPORT_DIR / "dialforge-benchmark-report.html"
REPORT_JSON = REPORT_DIR / "dialforge-benchmark-report.json"
QWEN_MODELS = ["qwen3:1.7b", "qwen3:4b", "qwen3:8b"]
OWNER = "SumamaAhmed69"
REPO = "Axemetric-Caller-Beta-Runtime"
RUNNER_PATH = "benchmarks/dialforge_colab_benchmark.py"


def show_cmd(cmd):
    if isinstance(cmd, str):
        return cmd
    return " ".join(str(x) for x in cmd)


def run(cmd, *, shell=False, check=True, capture=False, env=None):
    shown = show_cmd(cmd)
    print(f"\n> {shown}", flush=True)
    kwargs = {"shell": shell, "text": True, "env": env}
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.STDOUT
    result = subprocess.run(cmd, **kwargs)
    if capture and result.stdout:
        print(result.stdout[-16000:], flush=True)
    if check and result.returncode != 0:
        detail = ""
        if capture and result.stdout:
            detail = "\n--- command output ---\n" + result.stdout[-16000:]
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {shown}{detail}")
    return result


def apt_setup():
    run(["apt-get", "update", "-qq"])
    run([
        "apt-get", "install", "-y", "-qq",
        "pciutils", "curl", "ca-certificates", "git", "zstd",
    ])


def ollama_works():
    path = shutil.which("ollama")
    if not path:
        return False
    result = subprocess.run(
        [path, "--version"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    if result.returncode == 0:
        print(f"Ollama CLI ready: {result.stdout.strip()}", flush=True)
        return True
    print("Existing Ollama CLI is incomplete; repairing it.", flush=True)
    if result.stdout:
        print(result.stdout[-4000:], flush=True)
    return False


def download_with_resume(url: str, destination: pathlib.Path, attempts: int = 12):
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        existing = destination.stat().st_size if destination.exists() else 0
        print(
            f"\nOllama archive download attempt {attempt}/{attempts} "
            f"(already have {existing / 1024**2:.1f} MiB)",
            flush=True,
        )
        cmd = [
            "curl", "-fL",
            "--connect-timeout", "20",
            "--speed-time", "30",
            "--speed-limit", "1024",
            "--retry", "5",
            "--retry-all-errors",
            "--retry-delay", "2",
            "-C", "-",
            url,
            "-o", str(destination),
        ]
        result = run(cmd, check=False)
        if result.returncode == 0:
            test = run(["zstd", "-t", str(destination)], check=False, capture=True)
            if test.returncode == 0:
                return
            print("Download completed but archive integrity failed; restarting archive.", flush=True)
            destination.unlink(missing_ok=True)
        elif result.returncode in {33, 36}:
            print("Resume was refused by the server/proxy; restarting from byte 0.", flush=True)
            destination.unlink(missing_ok=True)
        time.sleep(min(3 * attempt, 15))
    raise RuntimeError("Could not download a complete Ollama archive after repeated resumable attempts.")


def install_ollama():
    if ollama_works():
        return
    print("\nInstalling Ollama from its official Linux archive with resume support...", flush=True)
    download_with_resume(OLLAMA_URL, OLLAMA_ARCHIVE)
    run(["tar", "--zstd", "-xf", str(OLLAMA_ARCHIVE), "-C", "/usr"])
    if not ollama_works():
        raise RuntimeError("Ollama archive extracted, but the CLI still does not run.")


def install_uv():
    existing = shutil.which("uv")
    if existing:
        run([existing, "--version"])
        return pathlib.Path(existing)

    print("\nInstalling uv so the benchmark can use managed Python 3.11...", flush=True)
    installer = ROOT / "uv-install.sh"
    result = run([
        "curl", "-fL",
        "--connect-timeout", "20",
        "--retry", "5",
        "--retry-all-errors",
        "--retry-delay", "2",
        "https://astral.sh/uv/install.sh",
        "-o", str(installer),
    ], check=False)
    if result.returncode == 0:
        env = os.environ.copy()
        env["UV_INSTALL_DIR"] = "/usr/local/bin"
        run(["sh", str(installer)], check=False, env=env)

    existing = shutil.which("uv") or ("/usr/local/bin/uv" if pathlib.Path("/usr/local/bin/uv").exists() else None)
    if not existing:
        print("Official uv installer did not expose the CLI; using pip fallback.", flush=True)
        run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--upgrade", "uv"], capture=True)
        existing = shutil.which("uv")
    if not existing:
        raise RuntimeError("Could not install the uv Python runtime manager.")
    run([existing, "--version"])
    return pathlib.Path(existing)


def venv_python():
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def python_is_311(python: pathlib.Path):
    if not python.exists():
        return False
    code = "import sys; print(sys.version); raise SystemExit(0 if sys.version_info[:2] == (3,11) else 31)"
    result = subprocess.run(
        [str(python), "-c", code], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    if result.stdout:
        print("Environment interpreter:", result.stdout.strip(), flush=True)
    return result.returncode == 0


def pip_is_healthy(python: pathlib.Path):
    result = subprocess.run(
        [str(python), "-m", "pip", "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.stdout:
        print(result.stdout.strip(), flush=True)
    return result.returncode == 0


def ensure_py311_env():
    python = venv_python()
    if python_is_311(python) and pip_is_healthy(python):
        print(f"Python 3.11 benchmark environment ready: {VENV}", flush=True)
        return python

    if VENV.exists():
        print(f"Removing incompatible/incomplete environment: {VENV}", flush=True)
        shutil.rmtree(VENV, ignore_errors=True)

    uv = install_uv()
    print("\nProvisioning managed CPython 3.11...", flush=True)
    run([str(uv), "python", "install", "3.11"], capture=True)
    found = run([str(uv), "python", "find", "3.11"], capture=True)
    if found.returncode != 0:
        raise RuntimeError("uv installed but could not locate managed Python 3.11.")

    print(f"\nCreating Python 3.11 environment at {VENV}", flush=True)
    run([str(uv), "venv", "--python", "3.11", "--seed", "--clear", str(VENV)], capture=True)

    python = venv_python()
    if not python_is_311(python):
        raise RuntimeError("Benchmark environment was created with the wrong Python version; expected 3.11.")
    if not pip_is_healthy(python):
        raise RuntimeError("Python 3.11 environment exists but pip is not functional.")
    return python


def setup_python_stack(python: pathlib.Path):
    env = os.environ.copy()
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    run([str(python), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"], env=env)
    run([
        str(python), "-m", "pip", "install",
        "torch==2.6.0", "torchaudio==2.6.0",
        "--index-url", "https://download.pytorch.org/whl/cu124",
    ], env=env)
    # Python 3.11 receives binary wheels for numpy 1.26.x, avoiding Colab's Python 3.13 source-build trap.
    run([
        str(python), "-m", "pip", "install",
        "psutil==7.0.0", "requests==2.32.5", "numpy==1.26.4",
        "faster-whisper==1.2.0", "chatterbox-tts==0.1.7",
    ], env=env)
    run([
        str(python), "-m", "pip", "install", "--force-reinstall", "--no-deps",
        "git+https://github.com/resemble-ai/Perth.git@ff1c8ac55a976971245cdd53c18d6131ca00d993",
    ], env=env)
    run([
        str(python), "-m", "pip", "install", "--force-reinstall", "--no-deps",
        "git+https://github.com/resemble-ai/chatterbox.git@5de7a54aa4e5e2baadb0182dde554908b48b85c2",
    ], env=env)


def validate_python_stack(python: pathlib.Path, require_cuda: bool):
    print("\n=== Validating Python AI stack component by component ===", flush=True)
    checks = [
        (
            "Python 3.11",
            "import sys; print(sys.version); assert sys.version_info[:2] == (3,11); print('Python 3.11 OK')",
        ),
        (
            "NumPy",
            "import numpy as np; print('numpy',np.__version__); assert np.__version__ == '1.26.4'",
        ),
        (
            "PyTorch / CUDA",
            "import torch,torchaudio; "
            "print('torch',torch.__version__,'torchaudio',torchaudio.__version__); "
            "print('cuda_build',torch.version.cuda,'cuda_available',torch.cuda.is_available()); "
            "print('gpu',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')",
        ),
        ("Perth", "import perth; print('perth import OK')"),
        (
            "Chatterbox Nano",
            "import inspect; from chatterbox.tts_turbo import ChatterboxTurboTTS; "
            "s=inspect.signature(ChatterboxTurboTTS.from_pretrained); "
            "print('from_pretrained',s); assert 'nano' in s.parameters; print('Chatterbox Nano import OK')",
        ),
        ("faster-whisper", "from faster_whisper import WhisperModel; print('faster-whisper import OK')"),
    ]
    for name, code in checks:
        print(f"\n[{name}]", flush=True)
        run([str(python), "-c", code], capture=True)
    if require_cuda:
        run([
            str(python), "-c",
            "import torch,sys; print('CUDA available:',torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 17)",
        ], capture=True)
    print("\nAll Python component validation checks passed.", flush=True)


def ollama_ready():
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def start_ollama():
    if ollama_ready():
        print("Ollama server already ready.", flush=True)
        return
    env = os.environ.copy()
    env["OLLAMA_HOST"] = "127.0.0.1:11434"
    env["OLLAMA_ORIGINS"] = "*"
    log_handle = open(OLLAMA_LOG, "w")
    process = subprocess.Popen(
        [shutil.which("ollama") or "ollama", "serve"],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
    )
    for _ in range(90):
        if ollama_ready():
            print("Ollama server ready.", flush=True)
            return
        if process.poll() is not None:
            break
        time.sleep(1)
    tail = OLLAMA_LOG.read_text(errors="replace")[-16000:] if OLLAMA_LOG.exists() else "(no log)"
    raise RuntimeError("Ollama server failed to start.\n--- ollama.log ---\n" + tail)


def api_json(url: str):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Dialforge-Benchmark", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def resolve_source_sha():
    pinned = os.environ.get("DIALFORGE_SOURCE_SHA", "").strip()
    if pinned:
        return pinned
    data = api_json(f"https://api.github.com/repos/{OWNER}/{REPO}/commits/main?dialforge={time.time_ns()}")
    return data["sha"]


def download_runner():
    source_sha = resolve_source_sha()
    print(f"\nDownloading benchmark runner from immutable commit {source_sha[:12]}...", flush=True)
    item = api_json(
        f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{RUNNER_PATH}?ref={source_sha}&dialforge={time.time_ns()}"
    )
    if item.get("encoding") != "base64" or not item.get("content"):
        raise RuntimeError("GitHub Contents API did not return benchmark runner content.")
    payload = base64.b64decode(item["content"])
    text = payload.decode("utf-8")
    compile(text, str(RUNNER), "exec")
    RUNNER.write_bytes(payload)
    print(f"Runner verified: {RUNNER} ({len(payload)} bytes)", flush=True)


def run_benchmark(python: pathlib.Path):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(python), str(RUNNER),
        "--models", *QWEN_MODELS,
        "--pipeline-models", *QWEN_MODELS,
        "--qwen-repeats", "3",
        "--component-repeats", "3",
        "--pipeline-turns", "5",
        "--output-dir", str(REPORT_DIR),
    ]
    result = run(cmd, check=False)
    if not REPORT_JSON.exists():
        raise RuntimeError(f"Benchmark exited with code {result.returncode} and did not create a JSON report.")
    if not REPORT_HTML.exists():
        raise RuntimeError(f"Benchmark exited with code {result.returncode} and did not create an HTML report.")
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--install-ollama-only", action="store_true")
    parser.add_argument("--test-py311-only", action="store_true")
    parser.add_argument("--no-require-cuda", action="store_true")
    args = parser.parse_args()

    print("=== Dialforge Cloud Benchmark bootstrap v5-py311 ===", flush=True)
    print("Host Python:", sys.version.replace("\n", " "), flush=True)
    if not args.no_require_cuda:
        if not shutil.which("nvidia-smi"):
            raise RuntimeError("No NVIDIA GPU detected. In Colab select Runtime > Change runtime type > T4 GPU.")
        run(["nvidia-smi"])

    apt_setup()

    if args.test_py311_only:
        python = ensure_py311_env()
        print(f"Managed Python 3.11 environment test passed: {python}", flush=True)
        return 0

    install_ollama()
    if args.install_ollama_only:
        print("Ollama installation test passed.", flush=True)
        return 0

    python = ensure_py311_env()
    setup_python_stack(python)
    validate_python_stack(python, require_cuda=not args.no_require_cuda)
    start_ollama()
    download_runner()
    code = run_benchmark(python)

    print("\n=== DIALFORGE BENCHMARK FINISHED ===", flush=True)
    print(f"HTML report: {REPORT_HTML}", flush=True)
    print(f"JSON report: {REPORT_JSON}", flush=True)
    print(f"Benchmark process exit code: {code}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.", flush=True)
        raise
    except Exception as exc:
        print("\n=== BENCHMARK STOPPED SAFELY ===", flush=True)
        print(f"{type(exc).__name__}: {exc}", flush=True)
        print("Nothing was installed on your PC. Everything above happened inside the temporary Colab VM.", flush=True)
        raise SystemExit(1)
