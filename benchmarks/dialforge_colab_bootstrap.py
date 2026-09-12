#!/usr/bin/env python3
"""Resilient one-click bootstrap for the Dialforge Google Colab benchmark.

This script keeps setup outside the notebook so the exact same path can be validated in CI.
It installs Ollama using a resumable official archive download, creates an isolated Python
environment for the speech stack, validates each component separately, starts Ollama, and
then runs the public benchmark runner.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path("/content") if pathlib.Path("/content").exists() else pathlib.Path.cwd()
VENV = ROOT / "dialforge-benchmark-venv-v4"
OLLAMA_ARCHIVE = ROOT / "ollama-linux-amd64.tar.zst"
OLLAMA_URL = "https://ollama.com/download/ollama-linux-amd64.tar.zst"
OLLAMA_LOG = ROOT / "ollama.log"
RUNNER = ROOT / "dialforge_colab_benchmark.py"
RUNNER_URL = "https://raw.githubusercontent.com/SumamaAhmed69/Axemetric-Caller-Beta-Runtime/main/benchmarks/dialforge_colab_benchmark.py"
REPORT_DIR = ROOT / "dialforge-benchmark"
REPORT_HTML = REPORT_DIR / "dialforge-benchmark-report.html"
REPORT_JSON = REPORT_DIR / "dialforge-benchmark-report.json"
QWEN_MODELS = ["qwen3:1.7b", "qwen3:4b", "qwen3:8b"]


def show_cmd(cmd):
    if isinstance(cmd, str):
        return cmd
    return " ".join(str(x) for x in cmd)


def run(cmd, *, shell=False, check=True, capture=False, env=None):
    shown = show_cmd(cmd)
    print(f"\n> {shown}", flush=True)
    kwargs = {
        "shell": shell,
        "text": True,
        "env": env,
    }
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.STDOUT
    result = subprocess.run(cmd, **kwargs)
    if capture and result.stdout:
        print(result.stdout[-12000:], flush=True)
    if check and result.returncode != 0:
        detail = ""
        if capture and result.stdout:
            detail = "\n--- command output ---\n" + result.stdout[-12000:]
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {shown}{detail}"
        )
    return result


def apt_setup():
    run(["apt-get", "update", "-qq"])
    # python3-virtualenv is intentional: Colab/Debian can have a broken stdlib
    # ensurepip path even when python3-venv is installed. virtualenv carries its own
    # seed wheels and avoids that failure mode.
    run([
        "apt-get", "install", "-y", "-qq",
        "pciutils", "curl", "ca-certificates", "git", "zstd",
        "python3-venv", "python3-virtualenv",
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


def venv_python():
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def venv_is_healthy(python: pathlib.Path):
    if not python.exists():
        return False
    result = subprocess.run(
        [str(python), "-m", "pip", "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode == 0:
        print(f"Isolated Python environment ready: {result.stdout.strip()}", flush=True)
        return True
    print("Existing isolated environment is incomplete and will be rebuilt.", flush=True)
    if result.stdout:
        print(result.stdout[-4000:], flush=True)
    return False


def ensure_venv():
    python = venv_python()
    if venv_is_healthy(python):
        return python

    if VENV.exists():
        print(f"Removing incomplete environment: {VENV}", flush=True)
        shutil.rmtree(VENV, ignore_errors=True)

    print(f"\nCreating isolated Python environment with virtualenv at {VENV}", flush=True)
    creator = shutil.which("virtualenv")
    if creator:
        result = run(
            [creator, "--python", sys.executable, str(VENV)],
            check=False,
            capture=True,
        )
    else:
        # Fallback if the distro exposes virtualenv only as a Python module.
        result = run(
            [sys.executable, "-m", "virtualenv", "--python", sys.executable, str(VENV)],
            check=False,
            capture=True,
        )

    if result.returncode != 0 or not venv_is_healthy(python):
        # Last-resort bootstrap from pip. This still avoids stdlib ensurepip.
        print("Distro virtualenv creation failed. Installing a pinned virtualenv wheel and retrying...", flush=True)
        shutil.rmtree(VENV, ignore_errors=True)
        install = run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "virtualenv==20.35.4"],
            check=False,
            capture=True,
        )
        if install.returncode != 0:
            raise RuntimeError("Could not install virtualenv after stdlib venv/ensurepip failure.")
        run(
            [sys.executable, "-m", "virtualenv", "--clear", "--python", sys.executable, str(VENV)],
            capture=True,
        )

    if not venv_is_healthy(python):
        raise RuntimeError("Isolated Python environment was created but pip is not functional.")
    return python


def setup_python_stack(python: pathlib.Path):
    env = os.environ.copy()
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    run([str(python), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"], env=env)
    # Chatterbox 0.1.7 pins torch/torchaudio 2.6.0. CUDA 12.4 wheels work with T4.
    run([
        str(python), "-m", "pip", "install",
        "torch==2.6.0", "torchaudio==2.6.0",
        "--index-url", "https://download.pytorch.org/whl/cu124",
    ], env=env)
    run([
        str(python), "-m", "pip", "install",
        "psutil==7.0.0", "requests==2.32.5", "numpy<2",
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
        ("Python", "import sys; print(sys.version); print(sys.executable)"),
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
        cuda_check = (
            "import torch,sys; "
            "print('CUDA available:',torch.cuda.is_available()); "
            "sys.exit(0 if torch.cuda.is_available() else 17)"
        )
        run([str(python), "-c", cuda_check], capture=True)
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
    tail = OLLAMA_LOG.read_text(errors="replace")[-12000:] if OLLAMA_LOG.exists() else "(no log)"
    raise RuntimeError("Ollama server failed to start.\n--- ollama.log ---\n" + tail)


def download_runner():
    print("\nDownloading current Dialforge benchmark runner...", flush=True)
    for attempt in range(1, 6):
        try:
            request = urllib.request.Request(
                RUNNER_URL + f"?v={time.time_ns()}",
                headers={"User-Agent": "Dialforge-Benchmark", "Cache-Control": "no-cache"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            if len(payload) > 1000:
                compile(payload.decode("utf-8"), str(RUNNER), "exec")
                RUNNER.write_bytes(payload)
                print(f"Runner downloaded: {RUNNER} ({RUNNER.stat().st_size} bytes)", flush=True)
                return
        except Exception as exc:
            print(f"Runner download attempt {attempt} failed: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(2 * attempt)
    raise RuntimeError("Could not download the Dialforge benchmark runner from GitHub.")


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
    parser.add_argument("--test-venv-only", action="store_true")
    parser.add_argument("--no-require-cuda", action="store_true")
    args = parser.parse_args()

    print("=== Dialforge Cloud Benchmark bootstrap v4 ===", flush=True)
    if not args.no_require_cuda:
        if not shutil.which("nvidia-smi"):
            raise RuntimeError("No NVIDIA GPU detected. In Colab select Runtime > Change runtime type > T4 GPU.")
        run(["nvidia-smi"])

    apt_setup()

    if args.test_venv_only:
        python = ensure_venv()
        print(f"virtualenv test passed: {python}", flush=True)
        return 0

    install_ollama()
    if args.install_ollama_only:
        print("Ollama installation test passed.", flush=True)
        return 0

    python = ensure_venv()
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
