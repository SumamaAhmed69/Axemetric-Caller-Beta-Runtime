#!/usr/bin/env python3
"""Immutable one-click bootstrap for Dialforge Final Acceptance v2."""
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

ROOT = pathlib.Path('/content') if pathlib.Path('/content').exists() else pathlib.Path.cwd()
VENV = ROOT / 'dialforge-final-py311-v1'
CACHE = ROOT / 'dialforge-final-cache'
OLLAMA_ARCHIVE = CACHE / 'ollama-linux-amd64.tar.zst'
OLLAMA_URL = 'https://ollama.com/download/ollama-linux-amd64.tar.zst'
OLLAMA_LOG = ROOT / 'dialforge-final-ollama.log'
RUNNER = ROOT / 'dialforge_final_acceptance_v2.py'
REPORT_DIR = ROOT / 'dialforge-final-acceptance'
REPORT_HTML = REPORT_DIR / 'dialforge-final-acceptance.html'
REPORT_JSON = REPORT_DIR / 'dialforge-final-acceptance.json'
REPORT_ZIP = ROOT / 'dialforge-final-acceptance-results.zip'
OWNER = 'SumamaAhmed69'
REPO = 'Axemetric-Caller-Beta-Runtime'
RUNNER_PATH = 'benchmarks/dialforge_final_acceptance_v2.py'
MODELS = ['qwen3:1.7b', 'qwen3:4b-instruct', 'qwen3:4b-instruct-2507-q8_0']


def show(cmd):
    return cmd if isinstance(cmd, str) else ' '.join(str(x) for x in cmd)


def run(cmd, *, check=True, capture=False, env=None):
    print(f'\n> {show(cmd)}', flush=True)
    kwargs = {'text': True, 'env': env}
    if capture:
        kwargs['stdout'] = subprocess.PIPE
        kwargs['stderr'] = subprocess.STDOUT
    result = subprocess.run(cmd, **kwargs)
    if capture and result.stdout:
        print(result.stdout[-20000:], flush=True)
    if check and result.returncode != 0:
        detail = ('\n--- command output ---\n' + result.stdout[-20000:]) if capture and result.stdout else ''
        raise RuntimeError(f'Command failed ({result.returncode}): {show(cmd)}{detail}')
    return result


def apt_setup():
    run(['apt-get', 'update', '-qq'])
    run(['apt-get', 'install', '-y', '-qq', 'pciutils', 'curl', 'ca-certificates', 'git', 'zstd'])


def ollama_works():
    path = shutil.which('ollama')
    if not path:
        return False
    result = subprocess.run([path, '--version'], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode == 0:
        print('Ollama CLI:', result.stdout.strip(), flush=True)
        return True
    return False


def download_with_resume(url: str, destination: pathlib.Path, attempts: int = 12):
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        existing = destination.stat().st_size if destination.exists() else 0
        print(f'Ollama download {attempt}/{attempts}, existing {existing / 1024**2:.1f} MiB', flush=True)
        result = run([
            'curl', '-fL', '--connect-timeout', '20', '--speed-time', '30', '--speed-limit', '1024',
            '--retry', '5', '--retry-all-errors', '--retry-delay', '2', '-C', '-', url, '-o', str(destination)
        ], check=False)
        if result.returncode == 0:
            test = run(['zstd', '-t', str(destination)], check=False, capture=True)
            if test.returncode == 0:
                return
            destination.unlink(missing_ok=True)
        elif result.returncode in {33, 36}:
            destination.unlink(missing_ok=True)
        time.sleep(min(attempt * 2, 12))
    raise RuntimeError('Could not download a valid Ollama archive after repeated resumable attempts.')


def install_ollama():
    if ollama_works():
        return
    download_with_resume(OLLAMA_URL, OLLAMA_ARCHIVE)
    run(['tar', '--zstd', '-xf', str(OLLAMA_ARCHIVE), '-C', '/usr'])
    if not ollama_works():
        raise RuntimeError('Ollama extracted but its CLI is not functional.')


def install_uv():
    existing = shutil.which('uv')
    if existing:
        return pathlib.Path(existing)
    CACHE.mkdir(parents=True, exist_ok=True)
    installer = CACHE / 'uv-install.sh'
    result = run([
        'curl', '-fL', '--connect-timeout', '20', '--retry', '5', '--retry-all-errors', '--retry-delay', '2',
        'https://astral.sh/uv/install.sh', '-o', str(installer)
    ], check=False)
    if result.returncode == 0:
        env = os.environ.copy(); env['UV_INSTALL_DIR'] = '/usr/local/bin'
        run(['sh', str(installer)], check=False, env=env)
    existing = shutil.which('uv') or ('/usr/local/bin/uv' if pathlib.Path('/usr/local/bin/uv').exists() else None)
    if not existing:
        run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--upgrade', 'uv'], capture=True)
        existing = shutil.which('uv')
    if not existing:
        raise RuntimeError('Could not install uv.')
    return pathlib.Path(existing)


def venv_python():
    return VENV / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def valid_venv(python: pathlib.Path):
    if not python.exists():
        return False
    result = subprocess.run([str(python), '-c', 'import sys; raise SystemExit(0 if sys.version_info[:2]==(3,11) else 1)'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def ensure_py311():
    python = venv_python()
    if valid_venv(python):
        print(f'Reusing Python 3.11 environment: {VENV}', flush=True)
        return python
    shutil.rmtree(VENV, ignore_errors=True)
    uv = install_uv()
    run([str(uv), 'python', 'install', '3.11'], capture=True)
    run([str(uv), 'venv', '--python', '3.11', '--seed', '--clear', str(VENV)], capture=True)
    python = venv_python()
    if not valid_venv(python):
        raise RuntimeError('Managed Python 3.11 environment could not be created.')
    return python


def setup_python_stack(python: pathlib.Path):
    env = os.environ.copy(); env['PIP_DISABLE_PIP_VERSION_CHECK'] = '1'; env['PIP_CACHE_DIR'] = str(CACHE / 'pip')
    run([str(python), '-m', 'pip', 'install', '--upgrade', 'pip', 'wheel', 'setuptools'], env=env)
    run([str(python), '-m', 'pip', 'install', 'torch==2.6.0', 'torchaudio==2.6.0', '--index-url', 'https://download.pytorch.org/whl/cu124'], env=env)
    run([str(python), '-m', 'pip', 'install', 'numpy==1.26.4', 'psutil==7.0.0', 'requests==2.32.5', 'faster-whisper==1.2.0', 'chatterbox-tts==0.1.7'], env=env)
    run([str(python), '-m', 'pip', 'install', '--force-reinstall', '--no-deps', 'git+https://github.com/resemble-ai/Perth.git@ff1c8ac55a976971245cdd53c18d6131ca00d993'], env=env)
    run([str(python), '-m', 'pip', 'install', '--force-reinstall', '--no-deps', 'git+https://github.com/resemble-ai/chatterbox.git@5de7a54aa4e5e2baadb0182dde554908b48b85c2'], env=env)


def validate_stack(python: pathlib.Path):
    checks = [
        ('Python', "import sys; print(sys.version); assert sys.version_info[:2]==(3,11)"),
        ('CUDA', "import torch; print(torch.__version__,torch.version.cuda,torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"),
        ('Whisper', "from faster_whisper import WhisperModel; print('faster-whisper OK')"),
        ('Perth', "import perth; print('Perth OK')"),
        ('Chatterbox', "import inspect; from chatterbox.tts_turbo import ChatterboxTurboTTS; s=inspect.signature(ChatterboxTurboTTS.from_pretrained); print(s); assert 'nano' in s.parameters; assert hasattr(ChatterboxTurboTTS,'prepare_conditionals')"),
    ]
    for name, code in checks:
        print(f'\n[{name}]', flush=True); run([str(python), '-c', code], capture=True)


def ollama_ready():
    try:
        with urllib.request.urlopen('http://127.0.0.1:11434/api/tags', timeout=2) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def start_ollama():
    if ollama_ready():
        print('Ollama server already ready.', flush=True); return
    env = os.environ.copy(); env.update({'OLLAMA_HOST':'127.0.0.1:11434','OLLAMA_ORIGINS':'*','OLLAMA_KEEP_ALIVE':'30m','OLLAMA_MAX_LOADED_MODELS':'1','OLLAMA_NUM_PARALLEL':'1','OLLAMA_FLASH_ATTENTION':'1'})
    log = open(OLLAMA_LOG, 'w')
    proc = subprocess.Popen([shutil.which('ollama') or 'ollama', 'serve'], stdout=log, stderr=subprocess.STDOUT, env=env)
    for _ in range(90):
        if ollama_ready():
            print('Ollama server ready.', flush=True); return
        if proc.poll() is not None: break
        time.sleep(1)
    tail = OLLAMA_LOG.read_text(errors='replace')[-20000:] if OLLAMA_LOG.exists() else '(no log)'
    raise RuntimeError('Ollama server failed to start.\n' + tail)


def api_json(url: str):
    req = urllib.request.Request(url, headers={'User-Agent':'Dialforge-Final-Colab-v2','Accept':'application/vnd.github+json'})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode('utf-8'))


def source_sha():
    pinned = os.environ.get('DIALFORGE_SOURCE_SHA','').strip()
    if pinned: return pinned
    return api_json(f'https://api.github.com/repos/{OWNER}/{REPO}/commits/main?x={time.time_ns()}')['sha']


def download_runner():
    sha = source_sha(); print(f'Downloading v2 runner from immutable commit {sha[:12]}...', flush=True)
    item = api_json(f'https://api.github.com/repos/{OWNER}/{REPO}/contents/{RUNNER_PATH}?ref={sha}&x={time.time_ns()}')
    payload = base64.b64decode(item['content']); text = payload.decode('utf-8'); compile(text, str(RUNNER), 'exec'); RUNNER.write_bytes(payload)
    print(f'Runner verified: {len(payload)} bytes', flush=True)


def benchmark_env():
    env = os.environ.copy(); env.update({'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','HF_HOME':str(CACHE/'huggingface'),'HF_HUB_CACHE':str(CACHE/'huggingface'/'hub'),'TOKENIZERS_PARALLELISM':'false','CUDA_VISIBLE_DEVICES':'0'})
    return env


def run_benchmark(python: pathlib.Path):
    shutil.rmtree(REPORT_DIR, ignore_errors=True); REPORT_DIR.mkdir(parents=True, exist_ok=True)
    result = run([str(python), str(RUNNER), '--models', *MODELS, '--comparison-repeats', '6', '--output-dir', str(REPORT_DIR)], check=False, env=benchmark_env())
    if not REPORT_HTML.exists() or not REPORT_JSON.exists():
        raise RuntimeError(f'Final v2 benchmark exited with {result.returncode} without complete report files.')
    REPORT_ZIP.unlink(missing_ok=True); shutil.make_archive(str(REPORT_ZIP.with_suffix('')), 'zip', REPORT_DIR)
    return result.returncode


def main():
    p = argparse.ArgumentParser(); p.add_argument('--validate-only', action='store_true'); p.add_argument('--no-require-cuda', action='store_true'); args = p.parse_args()
    print('=== DIALFORGE FINAL COLAB ACCEPTANCE BOOTSTRAP v2 ===', flush=True)
    print('Pinned source SHA:', os.environ.get('DIALFORGE_SOURCE_SHA','(resolved at runtime)'), flush=True)
    if not args.no_require_cuda:
        if not shutil.which('nvidia-smi'): raise RuntimeError('No NVIDIA GPU. In Colab choose Runtime > Change runtime type > T4 GPU.')
        run(['nvidia-smi'])
    apt_setup(); install_ollama(); python = ensure_py311(); setup_python_stack(python)
    if not args.no_require_cuda: validate_stack(python)
    start_ollama(); download_runner()
    if args.validate_only:
        print('Final v2 bootstrap validation passed.', flush=True); return 0
    code = run_benchmark(python)
    print('\n=== DIALFORGE FINAL ACCEPTANCE v2 COMPLETE ===', flush=True)
    print('HTML:', REPORT_HTML, '\nJSON:', REPORT_JSON, '\nZIP :', REPORT_ZIP, '\nRunner exit code:', code, flush=True)
    return 0

if __name__ == '__main__':
    try: raise SystemExit(main())
    except KeyboardInterrupt: raise
    except Exception as exc:
        print('\n=== FINAL v2 ACCEPTANCE STOPPED SAFELY ===', flush=True); print(f'{type(exc).__name__}: {exc}', flush=True); print('Everything happened inside the temporary Colab VM.', flush=True); raise SystemExit(1)
