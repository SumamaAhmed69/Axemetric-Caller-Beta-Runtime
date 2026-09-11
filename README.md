# Axemetric Caller Beta Runtime

Public binary distribution channel for Axemetric Caller beta testers.

This repository does **not** contain the private Axemetric Caller application source or owner signing keys. GitHub Releases from this repository host the Windows-local AI runtime used by the beta desktop app.

## Tester flow

1. Install Axemetric Caller for Windows.
2. Activate the beta license.
3. The app detects the PC hardware profile.
4. Required AI components are downloaded automatically from this repository's GitHub Release assets.
5. Every downloaded part and reconstructed runtime archive is SHA-256 verified before installation.
6. AI inference then runs locally on the tester's PC.

Testers do not need Git, GitHub Desktop, Python, PowerShell, or a GitHub account.

## Runtime contents

- Ollama Windows runtime
- Qwen 3 4B model store
- Portable Chatterbox Nano bridge with CUDA-capable PyTorch and CPU fallback
- Faster-Whisper `small.en` model cache

Large runtime ZIPs are split into release parts so individual GitHub assets stay below GitHub's per-asset limit. Axemetric Caller reassembles them automatically.

This repository is a beta distribution endpoint only. Production releases use a separately controlled signed distribution channel.
