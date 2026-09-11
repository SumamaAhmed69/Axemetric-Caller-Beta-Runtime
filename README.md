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

## Pinned beta channel

Axemetric Caller v20.4 beta uses the release tag `beta-v20.4.0`. The desktop application is pinned to that tag rather than `latest`, so a later release cannot silently change a tester's runtime.

## Runtime contents

- Official Ollama Windows runtime
- Qwen 3 4B model store used locally through Ollama; Ollama can use a supported NVIDIA GPU automatically
- Portable Chatterbox Nano bridge using the smaller CPU PyTorch runtime for broad Windows compatibility
- Faster-Whisper `small.en` model cache

Large runtime ZIPs are split into release parts so individual GitHub assets stay below GitHub's per-asset limit. Axemetric Caller reassembles them automatically, verifies each part, verifies the complete reconstructed archive, and safely extracts it into the app's local runtime directory.

The downloader supports retries and resume attempts for interrupted large assets.

## Security boundary

This public repository is a binary distribution endpoint only. It contains no private application repository, SIP credentials, customer data, production signing private keys, or owner licensing secrets.

The beta channel is deliberately separated from the production signed runtime channel. Production releases retain Ed25519 manifest verification in the private application code.
