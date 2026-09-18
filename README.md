# Axemetric Caller Beta Runtime

Public binary distribution channel for Lineborn beta testers.

This repository does not contain the private Lineborn application source or owner signing keys. GitHub Releases from this repository host the local AI runtime used by the beta desktop app.

## Tester flow

1. Install Lineborn.
2. Activate the beta license.
3. Setup detects the PC hardware and runtime requirements.
4. Required AI components are downloaded automatically from this repository's GitHub Release assets.
5. Every downloaded part and reconstructed runtime archive is SHA-256 verified before installation.
6. AI inference then runs locally on the tester's PC.

Testers do not need Git, GitHub Desktop, Python, PowerShell, or a GitHub account.

## Pinned beta channel

Lineborn 20.4 beta.15 uses the release tag beta-v20.4.0-beta.15.

## Trained Lineborn v6 model

The beta model is the trained Lineborn v6 Balanced build:

- Ollama tag: lineborn-v6-balanced
- Quantization: Q4_K_M
- Source GGUF: lineborn-v6-balanced-q4_k_m.gguf
- Source GGUF SHA-256: 467514dd216d25b7e291345f11dba9f57d2515473434f13715d159e2db6fa398
- Release benchmark: 96.9 sales score, 100% product tool accuracy, 0 critical failures

The raw GGUF is not committed to Git. The owner places it in model-source/lineborn-v6-balanced-q4_k_m.gguf and runs runtime/publish_lineborn_v6_balanced.ps1. The publisher verifies the exact model hash, imports it into an isolated Ollama store, creates split GitHub Release assets, updates the beta runtime manifest, uploads the assets and verifies the published digests.

## Runtime contents

- Official Ollama runtime
- Trained Lineborn v6 Balanced model store used locally through Ollama
- Portable Chatterbox Nano bridge
- Faster-Whisper small.en model cache
- Direct SIP runtime

Large runtime ZIPs are split into release parts so individual GitHub assets stay below GitHub's per-asset limit. Lineborn reassembles them automatically, verifies each part, verifies the complete reconstructed archive, and safely extracts it into the app's local runtime directory.

## Security boundary

This public repository is a binary distribution endpoint only. It contains no private application repository, SIP credentials, customer data, production signing private keys, or owner licensing secrets.
