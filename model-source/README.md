# Lineborn v6 model source

Place the validated Balanced GGUF in this folder with the exact filename:

    lineborn-v6-balanced-q4_k_m.gguf

Expected SHA-256:

    467514dd216d25b7e291345f11dba9f57d2515473434f13715d159e2db6fa398

Then, from the repository root on Windows, run:

    powershell -ExecutionPolicy Bypass -File .\runtime\publish_lineborn_v6_balanced.ps1

The publisher verifies the GGUF, imports it into an isolated Ollama model store as lineborn-v6-balanced, creates split release assets, patches the beta.15 runtime manifest, uploads the assets to GitHub Releases, and verifies the published checksums.

The GGUF itself is intentionally ignored by Git and is never committed to the repository.
