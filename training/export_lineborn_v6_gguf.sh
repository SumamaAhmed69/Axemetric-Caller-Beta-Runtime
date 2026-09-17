#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${LINEBORN_BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
ADAPTER_DIR="${1:-training/output/lineborn-sales-dpo/adapter}"
OUTPUT_DIR="${2:-training/output/lineborn-v6-export}"
BALANCED_QUANT="${LINEBORN_BALANCED_QUANT:-Q4_K_M}"
PERFORMANCE_QUANT="${LINEBORN_PERFORMANCE_QUANT:-Q8_0}"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-972d2313bc0bf0a45f634f77d95c9fb03aeab12c}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$OUTPUT_DIR/llama.cpp}"
KEEP_F16="${LINEBORN_KEEP_F16:-0}"

ADAPTER_DIR="$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$ADAPTER_DIR")"
OUTPUT_DIR="$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUTPUT_DIR")"
LLAMA_CPP_DIR="$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$LLAMA_CPP_DIR")"
MERGED_DIR="$OUTPUT_DIR/merged-hf"
F16_GGUF="$OUTPUT_DIR/lineborn-v6-f16.gguf"
BALANCED_GGUF="$OUTPUT_DIR/lineborn-v6-balanced-q4_k_m.gguf"
PERFORMANCE_GGUF="$OUTPUT_DIR/lineborn-v6-performance-q8_0.gguf"

mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$ADAPTER_DIR/adapter_config.json" || ! -f "$ADAPTER_DIR/adapter_model.safetensors" ]]; then
  echo "Missing LoRA adapter files in $ADAPTER_DIR" >&2
  exit 2
fi

python - "$ADAPTER_DIR/adapter_config.json" "$BASE_MODEL" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1], encoding="utf-8"))
actual = cfg.get("base_model_name_or_path")
expected = sys.argv[2]
if actual != expected:
    raise SystemExit(f"adapter base mismatch: {actual!r} != {expected!r}")
print(f"Adapter base verified: {actual}")
PY

python training/merge_sales_adapter.py \
  --base-model "$BASE_MODEL" \
  --adapter "$ADAPTER_DIR" \
  --output "$MERGED_DIR"

if [[ ! -d "$LLAMA_CPP_DIR/.git" ]]; then
  rm -rf "$LLAMA_CPP_DIR"
  git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR"
fi
git -C "$LLAMA_CPP_DIR" fetch --depth 1 origin "$LLAMA_CPP_REF"
git -C "$LLAMA_CPP_DIR" checkout --detach FETCH_HEAD

if [[ -f "$LLAMA_CPP_DIR/requirements.txt" ]]; then
  python -m pip install -q -r "$LLAMA_CPP_DIR/requirements.txt"
fi

python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
  "$MERGED_DIR" \
  --outfile "$F16_GGUF" \
  --outtype f16

cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF
cmake --build "$LLAMA_CPP_DIR/build" --config Release -j2 --target llama-quantize

QUANTIZER=""
for candidate in \
  "$LLAMA_CPP_DIR/build/bin/llama-quantize" \
  "$LLAMA_CPP_DIR/build/bin/Release/llama-quantize" \
  "$LLAMA_CPP_DIR/build/bin/llama-quantize.exe" \
  "$LLAMA_CPP_DIR/build/bin/Release/llama-quantize.exe"; do
  if [[ -x "$candidate" || -f "$candidate" ]]; then
    QUANTIZER="$candidate"
    break
  fi
done
if [[ -z "$QUANTIZER" ]]; then
  echo "Could not locate llama-quantize after build" >&2
  exit 3
fi

"$QUANTIZER" "$F16_GGUF" "$BALANCED_GGUF" "$BALANCED_QUANT"
"$QUANTIZER" "$F16_GGUF" "$PERFORMANCE_GGUF" "$PERFORMANCE_QUANT"

cat > "$OUTPUT_DIR/Modelfile.balanced" <<EOF
FROM ./$(basename "$BALANCED_GGUF")
PARAMETER num_ctx 4096
EOF
cat > "$OUTPUT_DIR/Modelfile.performance" <<EOF
FROM ./$(basename "$PERFORMANCE_GGUF")
PARAMETER num_ctx 4096
EOF

python - "$OUTPUT_DIR" "$BASE_MODEL" "$BALANCED_QUANT" "$PERFORMANCE_QUANT" "$LLAMA_CPP_REF" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

files = {}
for name in ["lineborn-v6-balanced-q4_k_m.gguf", "lineborn-v6-performance-q8_0.gguf"]:
    p = root / name
    files[name] = {"bytes": p.stat().st_size, "sha256": sha(p)}
manifest = {
    "schema": 1,
    "model_family": "Lineborn-v6",
    "base_model": sys.argv[2],
    "llama_cpp_commit": sys.argv[5],
    "profiles": {
        "balanced": {"quantization": sys.argv[3], "ollama_model": "lineborn-v6-balanced", "file": "lineborn-v6-balanced-q4_k_m.gguf"},
        "performance": {"quantization": sys.argv[4], "ollama_model": "lineborn-v6-performance", "file": "lineborn-v6-performance-q8_0.gguf"},
    },
    "files": files,
}
(root / "lineborn-v6-deployment-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
PY

if [[ "$KEEP_F16" != "1" ]]; then
  rm -f "$F16_GGUF"
fi

echo
echo "Lineborn v6 GGUF export complete."
echo "Balanced:    $BALANCED_GGUF"
echo "Performance: $PERFORMANCE_GGUF"
echo "Ollama: cd '$OUTPUT_DIR' && ollama create lineborn-v6-balanced -f Modelfile.balanced"
echo "        cd '$OUTPUT_DIR' && ollama create lineborn-v6-performance -f Modelfile.performance"
