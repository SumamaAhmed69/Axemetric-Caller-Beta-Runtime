#!/usr/bin/env bash
set -Eeuo pipefail

STAGE="bootstrap"
trap 'code=$?; echo >&2; echo "❌ Lineborn v6 export failed during stage: $STAGE (exit $code)" >&2; echo "Command: $BASH_COMMAND" >&2; exit $code' ERR

BASE_MODEL="${LINEBORN_BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
BASE_REVISION="${LINEBORN_BASE_REVISION:-cdbee75f17c01a7cc42f958dc650907174af0554}"
ADAPTER_DIR="${1:-training/output/lineborn-sales-dpo/adapter}"
OUTPUT_DIR="${2:-training/output/lineborn-v6-export}"
BALANCED_QUANT="${LINEBORN_BALANCED_QUANT:-Q4_K_M}"
PERFORMANCE_QUANT="${LINEBORN_PERFORMANCE_QUANT:-Q8_0}"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-972d2313bc0bf0a45f634f77d95c9fb03aeab12c}"
PYTHON_BIN="${LINEBORN_PYTHON:-$(command -v python3 || command -v python)}"
KEEP_F16="${LINEBORN_KEEP_F16:-0}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  echo "No Python interpreter found" >&2
  exit 127
fi

echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" --version

ADAPTER_DIR="$("$PYTHON_BIN" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$ADAPTER_DIR")"
OUTPUT_DIR="$("$PYTHON_BIN" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUTPUT_DIR")"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$OUTPUT_DIR/llama.cpp}"
LLAMA_CPP_DIR="$("$PYTHON_BIN" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$LLAMA_CPP_DIR")"

MERGED_DIR="$OUTPUT_DIR/merged-hf"
F16_GGUF="$OUTPUT_DIR/lineborn-v6-f16.gguf"
BALANCED_GGUF="$OUTPUT_DIR/lineborn-v6-balanced-q4_k_m.gguf"
PERFORMANCE_GGUF="$OUTPUT_DIR/lineborn-v6-performance-q8_0.gguf"

mkdir -p "$OUTPUT_DIR"

STAGE="adapter verification"
if [[ ! -f "$ADAPTER_DIR/adapter_config.json" || ! -f "$ADAPTER_DIR/adapter_model.safetensors" ]]; then
  echo "Missing LoRA adapter files in $ADAPTER_DIR" >&2
  exit 2
fi

"$PYTHON_BIN" - "$ADAPTER_DIR/adapter_config.json" "$BASE_MODEL" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1], encoding="utf-8"))
actual = cfg.get("base_model_name_or_path")
expected = sys.argv[2]
if actual != expected:
    raise SystemExit(f"adapter base mismatch: {actual!r} != {expected!r}")
print(f"Adapter base verified: {actual}", flush=True)
PY

STAGE="LoRA merge"
if [[ -f "$MERGED_DIR/config.json" ]] && compgen -G "$MERGED_DIR/*.safetensors" > /dev/null; then
  echo "✅ Merged HF model already exists; resuming from it: $MERGED_DIR"
else
  rm -rf "$MERGED_DIR"
  "$PYTHON_BIN" training/merge_sales_adapter.py \
    --base-model "$BASE_MODEL" \
    --base-revision "$BASE_REVISION" \
    --adapter "$ADAPTER_DIR" \
    --output "$MERGED_DIR" \
    --device auto \
    --dtype auto
fi

STAGE="llama.cpp checkout"
if [[ ! -d "$LLAMA_CPP_DIR/.git" ]]; then
  rm -rf "$LLAMA_CPP_DIR"
  mkdir -p "$LLAMA_CPP_DIR"
  git -C "$LLAMA_CPP_DIR" init -q
  git -C "$LLAMA_CPP_DIR" remote add origin https://github.com/ggml-org/llama.cpp.git
fi
git -C "$LLAMA_CPP_DIR" fetch --depth 1 origin "$LLAMA_CPP_REF"
git -C "$LLAMA_CPP_DIR" checkout --detach -f FETCH_HEAD

STAGE="llama.cpp Python dependencies"
# The top-level llama.cpp requirements also install a CPU-only torch wheel.
# Kaggle already provides a working CUDA torch build, and the Qwen conversion
# only needs the converter's shared Python dependencies. Keep CUDA torch intact.
CONVERTER_DEPS="$LLAMA_CPP_DIR/requirements/requirements-convert_legacy_llama.txt"
if [[ -f "$CONVERTER_DEPS" ]]; then
  "$PYTHON_BIN" -m pip install -q -r "$CONVERTER_DEPS"
fi
"$PYTHON_BIN" - <<'PY'
import torch
print(f"Converter torch preserved: {torch.__version__}; cuda_available={torch.cuda.is_available()}", flush=True)
PY

STAGE="canonical tokenizer restore"
"$PYTHON_BIN" - "$BASE_MODEL" "$BASE_REVISION" "$MERGED_DIR" <<'PY'
from pathlib import Path
import shutil
import sys

from huggingface_hub import hf_hub_download

repo_id = sys.argv[1]
revision = sys.argv[2]
merged_dir = Path(sys.argv[3])
# LoRA training never changes tokenizer weights/vocabulary. Always restore the
# exact upstream tokenizer artifacts before GGUF conversion. This avoids
# tokenizer serialization drift between Transformers versions (notably
# extra_special_tokens list-vs-dict incompatibilities).
required = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
optional = ("special_tokens_map.json", "generation_config.json")

for name in required + optional:
    try:
        src = Path(hf_hub_download(repo_id=repo_id, filename=name, revision=revision))
    except Exception:
        if name in required:
            raise
        continue
    dst = merged_dir / name
    shutil.copy2(src, dst)
    print(f"Restored canonical tokenizer artifact: {name}", flush=True)

# Validate the canonical tokenizer with the same installed Transformers that
# llama.cpp will call as its fallback tokenizer loader.
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(merged_dir, use_fast=True, local_files_only=True)
print(f"Canonical tokenizer validation OK: vocab_size={len(tok)}", flush=True)
PY

STAGE="HF to F16 GGUF conversion"
F16_DONE="$F16_GGUF.complete"
if [[ -s "$F16_GGUF" && -f "$F16_DONE" ]]; then
  echo "✅ Completed F16 GGUF already exists; resuming: $F16_GGUF"
else
  rm -f "$F16_GGUF" "$F16_DONE"
  "$PYTHON_BIN" "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
    "$MERGED_DIR" \
    --outfile "$F16_GGUF" \
    --outtype f16
  test -s "$F16_GGUF"
  touch "$F16_DONE"
fi

STAGE="llama.cpp quantizer build"
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

STAGE="Balanced Q4_K_M quantization"
BALANCED_DONE="$BALANCED_GGUF.complete"
if [[ -s "$BALANCED_GGUF" && -f "$BALANCED_DONE" ]]; then
  echo "✅ Completed Balanced GGUF already exists; resuming: $BALANCED_GGUF"
else
  rm -f "$BALANCED_GGUF" "$BALANCED_DONE"
  "$QUANTIZER" "$F16_GGUF" "$BALANCED_GGUF" "$BALANCED_QUANT"
  test -s "$BALANCED_GGUF"
  touch "$BALANCED_DONE"
fi

STAGE="Performance Q8_0 quantization"
PERFORMANCE_DONE="$PERFORMANCE_GGUF.complete"
if [[ -s "$PERFORMANCE_GGUF" && -f "$PERFORMANCE_DONE" ]]; then
  echo "✅ Completed Performance GGUF already exists; resuming: $PERFORMANCE_GGUF"
else
  rm -f "$PERFORMANCE_GGUF" "$PERFORMANCE_DONE"
  "$QUANTIZER" "$F16_GGUF" "$PERFORMANCE_GGUF" "$PERFORMANCE_QUANT"
  test -s "$PERFORMANCE_GGUF"
  touch "$PERFORMANCE_DONE"
fi

STAGE="Ollama Modelfiles"
cat > "$OUTPUT_DIR/Modelfile.balanced" <<EOF
FROM ./$(basename "$BALANCED_GGUF")
PARAMETER num_ctx 4096
EOF
cat > "$OUTPUT_DIR/Modelfile.performance" <<EOF
FROM ./$(basename "$PERFORMANCE_GGUF")
PARAMETER num_ctx 4096
EOF

STAGE="deployment manifest"
"$PYTHON_BIN" - "$OUTPUT_DIR" "$BASE_MODEL" "$BALANCED_QUANT" "$PERFORMANCE_QUANT" "$LLAMA_CPP_REF" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4*1024*1024), b""):
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
  rm -f "$F16_GGUF" "$F16_DONE"
fi

echo
echo "✅ Lineborn v6 GGUF export complete."
echo "Balanced:    $BALANCED_GGUF"
echo "Performance: $PERFORMANCE_GGUF"
echo "Ollama: cd '$OUTPUT_DIR' && ollama create lineborn-v6-balanced -f Modelfile.balanced"
echo "        cd '$OUTPUT_DIR' && ollama create lineborn-v6-performance -f Modelfile.performance"
