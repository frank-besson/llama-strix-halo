#!/bin/bash
# Launch llama-server with optimized settings for AMD Strix Halo (gfx1151)
# Usage: ./llama-serve.sh [model]
#   model: coder (default), qwen3-2507, gpt-oss, next-80b

MODEL="${1:-coder}"
MODELS_DIR=~/models

case "$MODEL" in
  coder|qwen3-coder)
    MODEL_PATH="${MODELS_DIR}/Qwen3-Coder-30B-A3B-Instruct-IQ4_NL.gguf"
    ALIAS="coder-qwen"
    CTX=65536
    ;;
  2507|qwen3-2507)
    MODEL_PATH="${MODELS_DIR}/Qwen3-30B-A3B-Instruct-2507-IQ4_NL.gguf"
    ALIAS="qwen3-2507"
    CTX=65536
    ;;
  gpt-oss|gpt)
    MODEL_PATH="${MODELS_DIR}/gpt-oss-20b-mxfp4.gguf"
    ALIAS="gpt-oss-20b"
    CTX=65536
    ;;
  next-80b|next|80b)
    MODEL_PATH="${MODELS_DIR}/Qwen3-Next-80B-A3B-Instruct-IQ4_NL.gguf"
    ALIAS="qwen3-next-80b"
    CTX=65536
    ;;
  *)
    # Treat as direct path to a GGUF file
    MODEL_PATH="$MODEL"
    ALIAS="custom"
    CTX=65536
    ;;
esac

if [ ! -f "$MODEL_PATH" ]; then
  echo "Error: Model not found: $MODEL_PATH"
  echo ""
  echo "Available models:"
  echo "  coder      - Qwen3-Coder-30B-A3B IQ4_NL (default)"
  echo "  2507       - Qwen3-30B-A3B-2507 IQ4_NL"
  echo "  gpt-oss    - gpt-oss-20b MXFP4"
  echo "  next-80b   - Qwen3-Next-80B-A3B IQ4_NL"
  echo "  <path>     - Direct path to any GGUF file"
  exit 1
fi

echo "Starting llama-server: ${ALIAS} ($(basename "$MODEL_PATH"))"
echo "Context: ${CTX} tokens"

RADV_PERFTEST=bfloat16 llama-server \
  -m "$MODEL_PATH" \
  -c "$CTX" \
  -fa on \
  -ngl 99 \
  --no-mmap \
  -ctk f16 \
  -ctv f16 \
  --jinja \
  -np 1 \
  -a "$ALIAS"
