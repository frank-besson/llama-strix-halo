#!/bin/bash
# Launch llama-server with optimized settings for AMD Strix Halo (gfx1151)
# Usage: ./llama-serve.sh [model]
#   model: glm (default), coder, qwen3-2507, gpt-oss, next-80b

MODEL="${1:-glm}"
MODELS_DIR=~/models

case "$MODEL" in
  coder|qwen3-coder)
    MODEL_PATH="${MODELS_DIR}/Qwen3-Coder-30B-A3B-Instruct-IQ4_NL.gguf"
    ALIAS="qwen3-coder-30b"
    CTX=163840
    ;;
  2507|qwen3-2507)
    MODEL_PATH="${MODELS_DIR}/Qwen3-30B-A3B-Instruct-2507-IQ4_NL.gguf"
    ALIAS="qwen3-2507"
    CTX=163840
    ;;
  gpt-oss|gpt)
    MODEL_PATH="${MODELS_DIR}/gpt-oss-20b-mxfp4.gguf"
    ALIAS="gpt-oss-20b"
    CTX=163840
    ;;
  next-80b|next|80b)
    MODEL_PATH="${MODELS_DIR}/Qwen3-Next-80B-A3B-Instruct-IQ4_NL.gguf"
    ALIAS="qwen3-next-80b"
    CTX=163840
    ;;
  glm|glm-flash|glm-4.7)
    MODEL_PATH="${MODELS_DIR}/zai-org_GLM-4.7-Flash-IQ4_NL.gguf"
    ALIAS="glm-4.7-flash"
    CTX=163840
    ;;
  *)
    # Treat as direct path to a GGUF file
    MODEL_PATH="$MODEL"
    ALIAS="custom"
    CTX=163840
    ;;
esac

if [ ! -f "$MODEL_PATH" ]; then
  echo "Error: Model not found: $MODEL_PATH"
  echo ""
  echo "Available models:"
  echo "  glm        - GLM-4.7-Flash IQ4_NL (default)"
  echo "  coder      - Qwen3-Coder-30B-A3B IQ4_NL"
  echo "  2507       - Qwen3-30B-A3B-2507 IQ4_NL"
  echo "  gpt-oss    - gpt-oss-20b MXFP4"
  echo "  next-80b   - Qwen3-Next-80B-A3B IQ4_NL"
  echo "  <path>     - Direct path to any GGUF file"
  exit 1
fi

echo "Starting llama-server: ${ALIAS} ($(basename "$MODEL_PATH"))"
echo "Context: ${CTX} tokens"

LD_LIBRARY_PATH=/usr/local/lib RADV_PERFTEST=bfloat16 llama-server \
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
