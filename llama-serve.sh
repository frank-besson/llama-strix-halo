#!/bin/bash
llama-server \
  -m ~/models/Qwen3-Coder-30B-A3B-Instruct-IQ4_NL.gguf \
  -c 65536 \
  -ngl 99 \
  --no-mmap \
  -ctk f16 \
  -ctv f16 \
  --jinja \
  -np 1 \
  -a coder-qwen
