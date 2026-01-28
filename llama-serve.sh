#!/bin/bash
HIP_VISIBLE_DEVICES=0 llama-server \
  -m ~/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
  -c 65536 \
  -fa on \
  -ngl 99 \
  --no-mmap \
  -ctk q8_0 \
  -ctv q8_0 \
  --jinja \
  -np 1 \
  -a coder-qwen
