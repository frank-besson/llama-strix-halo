# Local LLM Coding Agent on AMD Strix Halo

Running local coding agents (Claude Code, OpenCode) with llama.cpp on an AMD Ryzen AI Max (Strix Halo) with 96GB unified VRAM.

## Hardware

- **APU**: AMD Ryzen AI Max (Strix Halo)
- **GPU**: Radeon 8060S (gfx1151), 40 CUs @ 2.9 GHz
- **VRAM**: 96GB unified (LPDDR5X)
- **Memory Bandwidth**: ~212 GB/s
- **OS**: Arch Linux (kernel 6.18.3)

## Architecture Choices

### Why MoE Models?

On Strix Halo, memory bandwidth (~212 GB/s) is the bottleneck for token generation. MoE models with small active parameters generate tokens much faster than dense models:

| Model Type | Active Params | Token Gen Speed |
|------------|--------------|-----------------|
| MoE 30B (3B active) IQ4_NL | 3B | **~88 tok/s** |
| MoE 30B (3B active) Q8_0 | 3B | ~49 tok/s |
| Dense 32B | 32B | ~6.4 tok/s |
| Dense 70B | 70B | ~3 tok/s |

### Why Vulkan over ROCm?

On Strix Halo (gfx1151), the **Vulkan backend is 17-20% faster** than ROCm/HIP for token generation:

| Backend | Gen (100 tok) | Gen (500 tok) | Gen (8000 tok) |
|---------|---------------|---------------|----------------|
| **Vulkan (RADV) + FA + bf16** | **87.6 tok/s** | **86.3 tok/s** | **73.0 tok/s** |
| ROCm/HIP | 73.7 tok/s | 72.1 tok/s | — |
| Ollama (ROCm) | 44.7 tok/s | 43.3 tok/s | — |

Vulkan now supports flash attention via KHR_coopmat1 (our build includes the latest refactor). Combined with `RADV_PERFTEST=bfloat16`, this gives the best long-session performance.

**When to use ROCm instead**: If you need extremely fast prompt processing for very long inputs (>10K tokens) and your generation lengths are short. The ROCm build gets ~1000+ tok/s prompt eval at 151 tokens vs Vulkan's ~683 tok/s.

### Why IQ4_NL Quantization?

Strix Halo is strictly bandwidth-bound. Smaller model = less data per token = faster generation. IQ4_NL uses importance-based quantization for better quality than Q4_0 with simpler dequantization than K-quants (which are expensive on gfx1151).

| Quant | Size | pp512 (tok/s) | tg128 (tok/s) | Notes |
|-------|------|---------------|---------------|-------|
| Q4_0 | 16 GB | 1277 | 77 | Fastest, lowest quality |
| **IQ4_NL** | **16 GB** | **1188** | **75** | **Best quality at Q4 speed** |
| Q4_1 | 18 GB | 1228 | 72 | Simple dequant |
| Q4_K_M | 17 GB | 1090 | 70 | K-quant overhead hurts gfx1151 |
| Q5_K_M | 20 GB | 1134 | 64 | Diminishing returns |
| Q6_K | 23 GB | 848 | 60 | K-quant dequant expensive |
| Q8_0 | 30 GB | 1259 | 49 | Highest quality, bandwidth-limited |

**Key insight**: K-quant dequantization is expensive on gfx1151. IQ4_NL and Q4_0/Q4_1 use simpler dequant that runs faster on this architecture.

*(ROCm raw benchmarks via llama-bench. Vulkan adds +17% on top of these numbers.)*

## Models

All models use Unsloth GGUFs — Ollama's GGUFs have broken Jinja templates for tool calling in llama.cpp.

### Primary: Qwen3-Coder-30B-A3B-Instruct (IQ4_NL)
- 30.5B total / 3.3B active (MoE, 128 experts, 8 active)
- 256K native context, tool calling support
- **88 tok/s generation** on Strix Halo Vulkan
- Best choice for coding-specific tasks

```bash
huggingface-cli download unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF \
  Qwen3-Coder-30B-A3B-Instruct-IQ4_NL.gguf --local-dir ~/models
```

### Alternative: Qwen3-30B-A3B-Instruct-2507 (IQ4_NL)
- Same architecture (30.5B total / 3.3B active), same speed (~87 tok/s)
- July 2025 general-purpose refresh — broader capabilities, improved instruction following
- Tool calling verified working

```bash
huggingface-cli download unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF \
  Qwen3-30B-A3B-Instruct-2507-IQ4_NL.gguf --local-dir ~/models
```

### Alternative: gpt-oss-20b (MXFP4)
- 21B total / 3.6B active (OpenAI, Apache 2.0)
- Only 12GB — smallest model with tool calling support
- **71 tok/s generation** (MXFP4 dequant is heavier than IQ4_NL)
- Tool calling verified working on llama.cpp v7865+

```bash
huggingface-cli download ggml-org/gpt-oss-20b-GGUF \
  gpt-oss-20b-mxfp4.gguf --local-dir ~/models
```

### Alternative: Qwen3-Next-80B-A3B-Instruct (IQ4_NL)
- 80B total / 3.9B active (512 experts, 10 active)
- Significantly smarter (matches Qwen3-235B on some benchmarks)
- ~45GB IQ4_NL — fits in 96GB VRAM
- **43 tok/s generation** — slower due to larger total weight footprint (43GB vs 16GB)
- Tool calling verified working
- Vulkan more stable than ROCm for this model

```bash
huggingface-cli download unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF \
  Qwen3-Next-80B-A3B-Instruct-IQ4_NL.gguf --local-dir ~/models
```

### Model Comparison

| Model | Size | Active | Gen tok/s | Tool Calling | Notes |
|-------|------|--------|-----------|--------------|-------|
| **Qwen3-Coder-30B-A3B** | 16 GB | 3.3B | **88** | Yes | Coding-focused |
| Qwen3-30B-A3B-2507 | 16 GB | 3.3B | 87 | Yes | General-purpose |
| gpt-oss-20b | 12 GB | 3.6B | 71 | Yes | Smallest, Apache 2.0 |
| Qwen3-Next-80B-A3B | 43 GB | 3.9B | 43 | Yes | Smartest, slower |

### Not Viable

- **GLM-4.7-Flash**: `glm4moelite` architecture not supported in upstream llama.cpp
- **Mixtral 8x22B**: 39B active params = ~5-8 tok/s (too slow)
- **DeepSeek-V3**: 671B total, doesn't fit in 96GB
- **MiniMax-M2.1**: 230B total, Q4 = 129GB (doesn't fit)
- **Phi-3.5-MoE**: No tool calling support

## Setup

### 1. Build llama.cpp

```bash
# Vulkan build (recommended - faster token generation)
~/llama/build-llama.sh vulkan

# ROCm build (alternative - faster prompt processing, flash attention)
~/llama/build-llama.sh rocm
```

Prerequisites:
```bash
# Vulkan
sudo pacman -S cmake vulkan-headers glslang

# ROCm
sudo pacman -S cmake rocm-hip-sdk
```

After build:
```bash
sudo cp /tmp/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
sudo cp /tmp/llama.cpp/build/bin/llama-cli /usr/local/bin/llama-cli
```

### 2. Download Model

```bash
mkdir -p ~/models
huggingface-cli download unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF \
  Qwen3-Coder-30B-A3B-Instruct-IQ4_NL.gguf --local-dir ~/models
```

### 3. Run the Server

```bash
~/llama/llama-serve.sh              # Default: Qwen3-Coder-30B-A3B
~/llama/llama-serve.sh 2507         # Qwen3-30B-A3B-2507
~/llama/llama-serve.sh gpt-oss      # gpt-oss-20b
~/llama/llama-serve.sh next-80b     # Qwen3-Next-80B-A3B (smartest, slower)
```

### 4. Connect Coding Agents

**OpenCode** (`~/.config/opencode/opencode.json`):
```json
{
  "provider": {
    "llama-cpp": {
      "models": {
        "coder-qwen": {
          "name": "Qwen3 Coder 30B IQ4_NL (llama.cpp)"
        }
      },
      "name": "llama.cpp (local)",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://localhost:8080/v1"
      }
    }
  }
}
```

**Claude Code** (via Ollama):
```bash
ollama launch claude --model coder-qwen
```

## Server Flags Explained

```bash
RADV_PERFTEST=bfloat16 llama-server \
  -m ~/models/Qwen3-Coder-30B-A3B-Instruct-IQ4_NL.gguf \
  -c 65536 \        # Context window (64K tokens)
  -fa on \          # Flash attention (Vulkan coopmat1, scales at long context)
  -ngl 99 \         # Offload all layers to GPU
  --no-mmap \       # Required for Strix Halo (hangs without it)
  -ctk f16 \        # KV cache K type (f16 avoids dequant overhead)
  -ctv f16 \        # KV cache V type
  --jinja \         # Enable Jinja templates (required for tool calling)
  -np 1 \           # Single parallel slot (max bandwidth per session)
  -a coder-qwen     # Model alias for API responses
```

- **`RADV_PERFTEST=bfloat16`**: Enables bfloat16 cooperative matrix on GFX11-11.5. Small but consistent pp improvement.
- **`-fa on`**: Vulkan flash attention via KHR_coopmat1. Minor gains at short context, increasingly important as coding sessions grow to 10K-30K+ tokens.

## Strix Halo Gotchas

1. **`--no-mmap` is mandatory** — Without it, model loading hangs indefinitely on Strix Halo.

2. **Vulkan > ROCm for token generation on gfx1151** — Vulkan (RADV) is 17-20% faster for generation. ROCm's gfx1151 HIP kernels are immature (2-6x slower than gfx1100). Vulkan sidesteps this entirely.

3. **K-quant dequantization is expensive on gfx1151** — Q4_K_M is slower than Q4_1 despite being smaller. IQ4_NL and Q4_0/Q4_1 use simpler dequant paths that run faster. Choose quantization carefully.

4. **Vulkan flash attention now works** — Via KHR_coopmat1 (requires recent llama.cpp build). Enable with `-fa on`. Gives marginal gains at short context but increasingly important for long coding sessions (10K-30K+ tokens).

5. **ROCm version matters** — ROCm 6.4.4 was benchmarked faster than 7.0.1.

6. **Kernel version matters** — 15% performance difference observed between Linux 6.14 and 6.15. Use latest stable.

7. **`glm4moelite` not supported in upstream llama.cpp** — GLM-4.7-Flash only works via Ollama's custom fork.

8. **Ollama GGUF tool templates are broken** — Qwen3-Coder's Ollama GGUF uses Jinja features (`reject("in", ...)`) that llama.cpp's minja parser doesn't support. Use Unsloth GGUFs.

## Optimizations Tested

### What Helped

| Optimization | Impact |
|---|---|
| **Vulkan backend** | **+17-20% generation** (88 vs 73 tok/s) |
| **IQ4_NL quantization** | **+50% generation** vs Q8_0 (75 vs 49 tok/s raw) |
| **KV cache on GPU** (don't use `-nkvo`) | +10% generation |
| **KV cache f16** (no quant overhead) | +1-2% generation |
| **`RADV_PERFTEST=bfloat16`** | +5% prompt processing, consistent tg improvement |
| **Vulkan flash attention** (`-fa on`) | Better scaling at long context (8K+ tokens) |

### What Didn't Help

| Optimization | Result |
|---|---|
| `ROCBLAS_USE_HIPBLASLT=1` | No change (MoE compute not bottleneck) |
| `GPU_MAX_HW_QUEUES=2` | No change |
| `HSA_ENABLE_SDMA=0` | No change |
| `GGML_CUDA_FORCE_MMQ=1` | No change |
| `-t 1` (single thread) | No change |
| `--poll 100 --prio-batch 2` | Slightly worse |
| `-ctv q4_0` (aggressive V quant) | 20% slower (dequant > bandwidth savings) |
| `-ctk q8_0 -ctv q8_0` (vs f16) | 1-2% slower |
| Removing `-DGGML_HIP_ROCWMMA_FATTN=ON` | No change at 64K context |
| Speculative decoding (Qwen3-0.6B draft) | 50% slower (draft overhead > benefit for fast MoE) |
| `--spec-type ngram-simple` | No change on single requests (helps multi-turn) |
| `-dio` (direct I/O) | No change |
| `-lcd` (lookup cache) | No change on single requests |

## Ollama Configuration (Alternative)

If using Ollama instead of llama.cpp, apply these optimizations via systemd override:

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf << 'EOF'
[Service]
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
Environment="OLLAMA_NUM_PARALLEL=1"
EOF
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

## VRAM Budget

| Component | Size |
|-----------|------|
| Model weights (IQ4_NL) | ~16 GB |
| KV cache (64K, f16) | ~6.5 GB |
| Compute graph | ~0.4 GB |
| **Total** | **~23 GB** |
| **Free VRAM** | **~73 GB** |

## Performance Summary

**Starting point**: Ollama Q8_0 ROCm = 44.7 tok/s generation

**Final result**: llama.cpp IQ4_NL Vulkan = **88.1 tok/s generation** (+97%)

| Milestone | Gen (100 tok) | Change |
|-----------|---------------|--------|
| Ollama Q8_0 (baseline) | 44.7 tok/s | — |
| llama.cpp Q8_0 ROCm (tuned) | 49.0 tok/s | +10% |
| llama.cpp Q4_K_M ROCm | 66.6 tok/s | +49% |
| llama.cpp IQ4_NL ROCm | 73.7 tok/s | +65% |
| **llama.cpp IQ4_NL Vulkan** | **88.1 tok/s** | **+97%** |

### Generation Speed vs Output Length (Vulkan)

| Output length | tok/s |
|---------------|-------|
| 100 tokens | 88.1 |
| 500 tokens | 86.2 |
| 2000 tokens | 82.8 |
| 4000 tokens | 79.2 |
| 8000 tokens | 73.3 |

## Benchmarking

Run the included benchmark suite:
```bash
# Start the server
~/llama/llama-serve.sh &>/tmp/llama-server.log &

# Run benchmarks
~/llama/test-llama.sh
```

## Resources

- [Strix Halo LLM Performance Tests](https://community.frame.work/t/amd-strix-halo-ryzen-ai-max-395-gpu-llm-performance-tests/72521)
- [Strix Halo Toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes)
- [Strix Halo Testing Repo](https://github.com/lhl/strix-halo-testing)
- [llama.cpp ROCm HIP Discussion](https://github.com/ggml-org/llama.cpp/discussions/15021)
- [Unsloth Qwen3-Coder GGUF (fixed tool calling)](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF)
- [OpenCode + llama.cpp Jinja fix](https://github.com/sst/opencode/issues/1890)
- [Ollama Claude Code Integration](https://docs.ollama.com/integrations/claude-code)
