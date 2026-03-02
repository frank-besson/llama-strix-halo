# Local LLM Coding Agent on AMD Strix Halo

Running local coding agents (Claude Code, OpenCode) with llama.cpp on an AMD Ryzen AI Max (Strix Halo) with 96GB unified VRAM.

## Hardware

- **APU**: AMD Ryzen AI Max (Strix Halo)
- **GPU**: Radeon 8060S (gfx1151), 40 CUs @ 2.9 GHz
- **VRAM**: 96GB unified (LPDDR5X)
- **Memory Bandwidth**: ~212 GB/s
- **OS**: NixOS 25.11 (kernel 6.19)

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

All models use Unsloth/bartowski GGUFs (except gpt-oss which uses ggml-org) — Ollama's GGUFs have broken Jinja templates for tool calling in llama.cpp.

### Coding: Qwen3-Coder-30B-A3B-Instruct (IQ4_NL)
- 30.5B total / 3.3B active (MoE, 128 experts, 8 active)
- 256K native context, tool calling support
- **87 tok/s generation** on Strix Halo Vulkan
- Best choice for coding-specific tasks

```bash
hf download unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF \
  Qwen3-Coder-30B-A3B-Instruct-IQ4_NL.gguf --local-dir ~/models
```

### General Agent: GLM-4.7-Flash (IQ4_NL)
- 30B total / ~3B active (MoE)
- 200K context (with MLA), native tool calling
- **69 tok/s generation** on Strix Halo Vulkan
- Best tool calling benchmark: τ²-Bench 79.5 (vs Qwen3-30B 49.0)
- Best choice for agentic/research tasks with MCP tools

```bash
hf download bartowski/zai-org_GLM-4.7-Flash-GGUF \
  zai-org_GLM-4.7-Flash-IQ4_NL.gguf --local-dir ~/models
```

### Lightweight: gpt-oss-20b (MXFP4)
- 21B total / 3.6B active (OpenAI, Apache 2.0)
- Only 12GB — smallest model with tool calling support
- **77 tok/s generation** (MXFP4 dequant is heavier than IQ4_NL)
- Tool calling verified working on llama.cpp v7865+

```bash
hf download ggml-org/gpt-oss-20b-GGUF \
  gpt-oss-20b-mxfp4.gguf --local-dir ~/models
```

### Agent: Nemotron-3-Nano-30B-A3B (IQ4_NL)
- 30B total / 3.5B active (Mamba-2 + MoE hybrid: 23 Mamba-2, 23 MoE, 6 attention layers)
- 128 experts + 1 shared, 6 activated per token
- 1M context, explicitly fine-tuned for tool calling
- ~18GB IQ4_NL — lightweight, fast
- **75 tok/s generation** on Strix Halo Vulkan
- Officially supported in llama.cpp (NVIDIA collaboration)

```bash
hf download unsloth/Nemotron-3-Nano-30B-A3B-GGUF \
  Nemotron-3-Nano-30B-A3B-IQ4_NL.gguf --local-dir ~/models
```

### Coding Agent: Qwen3-Coder-Next (IQ4_NL)
- 80B total / 3B active (512 experts, 10 active)
- **Hybrid DeltaNet + MoE architecture** (3:1 linear-to-standard attention ratio)
- 256K context, coding/agent-specialized with tool calling
- ~43GB IQ4_NL — fits in 96GB VRAM
- Requires llama.cpp build 8185+ for Vulkan DeltaNet support

```bash
hf download unsloth/Qwen3-Coder-Next-GGUF \
  Qwen3-Coder-Next-IQ4_NL.gguf --local-dir ~/models
```

### General: Qwen3.5-35B-A3B (UD-Q4_K_XL)
- 35B total / 3B active (256 experts, 8 active)
- **Hybrid DeltaNet + MoE architecture** (same as Coder-Next)
- 262K context (extendable to 1M), general-purpose
- ~20GB UD-Q4_K_XL — no IQ4_NL available, uses Unsloth Dynamic 2.0 quant
- Requires llama.cpp build 8185+ for Vulkan DeltaNet support

```bash
hf download unsloth/Qwen3.5-35B-A3B-GGUF \
  Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf --local-dir ~/models
```

### Model Comparison

All benchmarked on Strix Halo, Vulkan backend, IQ4_NL quant (except where noted), 160K context, `-fa on`.

| Model | Arch | Size | Active | Gen tok/s | Capability | Best For |
|-------|------|------|--------|-----------|------------|----------|
| **Qwen3-Coder-30B-A3B** | MoE | 17 GB | 3.3B | **87** | 7/12 | Coding (fastest) |
| gpt-oss-20b | MoE | 12 GB | 3.6B | 77 | 8/12 | Lightweight |
| Nemotron-3-Nano | Mamba+MoE | 18 GB | 3.5B | 75 | 8/12 | Agent/tool use |
| GLM-4.7-Flash | MoE | 16 GB | ~3B | 69 | 8/12 | Agent/tool use |
| Qwen3-Coder-Next | DeltaNet+MoE | 43 GB | 3B | 44 | 8/12 | Quality coding |
| Qwen3.5-35B-A3B | DeltaNet+MoE | 20 GB | 3B | 41 | 7/12 | General |

### Capability Test Results

Tested with `~/llama/test-capabilities.sh` — 12 tests across 3 categories. Run `bash ~/llama/test-capabilities.sh` to reproduce.

| Model | Tool Calling | Instruction | Reasoning | Total | tok/s |
|-------|-------------|-------------|-----------|-------|-------|
| gpt-oss-20b | 2/5 | 3/4 | 3/3 | **8/12** | 77 |
| nemotron-nano | 2/5 | 3/4 | 3/3 | **8/12** | 75 |
| glm-4.7-flash | 2/5 | 3/4 | 3/3 | **8/12** | 69 |
| qwen3-coder-next | 2/5 | 3/4 | 3/3 | **8/12** | 44 |
| qwen3.5-35b | 2/5 | 2/4 | 3/3 | **7/12** | 41 |
| qwen3-coder-30b | 2/5 | 3/4 | 2/3 | **7/12** | 87 |

**Test categories:**
- **Tool Calling** (5 tests): single tool call, multi-tool call, complex args, tool_choice:none, multi-turn tool use
- **Instruction Following** (4 tests): JSON-only output, bullet point constraint, system persona, refusal
- **Reasoning** (3 tests): math (17×23+45=436), logic (syllogism), code generation

**Notes:**
- Multi-tool parallel calling works via `parallel_tool_calls: true` in the request (confirmed 5/5 on Nemotron and GLM). The automated test is non-deterministic — some models occasionally return sequential calls instead of parallel
- Complex nested args work (confirmed manually on GLM) but the automated test is non-deterministic — models sometimes flatten or omit nested objects
- `tool_choice: "none"` fails universally — **llama.cpp server ignores this parameter** and calls tools anyway
- JSON-only output fails on all models (thinking models emit chain-of-thought before JSON)
- Qwen3-Coder-30B fails the math test — chain-of-thought arrives at the wrong answer
- All models pass: single tool call, multi-turn tool use, system persona, refusal, logic, code generation

### Not Viable

- **GLM-5** (Feb 2026): 744B / 44B active — doesn't fit in 96GB VRAM
- **DeepSeek V4**: Not yet released (expected March 2026), 1T / ~32B active
- **Qwen3.5-397B-A17B**: 17B active = ~12 tok/s (too slow for interactive use)
- **Mistral Large 3**: 675B / 41B active — doesn't fit in 96GB VRAM
- **Mixtral 8x22B**: 39B active params = ~5-8 tok/s (too slow)
- **DeepSeek-V3**: 671B total, doesn't fit in 96GB
- **MiniMax-M2.1**: 230B total, Q4 = 129GB (doesn't fit)
- **Kimi-K2**: 1T total, doesn't fit
- **ERNIE-4.5-21B-A3B**: llama.cpp MoE support incomplete, tool calling template missing

## Setup

### 1. Build llama.cpp

```bash
# Vulkan build (recommended - faster token generation)
~/llama/build-llama.sh vulkan

# ROCm build (alternative - faster prompt processing, flash attention)
~/llama/build-llama.sh rocm
```

The build script handles prerequisites via `nix-shell` and installs binaries + libraries to `/usr/local/bin/` and `/usr/local/lib/` automatically.

### 2. Download Model

```bash
mkdir -p ~/models
huggingface-cli download unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF \
  Qwen3-Coder-30B-A3B-Instruct-IQ4_NL.gguf --local-dir ~/models
```

### 3. Run the Server

```bash
~/llama/llama-serve.sh              # Default: GLM-4.7-Flash (69 tok/s, best agentic)
~/llama/llama-serve.sh coder        # Qwen3-Coder-30B-A3B (87 tok/s)
~/llama/llama-serve.sh coder-next   # Qwen3-Coder-Next (80B/3B, DeltaNet+MoE)
~/llama/llama-serve.sh nemotron     # Nemotron-3-Nano (30B/3.5B, Mamba+MoE, 75 tok/s)
~/llama/llama-serve.sh qwen3.5      # Qwen3.5-35B-A3B (35B/3B, DeltaNet+MoE)
~/llama/llama-serve.sh gpt-oss      # gpt-oss-20b (77 tok/s)
```

### 4. Connect Coding Agents

**Claude Code** (via [claude-code-router](https://github.com/musistudio/claude-code-router)):

```bash
# Install
npm install -g @musistudio/claude-code-router
```

Create `~/.claude-code-router/config.json`:
```json
{
  "LOG": true,
  "HOST": "127.0.0.1",
  "PORT": 3456,
  "Providers": [
    {
      "name": "llama-cpp",
      "api_base_url": "http://localhost:8080/v1/chat/completions",
      "api_key": "local",
      "models": [
        "qwen3-coder-30b",
        "qwen3-coder-next",
        "nemotron-nano",
        "qwen3.5-35b",
        "glm-4.7-flash",
        "gpt-oss-20b"
      ]
    }
  ],
  "Router": {
    "default": "llama-cpp,qwen3-coder-30b"
  }
}
```

Then run:
```bash
# Start the router (runs in background)
ccr start

# Launch Claude Code through the router
ccr code
```

The router translates Anthropic's Messages API to OpenAI Chat Completions (which llama-server speaks), handling streaming, tool use, and thinking blocks. Your normal `claude` command is unaffected.

To switch models, edit the `Router.default` value in the config (e.g., `"llama-cpp,glm-4.7-flash"` for the best agentic model). The model name must match the `-a` alias used by `llama-serve.sh`.

**OpenCode** (`~/.config/opencode/opencode.json`):
```json
{
  "provider": {
    "llama-cpp": {
      "models": {
        "qwen3-coder-30b": {
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

## Server Flags Explained

```bash
RADV_PERFTEST=bfloat16 /usr/local/bin/llama-server \
  -m ~/models/Qwen3-Coder-30B-A3B-Instruct-IQ4_NL.gguf \
  -c 163840 \       # Context window (160K tokens)
  -fa on \          # Flash attention (Vulkan coopmat1, scales at long context)
  -ngl 99 \         # Offload all layers to GPU
  --no-mmap \       # Required for Strix Halo (hangs without it)
  -ctk f16 \        # KV cache K type (f16 avoids dequant overhead)
  -ctv f16 \        # KV cache V type
  --jinja \         # Enable Jinja templates (required for tool calling)
  -np 1 \           # Single parallel slot (max bandwidth per session)
  -a qwen3-coder-30b     # Model alias for API responses
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

7. **Ollama GGUF tool templates are broken** — Qwen3-Coder's Ollama GGUF uses Jinja features (`reject("in", ...)`) that llama.cpp's minja parser doesn't support. Use Unsloth GGUFs.

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

If using Ollama instead of llama.cpp, set these environment variables in your Ollama service configuration:

```
OLLAMA_FLASH_ATTENTION=1
OLLAMA_KV_CACHE_TYPE=q8_0
OLLAMA_NUM_PARALLEL=1
```

On NixOS, configure via `services.ollama.environmentVariables` in `configuration.nix`. On other distros, use a systemd override.

## VRAM Budget

Example for a typical ~16GB MoE model at 160K context:

| Component | Size |
|-----------|------|
| Model weights (IQ4_NL) | ~16 GB |
| KV cache (160K, f16) | ~16 GB |
| Compute graph | ~0.4 GB |
| **Total** | **~32 GB** |
| **Free VRAM** | **~64 GB** |

Larger models (Coder-Next at 43GB) use more weight memory but still fit within 96GB at 160K context.

## Performance Summary

**Starting point**: Ollama Q8_0 ROCm = 44.7 tok/s generation

**Final result**: llama.cpp IQ4_NL Vulkan = **87 tok/s generation** (+95%)

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
