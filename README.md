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
| MoE 30B (3B active) | 3B | ~43-72 tok/s |
| Dense 32B | 32B | ~6.4 tok/s |
| Dense 70B | 70B | ~3 tok/s |

### Why llama.cpp over Ollama?

- 13-80% faster token generation (no abstraction overhead)
- Direct control over GPU flags (UMA, rocWMMA flash attention)
- Better prompt processing speeds

### Why Q8 Quantization?

On bandwidth-limited hardware, FP16 and Q8 produce the same token generation speed (both bottlenecked by bandwidth). Q8 uses less VRAM with negligible quality loss.

## Model

**Qwen3-Coder-30B-A3B-Instruct** (Q8_0, Unsloth GGUF)
- 30.5B total params, 3.3B active (MoE, 128 experts, 8 active)
- 256K native context
- Tool calling support (required for coding agents)
- Unsloth GGUF required — Ollama's GGUF has broken Jinja templates for tool calling in llama.cpp

Download:
```bash
curl -L -o ~/models/Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf \
  https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/resolve/main/Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf
```

### Why Not GLM-4.7-Flash?

GLM-4.7-Flash scores higher on SWE-Bench (73.8%) but uses the `glm4moelite` architecture which upstream llama.cpp doesn't support yet. It works in Ollama (which has a custom fork) but not standalone llama.cpp.

## Setup

### 1. Build llama.cpp with Strix Halo Optimizations

Standard AUR/pre-built packages miss critical Strix Halo flags. Build from source:

```bash
~/llama/build-llama.sh
```

Key build flags:
- `-DGGML_HIP=ON` — ROCm/HIP backend
- `-DAMDGPU_TARGETS="gfx1100;gfx1151"` — Dual GPU target (gfx1100 kernels are faster on Strix Halo)
- `-DGGML_HIP_UMA=ON` — Unified Memory Architecture (critical for iGPU)
- `-DGGML_HIP_ROCWMMA_FATTN=ON` — Hardware-accelerated flash attention via rocWMMA

Prerequisites:
```bash
sudo pacman -S cmake rocm-hip-sdk
```

After build:
```bash
sudo cp /tmp/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
sudo cp /tmp/llama.cpp/build/bin/llama-cli /usr/local/bin/llama-cli
```

### 2. Fix Ollama Model Permissions

Ollama stores models in `/usr/share/ollama/.ollama/models/blobs/` with restricted permissions. If you want llama.cpp to access those blobs:

```bash
sudo chmod 751 /usr/share/ollama/
sudo chmod o+x /usr/share/ollama/.ollama/ /usr/share/ollama/.ollama/models/ /usr/share/ollama/.ollama/models/blobs/
```

Symlink models for convenience:
```bash
mkdir -p ~/models
sudo ln -s /usr/share/ollama/.ollama/models/blobs/<sha256-hash> ~/models/model-name.gguf
```

### 3. Add ROCm to PATH

Add to `~/.zshrc`:
```bash
export PATH="/opt/rocm/bin:$PATH"
```

### 4. Run the Server

```bash
~/llama/llama-serve.sh
```

### 5. Connect Coding Agents

**OpenCode** (`~/.config/opencode/opencode.json`):
```json
{
  "provider": {
    "llama-cpp": {
      "models": {
        "coder-qwen": {
          "name": "Qwen3 Coder 30B Q8 (llama.cpp)"
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
HIP_VISIBLE_DEVICES=0 llama-server \
  -m ~/models/Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf \
  -c 65536 \        # Context window (64K tokens)
  -fa on \          # Flash attention (critical for long context)
  -ngl 99 \         # Offload all layers to GPU
  --no-mmap \       # Required for Strix Halo (hangs without it)
  -ctk q8_0 \       # KV cache K quantization (halves context VRAM)
  -ctv q8_0 \       # KV cache V quantization
  --jinja \         # Enable Jinja templates (required for tool calling)
  -np 1 \           # Single parallel slot (max bandwidth per session)
  -a coder-qwen     # Model alias for API responses
```

## Strix Halo Gotchas

1. **`--no-mmap` is mandatory** — Without it, model loading hangs indefinitely on Strix Halo.

2. **Flash Attention + Vulkan don't mix** — Vulkan backend is faster for short context token generation, but doesn't support flash attention. For coding agents (long context), use ROCm + flash attention.

3. **gfx1151 kernels are 2-6x slower than gfx1100** — This is a ROCm maturity issue. Performance will improve as AMD updates ROCm for Strix Halo.

4. **ROCm version matters** — ROCm 6.4.4 was benchmarked faster than 7.0.1. Check before upgrading.

5. **Kernel version matters** — 15% performance difference observed between Linux 6.14 and 6.15. Use latest stable.

6. **`glm4moelite` not supported in upstream llama.cpp** — GLM-4.7-Flash only works via Ollama's custom fork.

7. **Ollama GGUF tool templates are broken** — Qwen3-Coder's Ollama GGUF uses Jinja features (`reject("in", ...)`) that llama.cpp's minja parser doesn't support. Use Unsloth GGUFs for llama.cpp.

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

Create optimized Modelfiles with reduced context (200K+ context is too slow):
```bash
# ~/llama/coder-glm.Modelfile
FROM glm-4.7-flash:q8_0
PARAMETER num_ctx 65536
PARAMETER temperature 1

# ~/llama/coder-qwen.Modelfile
FROM qwen3-coder:30b-a3b-q8_0
PARAMETER num_ctx 65536
PARAMETER temperature 0.7
PARAMETER top_k 20
PARAMETER top_p 0.8
PARAMETER repeat_penalty 1.05
```

## VRAM Budget

| Component | Size |
|-----------|------|
| Model weights (Q8) | ~30 GB |
| KV cache (64K, q8_0) | ~3.3 GB |
| Compute graph | ~0.4 GB |
| **Total** | **~34 GB** |
| **Free VRAM** | **~62 GB** |

## Performance

Benchmarked with Qwen3-Coder-30B-A3B Q8_0, 64K context, flash attention enabled.

| Metric | llama.cpp | Ollama |
|--------|-----------|--------|
| Generation (100 tok) | **49.0 tok/s** | 44.7 tok/s |
| Generation (500 tok) | **48.4 tok/s** | 43.3 tok/s |
| Prompt eval (short) | 156-182 tok/s | 257-327 tok/s |
| Prompt eval (151 tok) | **894 tok/s** | 860 tok/s |

Key tuning wins for llama.cpp:
1. Dual GPU target (`gfx1100;gfx1151`) — gfx1100 kernels run faster on Strix Halo
2. KV cache quantization (`-ctk q8_0 -ctv q8_0`) — reduces memory bandwidth pressure
3. KV cache on GPU (default, don't use `-nkvo`) — avoids CPU-GPU transfers on unified memory

## Resources

- [Strix Halo LLM Performance Tests](https://community.frame.work/t/amd-strix-halo-ryzen-ai-max-395-gpu-llm-performance-tests/72521)
- [Strix Halo Toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes)
- [Strix Halo Testing Repo](https://github.com/lhl/strix-halo-testing)
- [llama.cpp ROCm HIP Discussion](https://github.com/ggml-org/llama.cpp/discussions/15021)
- [Unsloth Qwen3-Coder GGUF (fixed tool calling)](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF)
- [OpenCode + llama.cpp Jinja fix](https://github.com/sst/opencode/issues/1890)
- [Ollama Claude Code Integration](https://docs.ollama.com/integrations/claude-code)
