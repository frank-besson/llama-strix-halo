# Model Capability Test Results

Tested on AMD Strix Halo (gfx1151), Vulkan backend, llama.cpp with `--jinja`.

| Model | Tool Calling | Instruction | Reasoning | Total | Gen tok/s |
|-------|-------------|-------------|-----------|-------|-----------|
| gpt-oss-20b | 2/5 | 3/4 | 3/3 | **8/12** | 74.83 |
| glm-4.7-flash | 2/5 | 3/4 | 3/3 | **8/12** | 69.64 |
| nemotron-nano | 2/5 | 3/4 | 3/3 | **8/12** | 74.14 |
| qwen3-next-80b | 2/5 | 3/4 | 3/3 | **8/12** | 44.30 |
| coder-qwen | 2/5 | 3/4 | 2/3 | **7/12** | 87.42 |
| qwen3-2507 | 2/5 | 3/4 | 2/3 | **7/12** | 87.10 |
| mistral-small-3.2 | 1/5 | 3/4 | 1/3 | **5/12** | 15.11 |

## Universal Failures (llama.cpp server limitations)

These 3 tests fail on all models due to llama.cpp server behavior, not model capability:

- **Multi-tool call**: Server only returns 1 `tool_call` per response (no parallel tool calls)
- **Complex args**: Nested JSON object arguments not properly passed through
- **Tool choice: none**: Server ignores `tool_choice: "none"` parameter entirely

## Per-Model Notes

- **gpt-oss-20b, glm-4.7-flash, nemotron-nano, qwen3-next-80b**: Perfect on instruction following and reasoning. Only fail the 3 universal tool calling tests + JSON-only output.
- **coder-qwen, qwen3-2507**: Additionally fail math (17x23+45) — chain-of-thought produces wrong answer.
- **mistral-small-3.2**: Additionally fails multi-turn tool use, math, and code generation. Slowest model at 15 tok/s (dense 24B architecture).
