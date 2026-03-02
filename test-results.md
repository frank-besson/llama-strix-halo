# Model Capability Test Results

Tested on AMD Strix Halo (gfx1151), Vulkan backend, llama.cpp with `--jinja`.

| Model | Tool Calling | Instruction | Reasoning | Total | Gen tok/s |
|-------|-------------|-------------|-----------|-------|-----------|
| qwen3-coder-30b | 2/5 | 3/4 | 2/3 | **7/12** | 87.26 |
| qwen3-coder-next | 2/5 | 3/4 | 3/3 | **8/12** | 43.66 |
| nemotron-nano | 2/5 | 3/4 | 3/3 | **8/12** | 74.52 |
| qwen3.5-35b | 2/5 | 2/4 | 3/3 | **7/12** | 40.97 |
| gpt-oss-20b | 2/5 | 3/4 | 3/3 | **8/12** | 76.57 |
| glm-4.7-flash | 2/5 | 3/4 | 3/3 | **8/12** | 69.35 |
