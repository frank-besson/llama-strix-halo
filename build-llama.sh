#!/bin/bash
set -e

export PATH="/opt/rocm/bin:$PATH"
export HIP_PATH="/opt/rocm"
export ROCM_PATH="/opt/rocm"
export CMAKE_PREFIX_PATH="/opt/rocm"

cd /tmp
rm -rf llama.cpp

git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

cmake -S . -B build \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS="gfx1100;gfx1151" \
  -DGGML_HIP_UMA=ON \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DCMAKE_HIP_COMPILER="/opt/rocm/lib/llvm/bin/clang++" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j$(nproc)

echo ""
echo "Build complete. Install with:"
echo "  sudo cp /tmp/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server"
echo "  sudo cp /tmp/llama.cpp/build/bin/llama-cli /usr/local/bin/llama-cli"
