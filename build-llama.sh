#!/bin/bash
set -e

BACKEND="${1:-vulkan}"

cd /tmp
rm -rf llama.cpp

git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

if [ "$BACKEND" = "rocm" ] || [ "$BACKEND" = "hip" ]; then
  echo "Building with ROCm/HIP backend..."
  export PATH="/opt/rocm/bin:$PATH"
  export HIP_PATH="/opt/rocm"
  export ROCM_PATH="/opt/rocm"
  export CMAKE_PREFIX_PATH="/opt/rocm"

  cmake -S . -B build \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS="gfx1100;gfx1151" \
    -DGGML_HIP_UMA=ON \
    -DGGML_HIP_ROCWMMA_FATTN=ON \
    -DCMAKE_HIP_COMPILER="/opt/rocm/lib/llvm/bin/clang++" \
    -DCMAKE_BUILD_TYPE=Release
else
  echo "Building with Vulkan backend..."
  cmake -S . -B build \
    -DGGML_VULKAN=ON \
    -DCMAKE_BUILD_TYPE=Release
fi

cmake --build build --config Release -j$(nproc)

echo ""
echo "Build complete ($BACKEND). Install with:"
echo "  sudo cp /tmp/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server"
echo "  sudo cp /tmp/llama.cpp/build/bin/llama-cli /usr/local/bin/llama-cli"
