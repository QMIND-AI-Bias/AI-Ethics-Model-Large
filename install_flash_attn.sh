#!/bin/bash
# Installation script for FlashAttention-2
# This must be run AFTER torch is installed

echo "Installing FlashAttention-2..."
echo "Note: This requires torch to be installed first"
echo ""

# Check if torch is installed
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: PyTorch is not installed!"
    echo "Please install PyTorch first: pip install torch>=2.0.0"
    exit 1
fi

# Detect GPU architecture (if CUDA is visible)
GPU_ARCH=$(python - <<'PY'
import torch
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability(0)
    print(f"sm_{cc[0]}{cc[1]}")
PY
)

if [ -n "$GPU_ARCH" ]; then
    echo "Detected CUDA compute capability: $GPU_ARCH"
fi

if [ "$GPU_ARCH" = "sm_120" ]; then
    echo "Detected NVIDIA Blackwell GPU. Forcing FlashAttention rebuild with sm_120 support."
    export TORCH_CUDA_ARCH_LIST="sm90;sm120"
    export FLASH_ATTENTION_FORCE_BUILD=1
fi

# Install build dependencies if not already installed
echo "Installing build dependencies (psutil, ninja, packaging)..."
pip install psutil>=5.9.0 ninja>=1.10.0 packaging>=21.0

# Install flash-attn
echo "Installing flash-attn..."
pip install flash-attn --no-build-isolation

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ FlashAttention-2 installed successfully!"
    python -c "from flash_attn import flash_attn_func; print('✓ flash_attn_func can be imported')"
else
    echo ""
    echo "✗ Installation failed. Common issues:"
    echo "  - CUDA 12.0+ required"
    echo "  - Compatible GPU required (A100, RTX 3090, RTX 4090, H100, etc.)"
    echo "  - Make sure torch is installed with CUDA support"
    exit 1
fi

