# NVIDIA Blackwell / RTX 6000 PRO Support

Blackwell GPUs (compute capability **sm\_120**) ship newer CUDA hardware than the
default stable PyTorch builds currently bundled in this repo. Follow the steps
below to get a compatible stack before launching `torchrun`.

---

## 1. Install a PyTorch build that includes sm\_120 kernels

```bash
# Remove any existing CUDA wheels first
pip uninstall -y torch torchvision torchaudio

# Install the PyTorch nightly wheel with CUDA 12.4+
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu124
```

> **Windows / PowerShell:** use the same command inside an elevated PowerShell
> session. Nightly wheels are currently the only PyTorch builds that ship
> kernels for compute capability 12.0.

Verify the install:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("arch list:", torch.cuda.get_arch_list())
print("device cc:", torch.cuda.get_device_capability(0))
PY
```

You should see `sm_120` inside the reported arch list.

---

## 2. Rebuild FlashAttention-2 for Blackwell

FlashAttention compiles GPU-specific kernels and must be rebuilt with the new
architecture flags.

**Linux / WSL:**

```bash
export TORCH_CUDA_ARCH_LIST="sm90;sm120"
export FLASH_ATTENTION_FORCE_BUILD=1
pip install --force-reinstall --no-build-isolation flash-attn
```

**Windows PowerShell:**

```powershell
$env:TORCH_CUDA_ARCH_LIST="sm90;sm120"
$env:FLASH_ATTENTION_FORCE_BUILD=1
pip install --force-reinstall --no-build-isolation flash-attn
```

Run `bash install_flash_attn.sh` (or replicate its steps on Windows) after the
environment variables are set if you prefer using the helper script.

---

## 3. Validate the stack

Once PyTorch and FlashAttention are rebuilt, run:

```bash
python setup_check.py
```

The script now verifies that your installed PyTorch binary advertises support
for `sm_120` and will warn if it does not.

---

## 4. Fallback attention path

If FlashAttention still cannot be compiled for Blackwell, the model will
automatically fall back to PyTorch's native scaled-dot-product attention. This
makes training slower but keeps the code functional until FlashAttention gains
official binaries.


