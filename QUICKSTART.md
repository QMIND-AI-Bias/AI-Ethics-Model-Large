# Quick Start Guide

## 1. Install Dependencies

```bash
# Step 1: Install dependencies (torch will be installed first)
pip install -r requirements.txt

# Step 2: Install FlashAttention-2 (requires CUDA 12.0+ and torch)
# Option A: Use the installation script (recommended - handles build deps automatically)
bash install_flash_attn.sh

# Option B: Manual installation
pip install psutil ninja packaging  # Build dependencies (already in requirements.txt)
pip install flash-attn --no-build-isolation
```

**Important Notes:**
- **torch must be installed first** - flash-attn needs it during build
- **Build dependencies** (psutil, ninja, packaging) are included in requirements.txt
- If flash-attn installation fails, check:
  - CUDA 12.0+ is installed
  - torch is installed with CUDA support
  - You have a compatible GPU (A100, RTX 3090, RTX 4090, H100, etc.)
  - All build dependencies are installed (psutil, ninja, packaging)
- For memory-constrained GPUs, enable activation checkpointing in `config.yaml` (`activation_checkpointing: true`) and use gradient accumulation (`gradient_accumulation_steps`)

## 2. Verify Setup

```bash
python setup_check.py
```

This checks:
- All required packages are installed
- FlashAttention-2 is working
- CUDA is available
- Model files can be imported

## 3. Test Training (CRITICAL - Do this first!)

```bash
python test_training.py
```

This runs a small test to verify:
- Model can be created
- Data can be loaded from FineWeb
- Training loop works
- Checkpointing works

**DO NOT skip this step!** It will save you from discovering issues during a 3-day training run.

## 4. Configure Training

Edit `config.yaml` to adjust:
- `batch_size`: Based on your GPU memory (start with 4, increase if you have memory)
- `data_path`: FineWeb dataset path
- `total_tokens`: Target tokens (7.7B default)
- `checkpoint_interval`: How often to save (default: every 1000 steps)

## 5. Start Training

### Single GPU:
```bash
python train.py --config config.yaml
```

### Multi-GPU (e.g., 4 GPUs):
```bash
torchrun --nproc_per_node=4 train.py --config config.yaml
```

### Resume from checkpoint:
```bash
python train.py --config config.yaml --resume
```

## 6. Monitor Training

Checkpoints are saved in `./checkpoints/`:
- `checkpoint_latest.pt`: Most recent checkpoint
- `checkpoint_best.pt`: Best checkpoint (lowest loss)
- `metadata.json`: Training metadata

## Troubleshooting

### FlashAttention-2 installation fails
- Ensure CUDA 12.0+ is installed
- Try: `pip install flash-attn --no-build-isolation`
- Check GPU compatibility (A100, RTX 3090, RTX 4090, H100)

### Out of memory errors
- Reduce `batch_size` in `config.yaml`
- Reduce `max_seq_len` (e.g., 4096 instead of 8192)
- Use gradient accumulation (not implemented, but can be added)

### Data loading is slow
- Ensure data is on fast storage (NVMe SSD)
- Increase `num_workers` in config (but note token counting may be less accurate)
- For exact token counting, use `num_workers: 0`

### Training interrupted
- Use `--resume` flag to continue from latest checkpoint
- Checkpoints are saved every `checkpoint_interval` steps

## Expected Training Time

- **7.7B tokens** with **1B parameter model**
- **Single A100**: ~3-4 days
- **4x A100**: ~1 day
- **Single RTX 4090**: ~5-6 days

Adjust based on your hardware and batch size.

