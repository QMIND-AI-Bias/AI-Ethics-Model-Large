# AI-Ethics-Model-Small

Training code for a 1.0 billion parameter decoder-only Transformer model on the FineWeb dataset.

## Architecture

The model uses a modern decoder-only Transformer architecture with:
- **FlashAttention-2** for efficient attention computation
- **RMSNorm** for layer normalization
- **SwiGLU** activation function
- **RoPE** (Rotary Positional Embeddings)
- **No biases** in linear layers

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install FlashAttention-2 (requires CUDA 12.0+ and torch to be installed first):
```bash
# Option 1: Use the installation script (recommended)
bash install_flash_attn.sh

# Option 2: Manual installation
pip install psutil ninja packaging  # Build dependencies
pip install flash-attn --no-build-isolation
```

**Important**: 
- FlashAttention-2 requires torch to be installed first because it needs torch during the build process
- Build dependencies (psutil, ninja, packaging) are included in requirements.txt and will be installed automatically
- If you get build errors, make sure torch is installed with CUDA support and all build dependencies are available

### NVIDIA Blackwell / RTX 6000 PRO GPUs

- Install the latest PyTorch nightly wheel with CUDA 12.4+ so that `sm_120` kernels are included.
- Rebuild FlashAttention with `TORCH_CUDA_ARCH_LIST="sm90;sm120"` (the install script auto-detects this).
- Detailed, copy/pasteable commands are documented in `docs/BLACKWELL.md`.

## Configuration

Edit `config.yaml` to adjust training parameters:
- `batch_size`: Per-GPU batch size (adjust based on GPU memory)
- `learning_rate`: Learning rate (default: 3e-4)
- `data_path`: Path to FineWeb dataset
- `total_tokens`: Total tokens to train on (7.7B default)
- `checkpoint_interval`: Steps between checkpoints
- `gradient_accumulation_steps`: Increases effective batch size without more GPU memory
- `activation_checkpointing`: Enable to save VRAM (slower but essential for long sequences)

## Setup Verification

Check if all dependencies are installed:
```bash
python setup_check.py
```

## Testing

Before running full training, test the setup:
```bash
python test_training.py
```

This will:
- Verify model creation
- Test data loading
- Run 100 training steps
- Test checkpointing

**Important:** Run this test first to ensure everything works before starting the full 3-day training run!

## Training

### Single GPU:
```bash
python train.py --config config.yaml
```

### Multi-GPU (Distributed):
```bash
torchrun --nproc_per_node=<num_gpus> train.py --config config.yaml
```

### Resume from checkpoint:
```bash
# Single GPU resume
python train.py --config config.yaml --resume

# Multi-GPU resume
torchrun --nproc_per_node=<num_gpus> train.py --config config.yaml --resume
```

## Checkpoints

Checkpoints are saved in `./checkpoints/`:
- `checkpoint_latest.pt`: Latest checkpoint
- `checkpoint_best.pt`: Best checkpoint (lowest loss)
- `checkpoint_step_<N>.pt`: Checkpoint at step N
- `metadata.json`: Checkpoint metadata

## Dataset

The code uses the FineWeb dataset via datatrove. The default configuration uses:
- Path: `hf://datasets/HuggingFaceFW/fineweb/sample/100BT`
- This is a 100 billion token sample
- The training will automatically stop after processing 7.7B tokens (configured in `config.yaml`)

### Token Limiting with Multiple Workers

The implementation uses **pre-calculated document counting** to accurately limit tokens when using multiple data loading workers:

1. **Sampling Phase**: Before training starts, the code samples 1000 documents to estimate average tokens per document
2. **Calculation**: Calculates how many documents are needed to reach the target token count (7.7B)
3. **Document Limit**: Uses this document count as a limit, which works correctly with multiple workers

This approach:
- ✅ Works accurately with any number of workers (`num_workers > 0`)
- ✅ No performance overhead during training
- ✅ Adds a 5% safety margin to ensure enough tokens
- ✅ One-time estimation at the start (takes ~1-2 minutes)

To use a different sample or adjust the token limit, modify `config.yaml`:
- Change `data_path` to use a different FineWeb sample
- Adjust `total_tokens` to change the training token limit

## Model Configuration

The 1B parameter model uses:
- Hidden dimension: 2048
- Number of layers: 24
- Number of heads: 16
- FFN dimension: 8192 (4x hidden dim)
- Vocabulary size: 50257 (GPT-2 tokenizer)

## Notes

- The model is trained as a next-token predictor
- It will require downstream fine-tuning for conversational/instruction-following tasks
- Training on 7.7B tokens with this architecture should take approximately 3 days on modern GPUs
- Ensure data is stored on fast NVMe SSD for optimal performance
- Checkpoints are saved regularly to handle disruptions - training can be resumed with `--resume`
- Token limiting works correctly with multiple workers using pre-calculated document counting
- FlashAttention-2 requires CUDA 12.0+ and compatible GPUs (A100, RTX 3090, RTX 4090, H100, etc.)
- Activation checkpointing is available (set `activation_checkpointing: true` in `config.yaml`) to cut activation memory usage at the cost of extra compute
