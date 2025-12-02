"""
Training script for decoder-only transformer on FineWeb dataset.
Supports 1B and 7B models with full architecture configuration.
Includes checkpointing for disruption recovery and cosine LR schedule with min_lr.
"""

import os
import json
import time
import math
import signal
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer
from tqdm import tqdm

from model import create_model_from_config, create_1b_model, create_7b_model
from data_loader import create_dataloader
from config import TrainingConfig

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def get_cosine_schedule_with_warmup_and_min_lr(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1
) -> LambdaLR:
    """
    Create a schedule with linear warmup followed by cosine decay to min_lr.
    
    Unlike the standard cosine schedule which decays to 0, this decays to
    min_lr_ratio * initial_lr (typically 10% of peak LR).
    
    Args:
        optimizer: The optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum LR as ratio of initial LR (default 0.1 = 10%)
    
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        # Warmup phase: linear increase from 0 to 1
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay phase: from 1 to min_lr_ratio
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        # Cosine annealing: starts at 1, ends at min_lr_ratio
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Scale: when progress=0, cosine_decay=1.0 -> returns 1.0
        #        when progress=1, cosine_decay=0.0 -> returns min_lr_ratio
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda)


from contextlib import contextmanager

@contextmanager
def defer_interrupts():
    """
    Context manager that defers SIGINT (Ctrl+C) until the block completes.
    This ensures atomic operations like checkpoint saves complete fully.
    """
    # Use a mutable container to track if interrupt was received DURING this block
    # This avoids issues with nested calls or re-entry after previous interrupts
    interrupt_received = [False]
    original_handler = signal.getsignal(signal.SIGINT)
    
    def deferred_handler(signum, frame):
        interrupt_received[0] = True
        print("\n[Interrupt received - will exit after current operation completes...]")
    
    signal.signal(signal.SIGINT, deferred_handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupt_received[0]:
            # Re-raise the interrupt now that we're done
            raise KeyboardInterrupt()

def setup_distributed():
    """Setup distributed training if available"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def broadcast_from_main(obj, src: int = 0):
    """
    Broadcast a picklable Python object from the main process to all other ranks.
    Returns the broadcast object on every rank (or the original object if not distributed).
    """
    if dist.is_available() and dist.is_initialized():
        object_list = [obj]
        dist.broadcast_object_list(object_list, src=src)
        return object_list[0]
    return obj


class CheckpointManager:
    """Manages saving and loading checkpoints with atomic writes"""
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _atomic_save(self, checkpoint: dict, path: Path):
        """
        Save checkpoint atomically by writing to a temp file first, then renaming.
        This prevents corruption if interrupted mid-save.
        """
        temp_path = path.with_suffix('.pt.tmp')
        torch.save(checkpoint, temp_path)
        # os.replace is atomic on POSIX systems
        os.replace(temp_path, path)
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        step: int,
        loss: float,
        config: TrainingConfig,
        is_best: bool = False,
        scaler: Any = None
    ):
        """Save training checkpoint atomically with interrupt protection"""
        # Defer interrupts during the entire save operation
        with defer_interrupts():
            checkpoint = {
                'epoch': epoch,
                'step': step,
                'model_state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'loss': loss,
                'config': config.__dict__
            }
            
            # Save latest checkpoint (atomic)
            latest_path = self.checkpoint_dir / 'checkpoint_latest.pt'
            self._atomic_save(checkpoint, latest_path)
            
            # Save step checkpoint (atomic)
            step_path = self.checkpoint_dir / f'checkpoint_step_{step}.pt'
            self._atomic_save(checkpoint, step_path)
            
            # Save best checkpoint (atomic)
            if is_best:
                best_path = self.checkpoint_dir / 'checkpoint_best.pt'
                self._atomic_save(checkpoint, best_path)
            
            # Save metadata (also atomic)
            metadata = {
                'latest_step': step,
                'latest_epoch': epoch,
                'latest_loss': loss,
                'checkpoint_dir': str(self.checkpoint_dir)
            }
            metadata_path = self.checkpoint_dir / 'metadata.json'
            temp_metadata_path = metadata_path.with_suffix('.json.tmp')
            with open(temp_metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            os.replace(temp_metadata_path, metadata_path)
            
            # Clean up any leftover temp files from previous interrupted saves
            self._cleanup_temp_files()
    
    def _cleanup_temp_files(self):
        """Remove any leftover .tmp files from interrupted saves"""
        for tmp_file in self.checkpoint_dir.glob('*.tmp'):
            try:
                tmp_file.unlink()
            except OSError:
                pass
    
    def _try_load_checkpoint(self, checkpoint_path: Path) -> Optional[dict]:
        """Try to load a checkpoint file, return None if corrupted"""
        try:
            return torch.load(checkpoint_path, map_location='cpu')
        except (RuntimeError, EOFError, Exception) as e:
            print(f"Warning: Failed to load {checkpoint_path}: {e}")
            return None
    
    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        checkpoint_path: Optional[str] = None,
        scaler: Any = None
    ) -> Dict[str, Any]:
        """Load training checkpoint with fallback to previous checkpoints if corrupted"""
        self._cleanup_temp_files()  # Clean up any temp files first
        
        if checkpoint_path is None:
            # Try to load latest
            latest_path = self.checkpoint_dir / 'checkpoint_latest.pt'
            if not latest_path.exists():
                return None
            checkpoint_path = latest_path
        else:
            checkpoint_path = Path(checkpoint_path)
        
        # Try loading the requested checkpoint
        checkpoint = self._try_load_checkpoint(checkpoint_path)
        
        # If failed, try to find a valid checkpoint
        if checkpoint is None:
            print("Primary checkpoint corrupted. Searching for valid backup...")
            
            # Get all step checkpoints sorted by step number (descending)
            step_checkpoints = sorted(
                self.checkpoint_dir.glob('checkpoint_step_*.pt'),
                key=lambda p: int(p.stem.split('_')[-1]),
                reverse=True
            )
            
            for backup_path in step_checkpoints:
                if backup_path == checkpoint_path:
                    continue  # Skip the one we already tried
                print(f"Trying backup: {backup_path}")
                checkpoint = self._try_load_checkpoint(backup_path)
                if checkpoint is not None:
                    print(f"Successfully loaded backup checkpoint from step {checkpoint['step']}")
                    # Remove corrupted checkpoint
                    try:
                        Path(checkpoint_path).unlink()
                        print(f"Removed corrupted checkpoint: {checkpoint_path}")
                    except OSError:
                        pass
                    break
            
            if checkpoint is None:
                raise RuntimeError(
                    f"All checkpoints are corrupted. Please delete {self.checkpoint_dir} and restart training."
                )
        
        # Load model state
        if isinstance(model, DDP):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state (for fp16 training)
        if scaler and checkpoint.get('scaler_state_dict'):
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        return checkpoint
    
    def get_latest_checkpoint_info(self) -> Optional[Dict[str, Any]]:
        """Get information about latest checkpoint"""
        metadata_path = self.checkpoint_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    criterion: nn.Module,
    autocast_dtype: Optional[torch.dtype] = None
) -> float:
    """Perform a single training step"""
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    
    # Forward pass with mixed precision if specified
    if autocast_dtype is not None:
        with torch.amp.autocast('cuda', dtype=autocast_dtype):
            logits = model(input_ids)
            # Reshape for loss calculation
            # logits: [batch_size, seq_len, vocab_size]
            # labels: [batch_size, seq_len]
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
    else:
        logits = model(input_ids)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
    
    return loss.item()


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    max_batches: int = 100,
    autocast_dtype: Optional[torch.dtype] = None
) -> float:
    """Validate model on validation set"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            
            loss = train_step(model, batch, device, criterion, autocast_dtype)
            total_loss += loss
            num_batches += 1
    
    model.train()
    return total_loss / num_batches if num_batches > 0 else 0.0


def train(config: TrainingConfig):
    """Main training function"""
    # Setup distributed training if available
    rank, world_size, local_rank = setup_distributed()
    is_main_process = rank == 0
    
    # Device setup
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    if is_main_process:
        print(f"Training configuration:")
        print(json.dumps(config.__dict__, indent=2, default=str))
        print(f"\nUsing device: {device}")
        print(f"World size: {world_size}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Set tokenizer max length to match our config to suppress warnings
    tokenizer.model_max_length = config.max_seq_len
    
    # Use config vocab_size, but update if tokenizer differs
    vocab_size = config.vocab_size
    tokenizer_vocab_size = len(tokenizer)
    if tokenizer_vocab_size != vocab_size:
        if is_main_process:
            print(f"Warning: Config vocab_size ({vocab_size}) differs from tokenizer ({tokenizer_vocab_size})")
            print(f"Using tokenizer vocab_size: {tokenizer_vocab_size}")
        vocab_size = tokenizer_vocab_size
        config.vocab_size = vocab_size
    
    # Create model from config
    model = create_model_from_config(config)
    
    if is_main_process:
        num_params = model.get_num_params()
        print(f"\nModel created with {num_params / 1e9:.2f}B parameters")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Layers: {config.num_hidden_layers}")
        print(f"  Attention heads: {config.num_attention_heads}")
        print(f"  KV heads: {config.num_key_value_heads or config.num_attention_heads}")
        print(f"  FFN dim: {config.intermediate_size}")
        print(f"  Max seq len: {config.max_seq_len}")
        print(f"  RoPE theta: {config.rope_theta}")
    
    # Setup dtype for mixed precision (required for FlashAttention)
    # GradScaler is needed for fp16 to prevent gradient underflow; bf16 doesn't need it
    scaler = None
    if config.dtype == "bf16":
        dtype = torch.bfloat16
        autocast_dtype = torch.bfloat16
    elif config.dtype == "fp16":
        dtype = torch.float16
        autocast_dtype = torch.float16
        scaler = torch.amp.GradScaler('cuda')  # Prevents gradient underflow in fp16
    else:
        dtype = torch.float32
        autocast_dtype = None  # No autocast for fp32 - avoids unnecessary overhead
    
    model = model.to(device)
    
    # Convert model to appropriate dtype (for FlashAttention compatibility)
    if config.dtype in ["bf16", "fp16"]:
        model = model.to(dtype)
        if is_main_process:
            print(f"Model converted to {config.dtype} for FlashAttention compatibility")
    
    # Setup distributed model
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    # Calculate total steps
    # Tokens per micro-batch (before gradient accumulation)
    tokens_per_micro_batch = config.batch_size * config.max_seq_len * world_size
    # Effective tokens per optimizer step (after gradient accumulation)
    tokens_per_step = tokens_per_micro_batch * config.gradient_accumulation_steps
    # Total optimizer steps (not micro-batches)
    total_steps = int(config.total_tokens / tokens_per_step)
    
    # Get warmup steps (uses warmup_steps if set, else calculates from warmup_ratio)
    warmup_steps = config.get_warmup_steps(total_steps)
    
    # Calculate min_lr ratio for scheduler
    min_lr_ratio = config.min_lr / config.learning_rate
    
    # Use cosine schedule with warmup and min_lr
    scheduler = get_cosine_schedule_with_warmup_and_min_lr(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr_ratio=min_lr_ratio
    )
    
    if is_main_process:
        print(f"\nTraining schedule:")
        print(f"  Total tokens: {config.total_tokens / 1e9:.2f}B")
        print(f"  Tokens per micro-batch: {tokens_per_micro_batch:,}")
        print(f"  Gradient accumulation steps: {config.gradient_accumulation_steps}")
        print(f"  Effective tokens per step: {tokens_per_step:,}")
        print(f"  Total optimizer steps: {total_steps:,}")
        print(f"  Warmup steps: {warmup_steps:,}")
        print(f"  Peak LR: {config.learning_rate:.2e}")
        print(f"  Min LR: {config.min_lr:.2e}")
        print(f"  LR schedule: Cosine with warmup")
    
    # Create dataloader with distributed sharding
    dataloader = create_dataloader(
        data_path=config.data_path,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_length=config.max_seq_len,
        num_workers=config.num_workers,
        pin_memory=True,
        limit=None,  # Use all available data
        streaming=True,
        target_tokens=config.total_tokens,  # Limit to target token count
        rank=rank,
        world_size=world_size
    )
    
    # Checkpoint manager
    checkpoint_manager = CheckpointManager(config.checkpoint_dir)
    
    # Resume from checkpoint if available
    start_epoch = 0
    start_step = 0
    best_loss = float('inf')
    
    checkpoint_info = None
    if config.resume:
        metadata = checkpoint_manager.get_latest_checkpoint_info() if is_main_process else None
        checkpoint_info = broadcast_from_main(metadata, src=0) if world_size > 1 else metadata
        if checkpoint_info:
            if is_main_process:
                print(f"\nResuming from checkpoint at step {checkpoint_info['latest_step']}")
            checkpoint = checkpoint_manager.load_checkpoint(model, optimizer, scheduler, scaler=scaler)
            if checkpoint:
                start_epoch = checkpoint['epoch']
                start_step = checkpoint['step']
                best_loss = checkpoint.get('loss', float('inf'))
            else:
                raise RuntimeError("Failed to load checkpoint despite metadata indicating availability.")
        elif is_main_process:
            print("\n--resume flag provided but no checkpoint metadata found. Starting from scratch.")
        if world_size > 1:
            # Make sure every rank finishes loading before training resumes
            dist.barrier()
    
    # Training loop
    model.train()
    global_step = start_step
    epoch = start_epoch
    
    # Progress tracking
    if is_main_process:
        pbar = tqdm(total=total_steps, initial=start_step, desc="Training")
    
    running_loss = 0.0
    loss_count = 0
    last_logged_loss = None
    accumulation_counter = 0
    
    # Initialize optimizer (zero gradients at start)
    optimizer.zero_grad()
    
    try:
        while global_step < total_steps:
            for batch in dataloader:
                if global_step >= total_steps:
                    break
                
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass with mixed precision
                if autocast_dtype is not None:
                    with torch.amp.autocast('cuda', dtype=autocast_dtype):
                        logits = model(input_ids)
                        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                else:
                    logits = model(input_ids)
                    loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                # Scale loss by accumulation steps for proper averaging
                loss = loss / config.gradient_accumulation_steps
                loss_value = loss.item() * config.gradient_accumulation_steps  # Unscale for logging
                
                # Backward pass (accumulates gradients)
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                accumulation_counter += 1
                running_loss += loss_value
                loss_count += 1
                
                # Update weights only after accumulating enough gradients
                if accumulation_counter % config.gradient_accumulation_steps == 0:
                    if scaler is not None:
                        # Unscale gradients before clipping when using GradScaler
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Gradient clipping (on accumulated gradients)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    # Increment global step (one step per accumulation cycle)
                    global_step += 1
                
                    # Logging
                    if is_main_process and global_step % config.log_interval == 0:
                        avg_loss = running_loss / loss_count
                        current_lr = scheduler.get_last_lr()[0]
                        
                        pbar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'lr': f'{current_lr:.2e}',
                            'step': global_step
                        })
                        pbar.update(config.log_interval)
                        
                        running_loss = 0.0
                        loss_count = 0
                        last_logged_loss = avg_loss
                    
                    # Checkpointing
                    if global_step % config.checkpoint_interval == 0:
                        if is_main_process:
                            avg_loss = last_logged_loss
                            if avg_loss is None:
                                avg_loss = running_loss / loss_count if loss_count > 0 else 0.0
                            is_best = avg_loss < best_loss
                            if is_best:
                                best_loss = avg_loss
                            
                            checkpoint_manager.save_checkpoint(
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                epoch=epoch,
                                step=global_step,
                                loss=avg_loss,
                                config=config,
                                is_best=is_best,
                                scaler=scaler
                            )
                            print(f"\nCheckpoint saved at step {global_step}, loss: {avg_loss:.4f}")
            
            epoch += 1
    
    except KeyboardInterrupt:
        if is_main_process:
            print("\nTraining interrupted by user.")
    
    finally:
        try:
            if is_main_process:
                pbar.close()
                
                # Finish any remaining accumulation
                if accumulation_counter % config.gradient_accumulation_steps != 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                # Save final checkpoint (atomic save with interrupt protection)
                avg_loss = running_loss / loss_count if loss_count > 0 else 0.0
                print(f"\nSaving final checkpoint at step {global_step}...")
                checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    step=global_step,
                    loss=avg_loss,
                    config=config,
                    is_best=False,
                    scaler=scaler
                )
                print(f"Checkpoint saved successfully.")
        except KeyboardInterrupt:
            # Second interrupt during final save - checkpoint still completed due to defer_interrupts
            if is_main_process:
                print("\nSecond interrupt received. Checkpoint was saved. Cleaning up...")
        finally:
            # Always cleanup distributed, even if interrupted during final save
            cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description='Train transformer model on FineWeb')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--model-size', type=str, choices=['1b', '7b'], default=None,
                        help='Use preset config for model size (overrides --config)')
    args = parser.parse_args()
    
    # Load config
    if args.model_size == '1b':
        config = TrainingConfig.for_1b_model()
        print("Using preset 1B model configuration")
    elif args.model_size == '7b':
        config = TrainingConfig.for_7b_model()
        print("Using preset 7B model configuration")
    elif os.path.exists(args.config):
        config = TrainingConfig.from_yaml(args.config)
    else:
        config = TrainingConfig()
    
    config.resume = args.resume
    
    # Run training
    train(config)


if __name__ == '__main__':
    main()
