"""
Training script for 1B parameter decoder-only transformer on FineWeb dataset.
Includes checkpointing for disruption recovery.
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from model import create_1b_model
from data_loader import create_dataloader
from config import TrainingConfig

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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


class CheckpointManager:
    """Manages saving and loading checkpoints"""
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        step: int,
        loss: float,
        config: TrainingConfig,
        is_best: bool = False
    ):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': loss,
            'config': config.__dict__
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, latest_path)
        
        # Save step checkpoint
        step_path = self.checkpoint_dir / f'checkpoint_step_{step}.pt'
        torch.save(checkpoint, step_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
        
        # Save metadata
        metadata = {
            'latest_step': step,
            'latest_epoch': epoch,
            'latest_loss': loss,
            'checkpoint_dir': str(self.checkpoint_dir)
        }
        metadata_path = self.checkpoint_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load training checkpoint"""
        if checkpoint_path is None:
            # Try to load latest
            latest_path = self.checkpoint_dir / 'checkpoint_latest.pt'
            if not latest_path.exists():
                return None
            checkpoint_path = latest_path
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
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
        print(json.dumps(config.__dict__, indent=2))
        print(f"\nUsing device: {device}")
        print(f"World size: {world_size}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    vocab_size = len(tokenizer)
    
    # Create model
    model = create_1b_model(
        vocab_size=vocab_size,
        max_seq_len=config.max_seq_len,
        use_checkpoint=config.activation_checkpointing
    )
    
    if is_main_process:
        num_params = model.get_num_params()
        print(f"\nModel created with {num_params / 1e9:.2f}B parameters")
    
    # Setup dtype for mixed precision (required for FlashAttention)
    if config.dtype == "bf16":
        dtype = torch.bfloat16
        autocast_dtype = torch.bfloat16
    elif config.dtype == "fp16":
        dtype = torch.float16
        autocast_dtype = torch.float16
    else:
        dtype = torch.float32
        autocast_dtype = torch.float32
    
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
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    if is_main_process:
        print(f"\nTraining schedule:")
        print(f"  Total tokens: {config.total_tokens / 1e9:.2f}B")
        print(f"  Tokens per micro-batch: {tokens_per_micro_batch:,}")
        print(f"  Gradient accumulation steps: {config.gradient_accumulation_steps}")
        print(f"  Effective tokens per step: {tokens_per_step:,}")
        print(f"  Total optimizer steps: {total_steps:,}")
        print(f"  Warmup steps: {warmup_steps:,}")
    
    # Create dataloader
    dataloader = create_dataloader(
        data_path=config.data_path,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_length=config.max_seq_len,
        num_workers=config.num_workers,
        pin_memory=True,
        limit=None,  # Use all available data
        streaming=True,
        target_tokens=config.total_tokens  # Limit to target token count
    )
    
    # Checkpoint manager
    checkpoint_manager = CheckpointManager(config.checkpoint_dir)
    
    # Resume from checkpoint if available
    start_epoch = 0
    start_step = 0
    best_loss = float('inf')
    
    checkpoint_info = checkpoint_manager.get_latest_checkpoint_info()
    if checkpoint_info and config.resume:
        if is_main_process:
            print(f"\nResuming from checkpoint at step {checkpoint_info['latest_step']}")
        checkpoint = checkpoint_manager.load_checkpoint(model, optimizer, scheduler)
        if checkpoint:
            start_epoch = checkpoint['epoch']
            start_step = checkpoint['step']
            best_loss = checkpoint.get('loss', float('inf'))
    
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
                loss.backward()
                
                accumulation_counter += 1
                running_loss += loss_value
                loss_count += 1
                
                # Update weights only after accumulating enough gradients
                if accumulation_counter % config.gradient_accumulation_steps == 0:
                    # Gradient clipping (on accumulated gradients)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    
                    # Optimizer step
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
                                is_best=is_best
                            )
                            print(f"\nCheckpoint saved at step {global_step}, loss: {avg_loss:.4f}")
            
            epoch += 1
    
    except KeyboardInterrupt:
        if is_main_process:
            print("\nTraining interrupted. Saving checkpoint...")
            # Finish current accumulation if needed
            if accumulation_counter % config.gradient_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            avg_loss = running_loss / loss_count if loss_count > 0 else 0.0
            checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=global_step,
                loss=avg_loss,
                config=config,
                is_best=False
            )
    
    finally:
        if is_main_process:
            pbar.close()
            # Finish any remaining accumulation
            if accumulation_counter % config.gradient_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            # Final checkpoint
            avg_loss = running_loss / loss_count if loss_count > 0 else 0.0
            checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=global_step,
                loss=avg_loss,
                config=config,
                is_best=False
            )
            print(f"\nTraining completed. Final checkpoint saved.")
        
        cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description='Train 1B parameter model on FineWeb')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    args = parser.parse_args()
    
    # Load config
    config = TrainingConfig.from_yaml(args.config) if os.path.exists(args.config) else TrainingConfig()
    config.resume = args.resume
    
    # Run training
    train(config)


if __name__ == '__main__':
    main()

