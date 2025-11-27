"""
Test script to verify training setup works before full training.
Runs on a small subset of data for 100-200 steps.
"""

import torch
from transformers import AutoTokenizer
from model import create_1b_model
from data_loader import create_dataloader
from train import train_step, CheckpointManager
from config import TrainingConfig
import torch.nn as nn
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def test_training():
    """Run a small test training loop"""
    print("=" * 60)
    print("Testing training setup")
    print("=" * 60)
    
    # Create minimal config for testing
    config = TrainingConfig()
    config.batch_size = 2
    config.max_seq_len = 512  # Smaller for testing
    config.data_path = "hf://datasets/HuggingFaceFW/fineweb/sample/100BT"
    config.checkpoint_interval = 50
    config.log_interval = 5
    config.total_tokens = 1_000_000  # Small test run
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    print(f"Tokenizer loaded. Vocab size: {vocab_size}")
    
    # Create model
    print("\nCreating model...")
    model = create_1b_model(
        vocab_size=vocab_size,
        max_seq_len=config.max_seq_len,
        use_checkpoint=config.activation_checkpointing
    )
    num_params = model.get_num_params()
    print(f"Model created with {num_params / 1e9:.2f}B parameters")
    
    # Setup dtype for mixed precision (required for FlashAttention)
    if config.dtype == "bf16":
        dtype = torch.bfloat16
        autocast_dtype = torch.bfloat16
    elif config.dtype == "fp16":
        dtype = torch.float16
        autocast_dtype = torch.float16
    else:
        dtype = torch.float32
        autocast_dtype = None
    
    model = model.to(device)
    
    # Convert model to appropriate dtype (for FlashAttention compatibility)
    if config.dtype in ["bf16", "fp16"]:
        model = model.to(dtype)
        print(f"Model converted to {config.dtype} for FlashAttention compatibility")
    
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
    
    # Create dataloader
    print("\nCreating dataloader...")
    try:
        dataloader = create_dataloader(
            data_path=config.data_path,
            tokenizer=tokenizer,
            batch_size=config.batch_size,
            max_length=config.max_seq_len,
            num_workers=2,  # Fewer workers for testing
            pin_memory=True,
            limit=100,  # Limit to 100 documents for testing
            streaming=True
        )
        print("Dataloader created successfully")
    except Exception as e:
        print(f"Error creating dataloader: {e}")
        print("Make sure datatrove is installed and you have access to HuggingFace datasets")
        return
    
    # Test training loop
    print("\n" + "=" * 60)
    print("Running test training (100 steps)...")
    print("=" * 60)
    
    model.train()
    total_loss = 0.0
    num_steps = 0
    
    try:
        for step, batch in enumerate(dataloader):
            if step >= 100:  # Run 100 steps
                break
            
            # Training step
            optimizer.zero_grad()
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
            
            loss_value = loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            
            total_loss += loss_value
            num_steps += 1
            
            if (step + 1) % 10 == 0:
                avg_loss = total_loss / num_steps
                print(f"Step {step + 1}/100: Loss = {avg_loss:.4f}")
        
        avg_loss = total_loss / num_steps
        print("\n" + "=" * 60)
        print(f"Test completed successfully!")
        print(f"Average loss: {avg_loss:.4f}")
        print("=" * 60)
        
        # Test checkpointing
        print("\nTesting checkpointing...")
        checkpoint_manager = CheckpointManager("./test_checkpoints")
        checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            epoch=0,
            step=100,
            loss=avg_loss,
            config=config,
            is_best=False
        )
        print("Checkpoint saved successfully!")
        
        # Test loading
        print("Testing checkpoint loading...")
        checkpoint = checkpoint_manager.load_checkpoint(model, optimizer, None)
        if checkpoint:
            print(f"Checkpoint loaded successfully! Step: {checkpoint['step']}, Loss: {checkpoint['loss']:.4f}")
        
        print("\n" + "=" * 60)
        print("All tests passed! Ready for full training.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during training test: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease fix the errors before running full training.")


if __name__ == '__main__':
    test_training()

