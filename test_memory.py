"""
Memory test script to verify that the full training configuration fits in GPU memory.
Tests with the current config (seq_len=4096) and sweeps batch sizes.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from model import create_1b_model
from config import TrainingConfig

def test_memory():
    """Test if model fits in memory with full sequence length"""
    print("=" * 60)
    print("Memory Test for Full Training Configuration")
    print("=" * 60)
    
    config = TrainingConfig()
    config.activation_checkpointing = True  # Ensure checkpointing is enabled for memory realism
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    
    # Create model
    print("\nCreating model...")
    model = create_1b_model(
        vocab_size=vocab_size,
        max_seq_len=config.max_seq_len,
        use_checkpoint=config.activation_checkpointing
    )
    num_params = model.get_num_params()
    print(f"Model created with {num_params / 1e9:.2f}B parameters")
    
    # Setup dtype
    if config.dtype == "bf16":
        dtype = torch.bfloat16
    elif config.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    model = model.to(device)
    if config.dtype in ["bf16", "fp16"]:
        model = model.to(dtype)
        print(f"Model converted to {config.dtype}")
    
    # Create dummy input
    print(f"\nTesting memory with seq_len={config.max_seq_len}")
    
    try:
        # Try a sweep of batch sizes to see what fits
        candidate_batches = [1, 2, 4, 6, 8]
        for bs in candidate_batches:
            print(f"\nTesting with batch_size={bs} ...")
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    memory_before = torch.cuda.memory_allocated() / 1e9
                
                input_ids = torch.randint(0, vocab_size, (bs, config.max_seq_len), device=device)
                model.train()
                if config.dtype in ["bf16", "fp16"]:
                    with torch.amp.autocast('cuda', dtype=dtype):
                        logits = model(input_ids)
                else:
                    logits = model(input_ids)
                
                criterion = nn.CrossEntropyLoss(ignore_index=-100)
                labels = torch.randint(0, vocab_size, (bs, config.max_seq_len), device=device)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss.backward()
                
                if torch.cuda.is_available():
                    memory_after = torch.cuda.memory_allocated() / 1e9
                    memory_used = memory_after - memory_before
                    print(f"  ✓ Success. Memory used: {memory_used:.2f} GB (peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB)")
                else:
                    print("  ✓ Success (CPU)")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  ✗ OOM at batch_size={bs}. Max safe batch is {bs-1}.")
                    break
                else:
                    raise
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n✗ Out of Memory Error!")
            print(f"  Your GPU doesn't have enough memory for this configuration.")
            print(f"  Suggestions:")
            print(f"    - Reduce batch_size (try 2 or 1)")
            print(f"    - Reduce max_seq_len (try 4096 or 2048)")
            print(f"    - Use gradient checkpointing (not implemented yet)")
        else:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_memory()

