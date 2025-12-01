"""
Configuration for training 1B parameter model on FineWeb dataset.
"""

import yaml
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Model parameters
    vocab_size: int = 50257  # GPT-2 tokenizer default
    max_seq_len: int = 4096
    
    # Training parameters
    batch_size: int = 4  # Per GPU batch size
    learning_rate: float = 3e-4  # Standard for 1B models (keep at 3e-4 with gradient accumulation)
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.01  # 1% of training steps for warmup
    gradient_accumulation_steps: int = 32  # Keeps effective tokens/step ≈1.05M with seq_len=4096 and batch_size=4
    activation_checkpointing: bool = True  # Enable activation checkpointing to save memory
    
    # Data parameters
    data_path: str = "hf://datasets/HuggingFaceFW/fineweb/sample/100BT"  # 100BT sample
    total_tokens: int = 7_700_000_000  # 7.7 billion tokens
    num_workers: int = 0  # Must be 0 for IterableDataset (streaming) to avoid duplicate data
    
    # Tokenizer
    tokenizer_name: str = "gpt2"
    
    # Mixed precision training (required for FlashAttention)
    # Options: "fp16", "bf16", or "fp32" (fp32 will not work with FlashAttention)
    dtype: str = "bf16"  # bf16 is preferred for training stability
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    checkpoint_interval: int = 1000  # Save checkpoint every N steps
    log_interval: int = 10  # Log every N steps
    
    # Resume training
    resume: bool = False
    
    def to_yaml(self, path: str):
        """Save config to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'TrainingConfig':
        """Load config from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.batch_size > 0
        assert self.learning_rate > 0
        assert self.max_seq_len > 0
        assert self.total_tokens > 0
        assert self.gradient_accumulation_steps > 0
        assert self.dtype in ["fp16", "bf16", "fp32"], f"dtype must be 'fp16', 'bf16', or 'fp32', got {self.dtype}"

