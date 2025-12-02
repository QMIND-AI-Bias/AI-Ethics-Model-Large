"""
Configuration for training transformer models on FineWeb dataset.
Supports both 1B and 7B parameter models with full architecture configuration.
"""

import yaml
from dataclasses import dataclass, asdict, field
from typing import Optional
from pathlib import Path


@dataclass
class TrainingConfig:
    """Training configuration with full model architecture support"""
    
    # ==========================================
    # Model Architecture Parameters
    # ==========================================
    vocab_size: int = 50257  # GPT-2 default; use 128256 for Llama-3
    hidden_size: int = 2048  # Model dimension (dim)
    intermediate_size: int = 8192  # FFN dimension (SwiGLU)
    num_hidden_layers: int = 24  # Number of transformer blocks
    num_attention_heads: int = 16  # Query heads
    num_key_value_heads: Optional[int] = None  # KV heads for GQA (None = same as num_attention_heads)
    max_seq_len: int = 4096  # Maximum sequence length / context window
    rms_norm_eps: float = 1e-6  # RMSNorm epsilon
    rope_theta: float = 10000.0  # RoPE base frequency
    tie_word_embeddings: bool = True  # Tie input/output embeddings
    
    # ==========================================
    # Training Parameters
    # ==========================================
    batch_size: int = 4  # Per GPU batch size
    learning_rate: float = 3e-4  # Peak learning rate
    min_lr: Optional[float] = None  # Minimum LR for cosine decay (default: 10% of learning_rate)
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    
    # Warmup configuration (warmup_steps takes precedence if set)
    warmup_steps: Optional[int] = None  # Explicit warmup steps (overrides warmup_ratio)
    warmup_ratio: float = 0.01  # Warmup as fraction of total steps
    
    gradient_accumulation_steps: int = 32
    activation_checkpointing: bool = True  # Enable activation checkpointing to save memory
    
    # ==========================================
    # Data Parameters
    # ==========================================
    data_path: str = "hf://datasets/HuggingFaceFW/fineweb/sample/100BT"
    total_tokens: int = 7_700_000_000  # Target tokens to train on
    num_workers: int = 0  # Must be 0 for IterableDataset (streaming)
    
    # ==========================================
    # Tokenizer
    # ==========================================
    tokenizer_name: str = "gpt2"
    
    # ==========================================
    # Mixed Precision Training
    # ==========================================
    # Options: "fp16", "bf16", or "fp32" (fp32 will not work with FlashAttention)
    dtype: str = "bf16"  # bf16 is preferred for training stability
    
    # ==========================================
    # Checkpointing
    # ==========================================
    checkpoint_dir: str = "./checkpoints"
    checkpoint_interval: int = 1000  # Save checkpoint every N steps
    log_interval: int = 10  # Log every N steps
    
    # Resume training
    resume: bool = False
    
    def to_yaml(self, path: str):
        """Save config to YAML file"""
        # Convert to dict, handling Optional types
        data = asdict(self)
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'TrainingConfig':
        """Load config from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def __post_init__(self):
        """Validate and set defaults for configuration"""
        # Basic validation
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.max_seq_len > 0, "max_seq_len must be positive"
        assert self.total_tokens > 0, "total_tokens must be positive"
        assert self.gradient_accumulation_steps > 0, "gradient_accumulation_steps must be positive"
        assert self.dtype in ["fp16", "bf16", "fp32"], f"dtype must be 'fp16', 'bf16', or 'fp32', got {self.dtype}"
        
        # Validate architecture
        assert self.hidden_size > 0, "hidden_size must be positive"
        assert self.intermediate_size > 0, "intermediate_size must be positive"
        assert self.num_hidden_layers > 0, "num_hidden_layers must be positive"
        assert self.num_attention_heads > 0, "num_attention_heads must be positive"
        assert self.hidden_size % self.num_attention_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"
        
        # Validate GQA configuration
        if self.num_key_value_heads is not None:
            assert self.num_key_value_heads > 0, "num_key_value_heads must be positive if specified"
            assert self.num_attention_heads % self.num_key_value_heads == 0, \
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})"
        
        # Set default min_lr if not specified (10% of max LR)
        if self.min_lr is None:
            self.min_lr = self.learning_rate * 0.1
        
        # Validate warmup
        if self.warmup_steps is not None:
            assert self.warmup_steps >= 0, "warmup_steps must be non-negative"
        assert 0.0 <= self.warmup_ratio <= 1.0, "warmup_ratio must be between 0 and 1"
    
    def get_warmup_steps(self, total_steps: int) -> int:
        """
        Get the number of warmup steps.
        Uses warmup_steps if explicitly set, otherwise calculates from warmup_ratio.
        """
        if self.warmup_steps is not None:
            return self.warmup_steps
        return int(total_steps * self.warmup_ratio)
    
    @classmethod
    def for_1b_model(cls) -> 'TrainingConfig':
        """Create default config for 1B parameter model"""
        return cls(
            vocab_size=50257,
            hidden_size=2048,
            intermediate_size=8192,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=None,  # Standard MHA
            max_seq_len=4096,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            tie_word_embeddings=True,
            learning_rate=3e-4,
            batch_size=4,
            gradient_accumulation_steps=32,
            activation_checkpointing=True,
            tokenizer_name="gpt2",
            total_tokens=7_700_000_000,
        )
    
    @classmethod
    def for_7b_model(cls) -> 'TrainingConfig':
        """
        Create default config for 7B parameter model.
        Optimized for 2x RTX Blackwell (96GB total) with ~30 day training target.
        """
        return cls(
            # Architecture (Llama-2/Mistral style)
            vocab_size=128256,  # Llama-3 tokenizer
            hidden_size=4096,
            intermediate_size=11008,  # SwiGLU ratio ~2.7x
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,  # GQA for fast inference
            max_seq_len=8192,  # Long context
            rms_norm_eps=1e-5,
            rope_theta=500000.0,  # Extended context RoPE
            tie_word_embeddings=False,
            
            # Training
            learning_rate=1.5e-4,  # Lower LR for larger model
            min_lr=1.5e-5,  # 10% of max LR
            warmup_steps=2000,
            batch_size=2,  # Low to fit in VRAM without checkpointing
            gradient_accumulation_steps=64,  # ~2.1M tokens/step
            activation_checkpointing=False,  # Speed over memory
            
            # Data
            tokenizer_name="unsloth/Llama-3.2-1B",  # Llama-3 tokenizer (ungated)
            total_tokens=60_000_000_000,  # 60B tokens
            
            # Other
            dtype="bf16",
            checkpoint_interval=500,  # More frequent checkpoints for long training
        )
