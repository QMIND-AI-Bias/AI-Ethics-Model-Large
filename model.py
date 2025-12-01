"""
Decoder-only Transformer model with modern architecture components:
- FlashAttention-2
- RMSNorm
- SwiGLU activation
- RoPE positional embeddings
- No biases in linear layers
"""

import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.utils.checkpoint import checkpoint as activation_checkpoint

try:
    from flash_attn import flash_attn_func
    _FLASH_ATTN_AVAILABLE = True
    _FLASH_ATTN_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - only hit when flash-attn missing
    flash_attn_func = None
    _FLASH_ATTN_AVAILABLE = False
    _FLASH_ATTN_IMPORT_ERROR = exc

_FLASH_ATTENTION_FALLBACK_WARNED = False


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return self.weight * (x / (norm + self.eps))


class RoPE(nn.Module):
    """Rotary Positional Embeddings"""
    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Apply rotary embeddings to input tensor.
        x: [batch_size, seq_len, num_heads, head_dim]
        """
        if seq_len is None:
            seq_len = x.shape[1]
        
        device = x.device
        dtype = x.dtype
        
        # Move inv_freq to correct device only if needed (buffer follows model.to() automatically,
        # but we check in case forward is called before model is moved)
        inv_freq = self.inv_freq
        if inv_freq.device != device:
            inv_freq = inv_freq.to(device)
        
        # Create frequency matrix: [seq_len, head_dim // 2]
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # [seq_len, head_dim // 2]
        
        # Create cos and sin: [seq_len, head_dim // 2]
        cos = freqs.cos().to(dtype)  # [seq_len, head_dim // 2]
        sin = freqs.sin().to(dtype)  # [seq_len, head_dim // 2]
        
        # Reshape for broadcasting: [1, seq_len, 1, head_dim // 2]
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        
        # Split x into two halves: each [batch, seq_len, num_heads, head_dim // 2]
        x1, x2 = x.chunk(2, dim=-1)
        
        # Apply rotary embedding: [x1*cos - x2*sin, x1*sin + x2*cos]
        rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        
        return rotated


class SwiGLU(nn.Module):
    """SwiGLU activation function: Swish(xW) ⊙ (xV) then project down"""
    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        # Gate and up projections expand to ffn_dim
        self.linear_gate = nn.Linear(dim, ffn_dim, bias=False)
        self.linear_up = nn.Linear(dim, ffn_dim, bias=False)
        # Down projection goes back to dim
        self.linear_down = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.linear_gate(x)
        up = self.linear_up(x)
        swish = F.silu(gate)
        return self.linear_down(swish * up)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with FlashAttention-2 and RoPE"""
    def __init__(self, dim: int, num_heads: int, max_seq_len: int = 8192, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        
        self.rope = RoPE(self.head_dim, max_seq_len)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # [batch, seq_len, dim]
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply RoPE
        q = self.rope(q, seq_len)
        k = self.rope(k, seq_len)
        
        # q, k, v are now [batch, seq_len, num_heads, head_dim]
        
        use_flash = (
            _FLASH_ATTN_AVAILABLE
            and x.is_cuda
            and q.dtype in (torch.float16, torch.bfloat16)
        )
        
        if use_flash:
            # FlashAttention expects [batch, seq_len, num_heads, head_dim] - no transpose needed!
            attn_output = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                softmax_scale=1.0 / math.sqrt(self.head_dim),
                causal=True
            )  # Output: [batch, seq_len, num_heads, head_dim]
            
            # Reshape to [batch, seq_len, dim]
            attn_output = attn_output.contiguous().view(batch_size, seq_len, self.dim)
        else:
            global _FLASH_ATTENTION_FALLBACK_WARNED
            if not _FLASH_ATTENTION_FALLBACK_WARNED:
                reason = (
                    f"import error: {_FLASH_ATTN_IMPORT_ERROR}"
                    if not _FLASH_ATTN_AVAILABLE
                    else "inputs not CUDA or dtype unsupported; using PyTorch SDPA"
                )
                warnings.warn(
                    f"FlashAttention-2 is unavailable ({reason}). "
                    "Falling back to torch.nn.functional.scaled_dot_product_attention. "
                    "Training will be slower on long contexts.",
                    RuntimeWarning,
                )
                _FLASH_ATTENTION_FALLBACK_WARNED = True
            
            # SDPA expects [batch, num_heads, seq_len, head_dim] - need to transpose
            q_sdpa = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
            k_sdpa = k.transpose(1, 2)
            v_sdpa = v.transpose(1, 2)
            
            attn_output = F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )  # Output: [batch, num_heads, seq_len, head_dim]
            
            # Transpose back and reshape
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        # Output projection
        output = self.o_proj(attn_output)
        return output


class TransformerBlock(nn.Module):
    """Transformer decoder block with RMSNorm, SwiGLU, and FlashAttention"""
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, max_seq_len: int = 8192, dropout: float = 0.0):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, max_seq_len, dropout)
        
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture
        # Attention
        residual = x
        x = self.attn_norm(x)
        x = self.attn(x, mask)
        x = self.dropout(x)
        x = residual + x
        
        # Feed-forward
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x
        
        return x


class DecoderOnlyTransformer(nn.Module):
    """Decoder-only Transformer model"""
    def __init__(
        self,
        vocab_size: int,
        dim: int = 2048,
        num_layers: int = 24,
        num_heads: int = 16,
        ffn_dim: int = 8192,
        max_seq_len: int = 8192,
        dropout: float = 0.0,
        tie_weights: bool = True,
        use_checkpoint: bool = False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.use_checkpoint = use_checkpoint
        
        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, dim)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, ffn_dim, max_seq_len, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = RMSNorm(dim)
        
        # Output head
        self.head = nn.Linear(dim, vocab_size, bias=False)
        
        # Tie weights if requested
        if tie_weights:
            self.head.weight = self.embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        # Embedding initialization
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        
        # Linear layer initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        input_ids: [batch_size, seq_len]
        Returns: [batch_size, seq_len, vocab_size]
        """
        # Embedding
        x = self.embedding(input_ids) * math.sqrt(self.dim)
        
        # Apply transformer blocks
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                # Pass mask as a keyword argument to activation_checkpoint
                x = activation_checkpoint(layer, x, mask, use_reentrant=False)
            else:
                x = layer(x, mask)
        
        # Final norm
        x = self.norm(x)
        
        # Output logits
        logits = self.head(x)
        
        return logits
    
    def get_num_params(self) -> int:
        """Calculate total number of parameters"""
        return sum(p.numel() for p in self.parameters())


def create_1b_model(
    vocab_size: int = 50257,
    max_seq_len: int = 8192,
    use_checkpoint: bool = False
) -> DecoderOnlyTransformer:
    """
    Create a ~1.0 billion parameter model
    
    Configuration:
    - dim: 2048
    - num_layers: 24
    - num_heads: 16
    - ffn_dim: 8192 (4x dim)
    - vocab_size: 50257 (GPT-2 tokenizer default)
    
    This should give approximately 1.0B parameters.
    """
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        dim=2048,
        num_layers=24,
        num_heads=16,
        ffn_dim=8192,
        max_seq_len=max_seq_len,
        dropout=0.1,
        tie_weights=True,
        use_checkpoint=use_checkpoint
    )
    return model

