"""
Data loading pipeline for FineWeb dataset using datatrove.
Supports streaming from HuggingFace datasets with efficient preprocessing.

Uses token buffer approach to pack multiple documents into each sequence,
maximizing compute efficiency by eliminating wasted padding tokens.
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datatrove.pipeline.readers import ParquetReader
from transformers import AutoTokenizer
from typing import Iterator, Optional, List
import math


class PackedTokenBuffer:
    """
    Token buffer that concatenates documents together to fill context windows efficiently.
    
    Instead of:
        [Doc A, Pad, Pad, Pad]
        [Doc B, Pad, Pad, Pad]
    
    We get:
        [Doc A, EOS, Doc B, EOS, Doc C (part 1)]
        [Doc C (part 2), EOS, Doc D, EOS, ...]
    """
    
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.buffer: List[int] = []
        
        # Get EOS token - critical for document separation
        self.eos_token_id = tokenizer.eos_token_id
        if self.eos_token_id is None:
            # Fallback to pad token or a special token
            self.eos_token_id = tokenizer.pad_token_id or 0
            print(f"Warning: No EOS token found, using token id {self.eos_token_id}")
    
    def add_document(self, text: str) -> None:
        """
        Tokenize a document and add it to the buffer with EOS separator.
        """
        # Tokenize WITHOUT padding or truncation - we handle that ourselves
        tokens = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,  # We don't truncate individual docs
            padding=False,
            return_tensors=None  # Return plain list
        )
        
        token_ids = tokens['input_ids']
        
        # Add document tokens to buffer
        self.buffer.extend(token_ids)
        
        # Add EOS token as document separator
        self.buffer.append(self.eos_token_id)
    
    def has_complete_sequence(self) -> bool:
        """Check if buffer has enough tokens for a complete sequence."""
        return len(self.buffer) >= self.max_length
    
    def get_sequence(self) -> Optional[tuple]:
        """
        Extract exactly max_length tokens from the buffer.
        Returns (tokens, num_real_tokens) tuple, or None if buffer doesn't have enough tokens.
        """
        if not self.has_complete_sequence():
            return None
        
        # Take exactly max_length tokens
        sequence = self.buffer[:self.max_length]
        
        # Keep the remainder in the buffer for next sequence
        self.buffer = self.buffer[self.max_length:]
        
        # All tokens are real (no padding in complete sequences)
        return (sequence, len(sequence))
    
    def get_remaining(self) -> Optional[tuple]:
        """
        Get remaining tokens (may be less than max_length).
        Returns (tokens, num_real_tokens) tuple, or None if buffer is empty.
        """
        if len(self.buffer) == 0:
            return None
        
        # Pad to max_length if needed (only for final batch)
        sequence = self.buffer.copy()
        num_real_tokens = len(sequence)
        self.buffer = []
        
        # Pad if necessary
        if len(sequence) < self.max_length:
            pad_token_id = self.tokenizer.pad_token_id or 0
            padding_needed = self.max_length - len(sequence)
            sequence.extend([pad_token_id] * padding_needed)
        
        return (sequence, num_real_tokens)
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


class StreamingFineWebDataset(IterableDataset):
    """
    Streaming dataset that packs multiple documents into each sequence.
    
    This eliminates wasted compute on padding tokens by concatenating
    documents together, separated by EOS tokens.
    
    Example output:
        [Doc1_tokens, EOS, Doc2_tokens, EOS, Doc3_start...]
        [Doc3_continued, EOS, Doc4_tokens, EOS, Doc5...]
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 8192,
        limit: Optional[int] = None
    ):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.limit = limit
        
        # Note: We don't mask by pad_token_id anymore to avoid EOS/PAD collision issues
        # Instead, we mask by position using num_real_tokens from the buffer
    
    def __iter__(self) -> Iterator[dict]:
        """
        Iterate over packed sequences.
        Each sequence is densely packed with document tokens.
        """
        # Create fresh reader and buffer for each iteration
        # ParquetReader expects -1 for no limit, not None
        data_reader = ParquetReader(self.data_path, limit=self.limit if self.limit is not None else -1)
        buffer = PackedTokenBuffer(self.tokenizer, self.max_length)
        
        # Process documents
        for doc in data_reader():
            if not hasattr(doc, 'text') or not doc.text:
                continue
            
            # Add document to buffer
            buffer.add_document(doc.text)
            
            # Yield complete sequences
            while buffer.has_complete_sequence():
                token_ids, num_real = buffer.get_sequence()
                yield self._create_training_sample(token_ids, num_real)
        
        # Optionally yield remaining tokens (will have some padding)
        # Comment this out if you want to drop incomplete final batches
        remaining = buffer.get_remaining()
        if remaining is not None:
            token_ids, num_real = remaining
            yield self._create_training_sample(token_ids, num_real)
    
    def _create_training_sample(self, token_ids: List[int], num_real_tokens: int) -> dict:
        """
        Create input_ids and labels tensors from token list.
        Labels are shifted for causal language modeling.
        
        Args:
            token_ids: List of token IDs (may include padding at end)
            num_real_tokens: Number of real tokens (rest are padding)
        """
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        
        # For causal LM: predict next token
        # labels[i] = input_ids[i+1]
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:].clone()
        labels[-1] = -100  # No target for last position
        
        # Mask padding positions by index, NOT by token ID
        # This avoids the EOS/PAD collision issue where both might be token 0
        # Position i predicts position i+1, so if position i+1 is padding,
        # we mask labels[i]. Padding starts at num_real_tokens.
        if num_real_tokens < len(token_ids):
            # Mask all positions that would predict padding
            # labels[num_real_tokens - 1] predicts token at num_real_tokens (first pad)
            labels[num_real_tokens - 1:] = -100
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }


class FineWebDataset(Dataset):
    """
    Map-style dataset that pre-caches and packs documents efficiently.
    Useful when you want to iterate multiple times over the same data.
    """
    
    def __init__(
        self,
        data_reader: ParquetReader,
        tokenizer,
        max_length: int = 8192,
        cache_size: int = 10000
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Build packed sequences from documents
        # Each entry is (token_ids, num_real_tokens)
        self.sequences = []
        self._build_packed_sequences(data_reader, cache_size)
    
    def _build_packed_sequences(self, data_reader: ParquetReader, max_docs: int):
        """Build densely packed sequences from documents."""
        buffer = PackedTokenBuffer(self.tokenizer, self.max_length)
        doc_count = 0
        
        for doc in data_reader():
            if doc_count >= max_docs:
                break
                
            if hasattr(doc, 'text') and doc.text:
                buffer.add_document(doc.text)
                doc_count += 1
                
                # Extract complete sequences (returns tuple)
                while buffer.has_complete_sequence():
                    self.sequences.append(buffer.get_sequence())
        
        # Add remaining tokens if any
        remaining = buffer.get_remaining()
        if remaining is not None:
            self.sequences.append(remaining)
        
        print(f"Built {len(self.sequences)} packed sequences from {doc_count} documents")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        token_ids, num_real_tokens = self.sequences[idx]
        
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        
        # Create labels (shifted for causal LM)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:].clone()
        labels[-1] = -100
        
        # Mask padding positions by index, NOT by token ID
        # This avoids the EOS/PAD collision issue
        if num_real_tokens < len(token_ids):
            labels[num_real_tokens - 1:] = -100
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }


class TokenLimitedDataset(IterableDataset):
    """
    Streaming dataset that stops after processing a target number of REAL tokens.
    Uses token packing for efficiency.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 8192,
        target_tokens: int = 7_700_000_000,
        document_limit: Optional[int] = None
    ):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_tokens = target_tokens
        self.document_limit = document_limit
        
        # Estimate number of sequences we'll produce
        # With packing, each sequence contains ~max_length real tokens
        self._estimated_length = max(1, target_tokens // max_length)
    
    def __len__(self):
        """Estimated length based on target tokens."""
        return self._estimated_length
    
    def __iter__(self) -> Iterator[dict]:
        """Iterate until target token count is reached."""
        # ParquetReader expects -1 for no limit, not None
        data_reader = ParquetReader(self.data_path, limit=self.document_limit if self.document_limit is not None else -1)
        buffer = PackedTokenBuffer(self.tokenizer, self.max_length)
        
        tokens_yielded = 0
        sequences_yielded = 0
        
        for doc in data_reader():
            if not hasattr(doc, 'text') or not doc.text:
                continue
            
            buffer.add_document(doc.text)
            
            while buffer.has_complete_sequence():
                if tokens_yielded >= self.target_tokens:
                    print(f"\nReached target: {tokens_yielded:,} tokens in {sequences_yielded:,} sequences")
                    return
                
                token_ids, num_real = buffer.get_sequence()
                yield self._create_training_sample(token_ids, num_real)
                
                tokens_yielded += num_real
                sequences_yielded += 1
        
        # Yield remaining if we haven't hit target yet
        if tokens_yielded < self.target_tokens:
            remaining = buffer.get_remaining()
            if remaining is not None:
                token_ids, num_real = remaining
                yield self._create_training_sample(token_ids, num_real)
                tokens_yielded += num_real
        
        print(f"\nDataset exhausted: {tokens_yielded:,} tokens in {sequences_yielded:,} sequences")
    
    def _create_training_sample(self, token_ids: List[int], num_real_tokens: int) -> dict:
        """Create input_ids and labels tensors."""
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:].clone()
        labels[-1] = -100
        
        # Mask padding positions by index, NOT by token ID
        if num_real_tokens < len(token_ids):
            labels[num_real_tokens - 1:] = -100
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }


def create_dataloader(
    data_path: str,
    tokenizer,
    batch_size: int = 4,
    max_length: int = 8192,
    num_workers: int = 0,  # IterableDataset works best with 0 workers
    pin_memory: bool = True,
    limit: Optional[int] = None,
    streaming: bool = True,
    target_tokens: Optional[int] = None,
    estimate_sample_size: int = 1000
) -> DataLoader:
    """
    Create a DataLoader for FineWeb dataset with efficient token packing.
    
    IMPORTANT: With token packing, each sequence is densely filled with real tokens,
    so you get ~100% token efficiency instead of ~30% with naive padding.
    
    Args:
        data_path: Path to FineWeb data (e.g., "hf://datasets/HuggingFaceFW/fineweb/sample/100BT")
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length (context window)
        num_workers: Number of data loading workers (0 recommended for IterableDataset)
        pin_memory: Whether to pin memory for faster GPU transfer
        limit: Limit number of documents (None for all)
        streaming: Whether to use streaming dataset (more memory efficient)
        target_tokens: Target number of REAL tokens to process
        estimate_sample_size: Number of documents to sample for estimation (unused with packing)
    
    Returns:
        DataLoader instance
    """
    if target_tokens is not None:
        # With packing, we don't need to estimate document count
        # The dataset will pack documents efficiently and stop at target_tokens
        dataset = TokenLimitedDataset(
            data_path, 
            tokenizer, 
            max_length, 
            target_tokens,
            document_limit=limit
        )
        print(f"Created TokenLimitedDataset targeting {target_tokens / 1e9:.2f}B tokens")
        print(f"With max_length={max_length}, expecting ~{target_tokens // max_length:,} sequences")
    elif streaming:
        dataset = StreamingFineWebDataset(data_path, tokenizer, max_length, limit)
        print(f"Created StreamingFineWebDataset with efficient token packing")
    else:
        # ParquetReader expects -1 for no limit, not None
        data_reader = ParquetReader(data_path, limit=limit if limit is not None else -1)
        dataset = FineWebDataset(data_reader, tokenizer, max_length)
    
    # For IterableDataset, num_workers > 0 can cause issues
    # Each worker would create its own iterator, potentially duplicating data
    if isinstance(dataset, IterableDataset) and num_workers > 0:
        print(f"Warning: Using num_workers={num_workers} with IterableDataset may cause issues. Consider num_workers=0.")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # IterableDataset doesn't support shuffling
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch
    )
    
    return dataloader


def estimate_tokens_per_document(
    data_path: str,
    tokenizer,
    sample_size: int = 1000,
    max_length: int = 8192
) -> float:
    """
    Estimate average tokens per document by sampling.
    
    Note: With token packing, this is mainly useful for progress estimation,
    not for calculating padding waste (which is now ~0%).
    
    Args:
        data_path: Path to FineWeb data
        tokenizer: Tokenizer to use
        sample_size: Number of documents to sample
        max_length: Maximum sequence length (for reference, not used for truncation here)
    
    Returns:
        Average tokens per document
    """
    data_reader = ParquetReader(data_path, limit=sample_size)
    
    total_tokens = 0
    count = 0
    
    for doc in data_reader():
        if hasattr(doc, 'text') and doc.text:
            tokens = tokenizer(
                doc.text,
                add_special_tokens=True,
                truncation=False,
                padding=False,
                return_tensors=None
            )
            token_count = len(tokens['input_ids'])
            total_tokens += token_count
            count += 1
    
    if count == 0:
        return 0.0
    
    avg_tokens_per_doc = total_tokens / count
    return avg_tokens_per_doc


def estimate_document_count_for_tokens(
    data_path: str,
    tokenizer,
    target_tokens: int,
    sample_size: int = 1000,
    max_length: int = 8192,
    safety_margin: float = 1.05
) -> int:
    """
    Estimate how many documents are needed to reach target token count.
    
    With token packing, this gives a more accurate estimate since we
    use actual token counts without padding.
    
    Args:
        data_path: Path to FineWeb data
        tokenizer: Tokenizer to use
        target_tokens: Target number of tokens
        sample_size: Number of documents to sample for estimation
        max_length: Maximum sequence length
        safety_margin: Multiplier to add safety margin (default 5%)
    
    Returns:
        Estimated number of documents needed
    """
    print(f"Estimating document count for {target_tokens / 1e9:.2f}B tokens...")
    print(f"Sampling {sample_size} documents...")
    
    avg_tokens = estimate_tokens_per_document(data_path, tokenizer, sample_size, max_length)
    
    if avg_tokens == 0:
        raise ValueError("No valid documents found in dataset")
    
    print(f"Average tokens per document: {avg_tokens:.0f}")
    
    # With token packing, we need documents totaling target_tokens
    # Add EOS token per document (~1 token overhead per doc)
    effective_tokens_per_doc = avg_tokens + 1  # +1 for EOS separator
    
    docs_needed = int((target_tokens / effective_tokens_per_doc) * safety_margin)
    
    print(f"Estimated documents needed: {docs_needed:,} (with {(safety_margin-1)*100:.0f}% safety margin)")
    print(f"With packing: ~{target_tokens // max_length:,} sequences of {max_length} tokens each")
    
    return docs_needed


# Utility function to verify packing efficiency
def verify_packing_efficiency(
    data_path: str,
    tokenizer,
    max_length: int = 4096,
    num_samples: int = 100
) -> dict:
    """
    Verify the token packing efficiency by comparing packed vs unpacked approaches.
    
    Returns statistics showing the improvement from packing.
    """
    data_reader = ParquetReader(data_path, limit=num_samples * 10)  # Read extra docs
    buffer = PackedTokenBuffer(tokenizer, max_length)
    
    total_doc_tokens = 0
    doc_count = 0
    complete_sequences = 0
    remaining_tokens = 0
    
    for doc in data_reader():
        if doc_count >= num_samples:
            break
        if hasattr(doc, 'text') and doc.text:
            # Count tokens in this document
            tokens = tokenizer(doc.text, add_special_tokens=True, return_tensors=None)
            doc_tokens = len(tokens['input_ids'])
            total_doc_tokens += doc_tokens
            doc_count += 1
            
            # Add to buffer
            buffer.add_document(doc.text)
            while buffer.has_complete_sequence():
                _ = buffer.get_sequence()  # Returns tuple, we just need to count
                complete_sequences += 1
    
    # Track remaining partial sequence separately
    remaining_tokens = len(buffer)
    packed_sequences = complete_sequences + (1 if remaining_tokens > 0 else 0)
    
    # Calculate stats
    avg_tokens_per_doc = total_doc_tokens / doc_count if doc_count > 0 else 0
    
    # Without packing: each doc gets max_length, so lots of padding
    unpacked_total_tokens = doc_count * max_length
    unpacked_real_ratio = total_doc_tokens / unpacked_total_tokens if unpacked_total_tokens > 0 else 0
    
    # With packing: complete sequences are full, partial sequence has actual remaining tokens
    # This gives accurate efficiency calculation
    packed_total_tokens = (complete_sequences * max_length) + remaining_tokens
    packed_real_ratio = total_doc_tokens / packed_total_tokens if packed_total_tokens > 0 else 0
    
    stats = {
        'documents_sampled': doc_count,
        'avg_tokens_per_doc': avg_tokens_per_doc,
        'max_length': max_length,
        'unpacked': {
            'sequences': doc_count,
            'total_tokens': unpacked_total_tokens,
            'real_tokens': total_doc_tokens,
            'efficiency': unpacked_real_ratio * 100,
            'wasted_on_padding': (1 - unpacked_real_ratio) * 100
        },
        'packed': {
            'sequences': packed_sequences,
            'total_tokens': packed_total_tokens,
            'real_tokens': total_doc_tokens,
            'efficiency': packed_real_ratio * 100,
            'wasted_on_padding': (1 - packed_real_ratio) * 100
        },
        'improvement_factor': packed_real_ratio / unpacked_real_ratio if unpacked_real_ratio > 0 else 0
    }
    
    return stats


if __name__ == "__main__":
    # Test the packing efficiency
    from transformers import AutoTokenizer
    
    print("Testing Token Packing Efficiency")
    print("=" * 60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test with sample data
    data_path = "hf://datasets/HuggingFaceFW/fineweb/sample/10BT"
    
    print("\nVerifying packing efficiency...")
    stats = verify_packing_efficiency(data_path, tokenizer, max_length=4096, num_samples=100)
    
    print(f"\n{'Metric':<35} {'Unpacked':<15} {'Packed':<15}")
    print("-" * 65)
    print(f"{'Documents sampled':<35} {stats['documents_sampled']:<15}")
    print(f"{'Avg tokens/doc':<35} {stats['avg_tokens_per_doc']:<15.0f}")
    print(f"{'Max sequence length':<35} {stats['max_length']:<15}")
    print("-" * 65)
    print(f"{'Sequences produced':<35} {stats['unpacked']['sequences']:<15} {stats['packed']['sequences']:<15}")
    print(f"{'Total tokens (incl. padding)':<35} {stats['unpacked']['total_tokens']:<15,} {stats['packed']['total_tokens']:<15,}")
    print(f"{'Real tokens':<35} {stats['unpacked']['real_tokens']:<15,} {stats['packed']['real_tokens']:<15,}")
    print(f"{'Efficiency %':<35} {stats['unpacked']['efficiency']:<15.1f} {stats['packed']['efficiency']:<15.1f}")
    print(f"{'Wasted on padding %':<35} {stats['unpacked']['wasted_on_padding']:<15.1f} {stats['packed']['wasted_on_padding']:<15.1f}")
    print("-" * 65)
    print(f"{'Improvement factor':<35} {stats['improvement_factor']:.2f}x")
