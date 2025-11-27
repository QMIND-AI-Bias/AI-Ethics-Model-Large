"""
Data loading pipeline for FineWeb dataset using datatrove.
Supports streaming from HuggingFace datasets with efficient preprocessing.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datatrove.pipeline.readers import ParquetReader
from transformers import AutoTokenizer
from typing import Iterator, Optional
import math


class FineWebDataset(Dataset):
    """Dataset for FineWeb documents"""
    def __init__(
        self,
        data_reader: ParquetReader,
        tokenizer,
        max_length: int = 8192,
        cache_size: int = 10000
    ):
        self.data_reader = data_reader
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_size = cache_size
        
        # Cache for documents
        self.cache = []
        self._fill_cache()
    
    def _fill_cache(self):
        """Fill the cache with documents"""
        self.cache = []
        for doc in self.data_reader():
            if hasattr(doc, 'text') and doc.text:
                self.cache.append(doc.text)
                if len(self.cache) >= self.cache_size:
                    break
    
    def __len__(self):
        return len(self.cache)
    
    def __getitem__(self, idx):
        if idx >= len(self.cache):
            # Refill cache if needed
            self._fill_cache()
            idx = idx % len(self.cache)
        
        text = self.cache[idx]
        
        # Tokenize
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = tokens['input_ids'].squeeze(0)
        
        # For causal language modeling, labels should be shifted by one position
        # Model predicts token[i+1] given tokens[0:i+1], so labels[i] = input_ids[i+1]
        labels = input_ids.clone()
        # Shift labels: labels[i] should be input_ids[i+1]
        labels[:-1] = input_ids[1:].clone()
        # Last token has no next token, so set to ignore_index
        labels[-1] = -100  # -100 is the default ignore_index for CrossEntropyLoss
        # Also ignore padding tokens in labels
        pad_token_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None else -100
        labels[input_ids == pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }


class StreamingFineWebDataset(Dataset):
    """
    Streaming dataset that doesn't cache all data in memory.
    More memory efficient for large datasets.
    """
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 8192,
        limit: Optional[int] = None
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.limit = limit
        
        # Create iterator
        self.data_reader = ParquetReader(data_path, limit=limit)
        self.iterator = iter(self.data_reader())
        self.current_doc = None
        self._load_next_doc()
    
    def _load_next_doc(self):
        """Load next document from iterator"""
        try:
            doc = next(self.iterator)
            if hasattr(doc, 'text') and doc.text:
                self.current_doc = doc.text
            else:
                self._load_next_doc()
        except StopIteration:
            self.current_doc = None
    
    def __len__(self):
        # For streaming datasets, we don't know the exact length
        # Return a large number to allow iteration
        return 10**9  # Large number
    
    def __getitem__(self, idx):
        if self.current_doc is None:
            self._load_next_doc()
        
        if self.current_doc is None:
            # No more documents, return padding
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'labels': torch.zeros(self.max_length, dtype=torch.long)
            }
        
        text = self.current_doc
        self._load_next_doc()  # Prepare next document
        
        # Tokenize
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = tokens['input_ids'].squeeze(0)
        
        # For causal language modeling, labels should be shifted by one position
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:].clone()
        labels[-1] = -100  # -100 is the default ignore_index for CrossEntropyLoss
        # Also ignore padding tokens in labels
        pad_token_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None else -100
        labels[input_ids == pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }


def create_dataloader(
    data_path: str,
    tokenizer,
    batch_size: int = 4,
    max_length: int = 8192,
    num_workers: int = 4,
    pin_memory: bool = True,
    limit: Optional[int] = None,
    streaming: bool = True,
    target_tokens: Optional[int] = None,
    estimate_sample_size: int = 1000
) -> DataLoader:
    """
    Create a DataLoader for FineWeb dataset
    
    Args:
        data_path: Path to FineWeb data (e.g., "hf://datasets/HuggingFaceFW/fineweb/sample/100BT")
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
        limit: Limit number of documents (None for all, overrides target_tokens if set)
        streaming: Whether to use streaming dataset (more memory efficient)
        target_tokens: Target number of tokens to process (estimates document count needed)
        estimate_sample_size: Number of documents to sample for token estimation
    
    Returns:
        DataLoader instance
    """
    document_limit = limit
    
    if target_tokens is not None and document_limit is None:
        # Estimate document count needed for target tokens
        # This works correctly with multiple workers since we limit by document count
        document_limit = estimate_document_count_for_tokens(
            data_path=data_path,
            tokenizer=tokenizer,
            target_tokens=target_tokens,
            sample_size=estimate_sample_size,
            max_length=max_length
        )
    
    if target_tokens is not None:
        # Use token-limited dataset with pre-calculated document limit
        # This ensures accurate token counting with multiple workers
        dataset = TokenLimitedDataset(
            data_path, 
            tokenizer, 
            max_length, 
            target_tokens, 
            document_limit=document_limit
        )
    else:
        data_reader = ParquetReader(data_path, limit=document_limit)
        if streaming:
            dataset = StreamingFineWebDataset(data_path, tokenizer, max_length, document_limit)
        else:
            dataset = FineWebDataset(data_reader, tokenizer, max_length)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Streaming datasets are already shuffled by nature
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
    Accounts for truncation and padding.
    
    Args:
        data_path: Path to FineWeb data
        tokenizer: Tokenizer to use
        sample_size: Number of documents to sample
        max_length: Maximum sequence length (for truncation)
    
    Returns:
        Average tokens per document (after truncation to max_length)
    """
    data_reader = ParquetReader(data_path, limit=sample_size)
    
    total_tokens = 0
    count = 0
    
    for doc in data_reader():
        if hasattr(doc, 'text') and doc.text:
            # Tokenize with same settings as dataset
            tokens = tokenizer(
                doc.text,
                max_length=max_length,
                truncation=True,
                padding=False,
                return_tensors='pt'
            )
            # Count actual tokens (not padding)
            token_count = tokens['input_ids'].shape[1]
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
    Uses sampling to estimate average tokens per document.
    
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
    
    print(f"Average tokens per document (after truncation): {avg_tokens:.0f}")
    
    # Calculate documents needed with safety margin
    docs_needed = int((target_tokens / avg_tokens) * safety_margin)
    
    print(f"Estimated documents needed: {docs_needed:,} (with {safety_margin*100:.0f}% safety margin)")
    
    return docs_needed


class TokenLimitedDataset(Dataset):
    """
    Dataset that stops after processing a target number of tokens.
    Uses pre-calculated document count for accurate multi-worker support.
    """
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 8192,
        target_tokens: int = 7_700_000_000,
        document_limit: Optional[int] = None
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_tokens = target_tokens
        self.document_limit = document_limit
        
        # If document_limit is provided, use it directly
        # Otherwise, this dataset will use the limit from ParquetReader
        self.data_reader = ParquetReader(data_path, limit=document_limit)
        self.iterator = iter(self.data_reader())
        self.current_doc = None
        self._load_next_doc()
    
    def _load_next_doc(self):
        """Load next document from iterator"""
        try:
            doc = next(self.iterator)
            if hasattr(doc, 'text') and doc.text:
                self.current_doc = doc.text
            else:
                self._load_next_doc()
        except StopIteration:
            self.current_doc = None
    
    def __len__(self):
        # Estimate based on target tokens or document limit
        if self.document_limit:
            return self.document_limit
        return int(self.target_tokens / self.max_length)
    
    def __getitem__(self, idx):
        if self.current_doc is None:
            self._load_next_doc()
        
        if self.current_doc is None:
            # No more documents, return padding
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'labels': torch.zeros(self.max_length, dtype=torch.long)
            }
        
        text = self.current_doc
        self._load_next_doc()
        
        # Tokenize
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = tokens['input_ids'].squeeze(0)
        
        # For causal language modeling, labels should be shifted by one position
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:].clone()
        labels[-1] = -100  # -100 is the default ignore_index for CrossEntropyLoss
        # Also ignore padding tokens in labels
        pad_token_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None else -100
        labels[input_ids == pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }

