"""Performance tests for medical model adapters and KV cache."""

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import gc
import time
from typing import Dict, List, Tuple

import numpy as np
import psutil
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from medvllm.models.adapters import BioBERTAdapter, ClinicalBERTAdapter
from medvllm.utils.datasets import load_medical_dataset

# Test configuration
BATCH_SIZES = [1, 4, 8]
SEQUENCE_LENGTHS = [128, 256, 512]
MODEL_TYPES = ["biobert", "clinicalbert"]


class MedicalTextDataset(Dataset):
    """Dummy medical text dataset for benchmarking."""
    
    def __init__(self, num_samples=100, max_length=512):
        self.samples = [
            "Patient presents with fever and cough for 3 days. " + 
            "No known allergies. Vitals stable. " * (max_length // 100)
            for _ in range(num_samples)
        ]
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage in MB."""
    process = psutil.Process()
    mem_info = process.memory_info()
    torch.cuda.synchronize()
    return {
        'rss_mb': mem_info.rss / 1024 / 1024,
        'vms_mb': mem_info.vms / 1024 / 1024,
        'gpu_mb': torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    }


class TestMedicalModelPerformance:
    """Performance test suite for medical model adapters."""
    
    @pytest.fixture(params=MODEL_TYPES)
    def model_adapter(self, request):
        """Initialize model adapter with specified type."""
        model_type = request.param
        if model_type == "biobert":
            return BioBERTAdapter.from_pretrained("monologg/biobert_v1.1_pubmed")
        else:  # clinicalbert
            return ClinicalBERTAdapter.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("seq_len", SEQUENCE_LENGTHS)
    def test_inference_throughput(self, model_adapter, batch_size, seq_len):
        """Test inference throughput for different batch sizes and sequence lengths."""
        # Prepare test data
        dataset = MedicalTextDataset(num_samples=10, max_length=seq_len)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        # Warmup
        for batch in dataloader:
            _ = model_adapter(batch)
            break
            
        # Benchmark
        start_time = time.time()
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                outputs = model_adapter(batch)
                total_tokens += sum(len(text) for text in batch)  # Approximate token count
        
        elapsed = time.time() - start_time
        tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
        
        print(f"\nThroughput - Batch: {batch_size}, SeqLen: {seq_len}, "
              f"Tokens/sec: {tokens_per_sec:.2f}")
    
    @pytest.mark.parametrize("seq_len", SEQUENCE_LENGTHS)
    def test_memory_usage(self, model_adapter, seq_len):
        """Test memory usage for different sequence lengths."""
        # Initial memory
        start_mem = get_memory_usage()
        
        # Run inference
        text = "Patient with " + "fever and cough. " * (seq_len // 10)
        with torch.no_grad():
            _ = model_adapter([text])
        
        # Get peak memory
        torch.cuda.synchronize()
        peak_mem = get_memory_usage()
        
        # Calculate memory delta
        mem_delta = {k: peak_mem[k] - start_mem[k] for k in peak_mem}
        
        print(f"\nMemory Usage - SeqLen: {seq_len}, "
              f"GPU Memory: {mem_delta['gpu_mb']:.2f}MB")
    
    def test_kv_cache_impact(self, model_adapter):
        """Test the impact of KV caching on performance."""
        # Disable cache
        model_adapter.disable_cache()
        text = "Patient with fever and cough. " * 50
        
        # Without cache
        start = time.time()
        for _ in range(10):
            _ = model_adapter([text])
        no_cache_time = time.time() - start
        
        # With cache
        model_adapter.enable_cache()
        start = time.time()
        for _ in range(10):
            _ = model_adapter([text])
        cache_time = time.time() - start
        
        speedup = (no_cache_time - cache_time) / no_cache_time * 100
        print(f"\nKV Cache Impact - Speedup: {speedup:.1f}%")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
