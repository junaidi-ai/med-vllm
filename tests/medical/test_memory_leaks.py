"""Stress tests for memory leaks in medical model inference."""

import gc
import os
import psutil
import time
import unittest
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from medvllm.models import MedicalModel


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    return {
        'rss_mb': mem_info.rss / (1024 * 1024),  # Resident Set Size
        'vms_mb': mem_info.vms / (1024 * 1024),  # Virtual Memory Size
    }


class MemoryLeakTest(unittest.TestCase):
    """Test for memory leaks in medical model inference."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cls.model_name = 'biobert'  # Can be parameterized for different models
        
        # Initialize the model
        cls.model = MedicalModel.from_pretrained(
            f"medical-{cls.model_name}",
            device=cls.device
        )
        
        # Generate test data
        cls.test_data = cls._generate_test_data(num_samples=1000)
    
    @staticmethod
    def _generate_test_data(num_samples: int = 1000) -> List[str]:
        """Generate test data for memory leak testing."""
        base_texts = [
            "Patient presents with {} and {}. Vital signs stable.",
            "History of {} with current complaint of {}.",
            "Assessment: {} and {}.",
            "Plan: Monitor {} and treat {}.",
        ]
        
        symptoms = [
            "chest pain", "headache", "shortness of breath", "nausea",
            "dizziness", "fatigue", "fever", "cough", "abdominal pain"
        ]
        
        # Generate variations of the base texts with different symptoms
        test_data = []
        for i in range(num_samples):
            base = base_texts[i % len(base_texts)]
            symptom1 = symptoms[i % len(symptoms)]
            symptom2 = symptoms[(i + 1) % len(symptoms)]
            test_data.append(base.format(symptom1, symptom2))
        
        return test_data
    
    def _run_inference_batch(self, batch_size: int = 32, num_batches: int = 100):
        """Run inference in batches and track memory usage."""
        memory_samples = []
        
        for i in tqdm(range(num_batches), desc="Running inference batches"):
            # Get a batch of data
            start_idx = (i * batch_size) % len(self.test_data)
            batch = self.test_data[start_idx:start_idx + batch_size]
            
            # Run inference
            with torch.no_grad():
                _ = self.model.predict(batch)
            
            # Measure memory usage every 10 batches
            if i % 10 == 0:
                memory_samples.append(get_memory_usage())
                
                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return memory_samples
    
    def test_memory_leak_inference(self):
        """Test for memory leaks during repeated inference."""
        # Initial memory usage
        initial_memory = get_memory_usage()
        
        # Run multiple batches of inference
        memory_samples = self._run_inference_batch(
            batch_size=16,
            num_batches=100
        )
        
        # Get final memory usage
        final_memory = get_memory_usage()
        
        # Calculate memory growth
        rss_growth = final_memory['rss_mb'] - initial_memory['rss_mb']
        vms_growth = final_memory['vms_mb'] - initial_memory['vms_mb']
        
        print(f"\nMemory usage after stress test:")
        print(f"  RSS: {initial_memory['rss_mb']:.2f}MB -> {final_memory['rss_mb']:.2f}MB "
              f"(Δ{rss_growth:+.2f}MB)")
        print(f"  VMS: {initial_memory['vms_mb']:.2f}MB -> {final_memory['vms_mb']:.2f}MB "
              f"(Δ{vms_growth:+.2f}MB)")
        
        # Check for excessive memory growth (more than 100MB)
        self.assertLess(rss_growth, 100, 
                       f"Excessive RSS memory growth: {rss_growth:.2f}MB")
        self.assertLess(vms_growth, 100,
                       f"Excessive VMS memory growth: {vms_growth:.2f}MB")
    
    def test_long_running_inference(self):
        """Test for memory leaks in long-running inference."""
        # This test runs a smaller number of larger batches to test for
        # different types of memory issues
        initial_memory = get_memory_usage()
        
        # Run fewer, larger batches
        memory_samples = self._run_inference_batch(
            batch_size=64,
            num_batches=50
        )
        
        # Get final memory usage
        final_memory = get_memory_usage()
        
        # Calculate memory growth
        rss_growth = final_memory['rss_mb'] - initial_memory['rss_mb']
        
        print(f"\nMemory usage after long-running test:")
        print(f"  RSS: {initial_memory['rss_mb']:.2f}MB -> {final_memory['rss_mb']:.2f}MB "
              f"(Δ{rss_growth:+.2f}MB)")
        
        # Check for excessive memory growth (more than 50MB)
        self.assertLess(rss_growth, 50,
                       f"Excessive RSS memory growth in long-running test: {rss_growth:.2f}MB")


if __name__ == "__main__":
    unittest.main()
