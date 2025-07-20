#!/usr/bin/env python3
"""Test script for modular adapter architecture."""

import os
import sys
from typing import Any, Dict

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_modular_imports():
    """Test that all modular adapter components can be imported."""
    print("🧪 Testing modular adapter imports...")

    try:
        from medvllm.models.adapters.biobert import BioBERTAdapter

        print("✅ BioBERT adapter imported successfully")
    except Exception as e:
        print(f"❌ BioBERT adapter import failed: {e}")
        return False

    try:
        from medvllm.models.adapters.clinicalbert import ClinicalBERTAdapter

        print("✅ ClinicalBERT adapter imported successfully")
    except Exception as e:
        print(f"❌ ClinicalBERT adapter import failed: {e}")
        return False

    try:
        from medvllm.models.adapter import create_medical_adapter

        print("✅ Factory function imported successfully")
    except Exception as e:
        print(f"❌ Factory function import failed: {e}")
        return False

    return True


def test_factory_function():
    """Test the factory function for creating adapters."""
    print("\n🧪 Testing factory function...")

    try:
        from medvllm.models.adapter import create_medical_adapter

        # Create a dummy model for testing
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type(
                    "Config",
                    (),
                    {
                        "num_hidden_layers": 12,
                        "num_attention_heads": 12,
                        "hidden_size": 768,
                        "_name_or_path": "test-model",
                    },
                )()

        dummy_model = DummyModel()
        config = {"skip_tokenizer_setup": True}

        # Test BioBERT adapter creation
        biobert_adapter = create_medical_adapter("biobert", dummy_model, config)
        print(f"✅ BioBERT adapter created: {type(biobert_adapter).__name__}")

        # Test ClinicalBERT adapter creation
        clinical_adapter = create_medical_adapter("clinicalbert", dummy_model, config)
        print(f"✅ ClinicalBERT adapter created: {type(clinical_adapter).__name__}")

        # Test error handling for unsupported model type
        try:
            unsupported_adapter = create_medical_adapter(
                "unsupported", dummy_model, config
            )
            print("❌ Should have raised ValueError for unsupported model type")
            return False
        except ValueError as e:
            print(f"✅ Correctly raised ValueError for unsupported model: {e}")

        return True

    except Exception as e:
        print(f"❌ Factory function test failed: {e}")
        return False


def test_adapter_functionality():
    """Test basic adapter functionality."""
    print("\n🧪 Testing adapter functionality...")

    try:
        from medvllm.models.adapters.biobert import BioBERTAdapter
        from medvllm.models.adapters.clinicalbert import ClinicalBERTAdapter

        # Create dummy model
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type(
                    "Config",
                    (),
                    {
                        "num_hidden_layers": 12,
                        "num_attention_heads": 12,
                        "hidden_size": 768,
                        "_name_or_path": "test-model",
                    },
                )()

            def forward(self, input_ids, **kwargs):
                return torch.randn(1, input_ids.size(1), 768)

        dummy_model = DummyModel()
        config = {
            "skip_tokenizer_setup": True,
            "tensor_parallel_size": 1,
            "memory_efficient": True,
            "enable_mixed_precision": False,
        }

        # Test BioBERT adapter
        biobert_adapter = BioBERTAdapter(dummy_model, config)
        print(
            f"✅ BioBERT adapter initialized with model type: {biobert_adapter.model_type}"
        )

        # Test ClinicalBERT adapter
        clinical_adapter = ClinicalBERTAdapter(dummy_model, config)
        print(
            f"✅ ClinicalBERT adapter initialized with model type: {clinical_adapter.model_type}"
        )

        # Test memory stats
        biobert_stats = biobert_adapter.get_memory_stats()
        print(
            f"✅ BioBERT memory stats: {biobert_stats['total_parameters']} parameters"
        )

        clinical_stats = clinical_adapter.get_memory_stats()
        print(
            f"✅ ClinicalBERT memory stats: {clinical_stats['total_parameters']} parameters"
        )

        return True

    except Exception as e:
        print(f"❌ Adapter functionality test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Testing Modular Adapter Architecture")
    print("=" * 50)

    tests = [test_modular_imports, test_factory_function, test_adapter_functionality]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Modular adapter architecture is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    exit(main())
