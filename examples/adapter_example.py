#!/usr/bin/env python3
"""Example demonstrating medical model adapter usage with Nano vLLM.

This example shows how to:
1. Load a medical language model with automatic adapter detection
2. Configure adapter-specific settings
3. Use the adapted model for medical text generation
"""
import os
import sys

# Add the parent directory to the path so we can import medvllm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medvllm import LLM, AdapterManager, SamplingParams
from medvllm.config import Config


def biobert_specific_example():
    """Example of BioBERT-specific features."""
    print("\n4. BioBERT-Specific Features")
    print("-" * 30)
    
    try:
        from medvllm.models.adapter_manager import AdapterManager
        from medvllm.models.adapter import BioBERTAdapter
        import torch
        import torch.nn as nn
        
        # Create a mock BioBERT model for demonstration
        class MockBioBERTModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type('Config', (), {
                    'num_hidden_layers': 12,
                    'num_attention_heads': 12, 
                    'hidden_size': 768,
                    '_name_or_path': 'dmis-lab/biobert-v1.1'
                })()
                
            def forward(self, input_ids, **kwargs):
                return torch.randn(input_ids.shape[0], input_ids.shape[1], 768)
        
        model = MockBioBERTModel()
        config = {"use_kv_cache": True}
        
        # Create BioBERT adapter
        print("Creating BioBERT adapter with biomedical features...")
        adapter = BioBERTAdapter(model, config)
        
        # Example medical text processing
        medical_texts = [
            "Patient has q.d. medication for cardio-vascular issues.",
            "Diagnosed with gastro-enteritis and myocardial infarction.", 
            "Treatment includes i.v. antibiotics for pneumonia."
        ]
        
        print("\nProcessing biomedical text:")
        for i, text in enumerate(medical_texts, 1):
            processed = adapter._preserve_medical_terms(text)
            print(f"  {i}. Original: {text}")
            print(f"     Processed: {processed}")
        
        print("\n✅ BioBERT adapter features:")
        print("   - Biomedical vocabulary handling")
        print("   - Medical term preservation")
        print("   - Weight conversion utilities")
        print("   - Embedding extension for new tokens")
        
    except Exception as e:
        print(f"BioBERT example failed: {e}")
        print("Note: This requires proper model setup")


def main():
    """Demonstrate adapter usage."""
    print("=== Medical Model Adapter Example ===\n")

    # Example 1: Automatic adapter detection
    print("1. Creating LLM with automatic adapter detection...")
    try:
        # This would work with an actual BioBERT model path
        # For demo purposes, we'll show the configuration
        model_path = "dmis-lab/biobert-v1.1"  # Example BioBERT model

        config = Config(
            model=model_path,
            use_medical_adapter=True,  # Enable adapter
            adapter_type=None,  # Auto-detect
            use_cuda_graphs=False,
            max_model_len=512,
            max_num_seqs=4,
        )

        print(f"Configuration:")
        print(f"  - Model: {config.model}")
        print(f"  - Use Medical Adapter: {config.use_medical_adapter}")
        print(f"  - Adapter Type: {config.adapter_type or 'Auto-detect'}")
        print(f"  - CUDA Graphs: {config.use_cuda_graphs}")

        # Note: This would actually load the model if the path exists
        # llm = LLM(model=model_path, **config.__dict__)

    except Exception as e:
        print(f"Note: Model loading skipped (model not available): {e}")

    print("\n" + "=" * 50 + "\n")

    # Example 2: Manual adapter type specification
    print("2. Manual adapter configuration...")

    adapter_config = {
        "use_kv_cache": True,
        "max_batch_size": 8,
        "max_seq_length": 512,
        "vocab_size": 30522,
        "hidden_size": 768,
    }

    config2 = Config(
        model="emilyalsentzer/Bio_ClinicalBERT",  # Example ClinicalBERT
        use_medical_adapter=True,
        adapter_type="clinicalbert",  # Explicitly specify
        adapter_config=adapter_config,
        use_cuda_graphs=False,
        max_model_len=512,
    )

    print(f"Configuration:")
    print(f"  - Model: {config2.model}")
    print(f"  - Adapter Type: {config2.adapter_type}")
    print(f"  - Custom Config: {config2.adapter_config}")

    print("\n" + "=" * 50 + "\n")

    # Example 3: Demonstrate adapter manager directly
    print("3. Using AdapterManager directly...")

    # Show model type detection
    test_models = [
        "dmis-lab/biobert-v1.1",
        "emilyalsentzer/Bio_ClinicalBERT",
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
    ]

    for model_name in test_models:
        detected_type = AdapterManager.detect_model_type(model_name)
        print(f"  - {model_name}")
        print(f"    Detected type: {detected_type}")

    print("\n" + "=" * 50 + "\n")

    # Example 4: Show default configurations
    print("4. Default adapter configurations...")

    for adapter_type in ["biobert", "clinicalbert", "pubmedbert", "bluebert"]:
        default_config = AdapterManager.get_default_adapter_config(adapter_type)
        print(f"  - {adapter_type.upper()}:")
        for key, value in default_config.items():
            print(f"    {key}: {value}")
        print()

    print("=" * 50)
    print("\nAdapter interface architecture is now fully integrated!")
    print("\nKey features implemented:")
    print("✓ Abstract base class (MedicalModelAdapter)")
    print("✓ Concrete adapter implementations (BioBERT, ClinicalBERT)")
    print("✓ Factory pattern (create_medical_adapter)")
    print("✓ Automatic model type detection")
    print("✓ Configuration management")
    print("✓ Integration with Nano vLLM engine")
    print("✓ KV caching and CUDA graphs support")
    print("✓ Standardized input/output formats")

    print("\nTo use with real models:")
    print("1. Ensure you have the medical model downloaded")
    print("2. Create LLM instance with adapter configuration")
    print("3. Use generate() method as normal - adapter is transparent")


if __name__ == "__main__":
    main()
