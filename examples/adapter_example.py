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
        import torch
        import torch.nn as nn

        from medvllm.models.adapter import BioBERTAdapter
        from medvllm.models.adapter_manager import AdapterManager

        # Create a mock BioBERT model for demonstration
        class MockBioBERTModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type(
                    "Config",
                    (),
                    {
                        "num_hidden_layers": 12,
                        "num_attention_heads": 12,
                        "hidden_size": 768,
                        "_name_or_path": "dmis-lab/biobert-v1.1",
                    },
                )()

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
            "Treatment includes i.v. antibiotics for pneumonia.",
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


def clinicalbert_specific_example():
    """Example of ClinicalBERT-specific features."""
    print("\n5. ClinicalBERT-Specific Features")
    print("-" * 35)

    try:
        import torch
        import torch.nn as nn

        from medvllm.models.adapter import ClinicalBERTAdapter
        from medvllm.models.adapter_manager import AdapterManager

        # Create a mock ClinicalBERT model for demonstration
        class MockClinicalBERTModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type(
                    "Config",
                    (),
                    {
                        "num_hidden_layers": 12,
                        "num_attention_heads": 12,
                        "hidden_size": 768,
                        "_name_or_path": "emilyalsentzer/Bio_ClinicalBERT",
                    },
                )()

            def forward(self, input_ids, **kwargs):
                return torch.randn(input_ids.shape[0], input_ids.shape[1], 768)

        model = MockClinicalBERTModel()
        config = {"use_kv_cache": True}

        # Create ClinicalBERT adapter
        print("Creating ClinicalBERT adapter with clinical features...")
        adapter = ClinicalBERTAdapter(model, config)

        # Example clinical texts
        clinical_texts = [
            "Patient admitted to ICU with COPD exacerbation, BP 140/90.",
            "CHF patient underwent CABG, currently stable in CCU.",
            "EKG shows MI, troponins elevated, transferred to OR.",
        ]

        print("\nProcessing clinical text:")
        for i, text in enumerate(clinical_texts, 1):
            processed = adapter._preserve_clinical_context(text)
            print(f"  {i}. Original: {text}")
            print(f"     Processed: {processed}")

        # Example clinical note processing
        print("\nClinical note contextualization:")
        note_examples = [
            ("Patient stable, vitals WNL, continue medications.", "progress"),
            ("Patient discharged home in stable condition.", "discharge"),
            ("65 y/o male admitted with chest pain, rule out MI.", "admission"),
        ]

        for note_text, note_type in note_examples:
            processed_note = adapter.process_clinical_note(note_text, note_type)
            print(f"  {note_type.title()}: {processed_note}")

        print("\n✅ ClinicalBERT adapter features:")
        print("   - Clinical terminology handling (COPD, CHF, MI, etc.)")
        print("   - Vital signs preservation (BP, HR, measurements)")
        print("   - Clinical note contextualization")
        print("   - Weight conversion utilities")
        print("   - Clinical vocabulary extension")
        print("   - Tensor parallelism support")
        print("   - CUDA memory optimization")

    except Exception as e:
        print(f"ClinicalBERT example failed: {e}")
        print("Note: This requires proper model setup")


def tensor_parallelism_example():
    """Example of tensor parallelism and CUDA optimization features."""
    print("\n6. Tensor Parallelism & CUDA Optimization")
    print("-" * 45)

    try:
        import torch
        import torch.nn as nn

        from medvllm.models.adapter import BioBERTAdapter
        from medvllm.models.adapter_manager import AdapterManager

        # Create a mock model for demonstration
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type(
                    "Config",
                    (),
                    {
                        "num_hidden_layers": 12,
                        "num_attention_heads": 12,
                        "hidden_size": 768,
                        "_name_or_path": "dmis-lab/biobert-v1.1",
                    },
                )()

            def forward(self, input_ids, **kwargs):
                return torch.randn(input_ids.shape[0], input_ids.shape[1], 768)

        model = MockModel()

        # Example 1: Single GPU configuration
        print("Single GPU configuration:")
        single_gpu_config = {
            "tensor_parallel_size": 1,
            "rank": 0,
            "world_size": 1,
            "use_cuda_graphs": True,
            "memory_efficient": True,
            "enable_mixed_precision": False,
            "skip_tokenizer_setup": True,
        }

        adapter_single = BioBERTAdapter(model, single_gpu_config)
        adapter_single.setup_for_inference(
            use_cuda_graphs=True, memory_efficient=True, enable_mixed_precision=False
        )

        stats = adapter_single.get_memory_stats()
        print(f"  - Total parameters: {stats['total_parameters']:,}")
        print(f"  - Tensor parallel size: {stats['tensor_parallel_size']}")
        print(f"  - Memory efficient: {adapter_single.memory_efficient}")

        # Example 2: Multi-GPU configuration
        print("\nMulti-GPU configuration (simulated):")
        multi_gpu_config = {
            "tensor_parallel_size": 4,
            "rank": 0,
            "world_size": 4,
            "use_cuda_graphs": False,  # Disable for multi-GPU demo
            "memory_efficient": True,
            "enable_mixed_precision": True,
            "skip_tokenizer_setup": True,
        }

        adapter_multi = BioBERTAdapter(model, multi_gpu_config)

        # Demonstrate tensor sharding
        test_tensor = torch.randn(768, 768)
        sharded_tensor = adapter_multi._shard_tensor(test_tensor, dim=0)
        print(f"  - Original tensor shape: {test_tensor.shape}")
        print(f"  - Sharded tensor shape: {sharded_tensor.shape}")
        print(f"  - Tensor parallel size: {adapter_multi.tensor_parallel_size}")
        print(f"  - Rank: {adapter_multi.rank}/{adapter_multi.world_size}")

        # Example 3: CUDA optimization features
        print("\nCUDA optimization features:")
        cuda_config = {
            "tensor_parallel_size": 1,
            "use_cuda_graphs": True,
            "memory_efficient": True,
            "enable_mixed_precision": True,
            "skip_tokenizer_setup": True,
        }

        adapter_cuda = BioBERTAdapter(model, cuda_config)

        # Setup with all optimizations
        adapter_cuda.setup_for_inference(
            use_cuda_graphs=True, memory_efficient=True, enable_mixed_precision=True
        )

        print(f"  - CUDA graphs enabled: {adapter_cuda.use_cuda_graphs}")
        print(f"  - Memory efficient: {adapter_cuda.memory_efficient}")
        print(f"  - Mixed precision: {adapter_cuda.enable_mixed_precision}")

        print("\n✅ Tensor parallelism & CUDA optimization features:")
        print("   - Multi-GPU tensor sharding")
        print("   - Distributed training support")
        print("   - CUDA memory optimization")
        print("   - Mixed precision training/inference")
        print("   - CUDA graphs for faster inference")
        print("   - Memory usage statistics")
        print("   - Automatic device management")

    except Exception as e:
        print(f"Tensor parallelism example failed: {e}")
        print("Note: This requires proper CUDA setup for full functionality")


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

    # Example 4: BioBERT-specific features
    biobert_specific_example()

    # Example 5: ClinicalBERT-specific features
    clinicalbert_specific_example()

    # Example 6: Tensor parallelism and CUDA optimization
    tensor_parallelism_example()

    print("\nTo use with real models:")
    print("1. Ensure you have the medical model downloaded")
    print("2. Create LLM instance with adapter configuration")
    print("3. Use generate() method as normal - adapter is transparent")


if __name__ == "__main__":
    main()
