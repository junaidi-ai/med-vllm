import pytest
import torch
from medvllm.models.adapters import BioBERTAdapter, ClinicalBERTAdapter

class TestMedicalModelPerformance:
    @pytest.fixture(params=["biobert", "clinicalbert"])
    def model_adapter(self, request):
        if request.param == "biobert":
            return BioBERTAdapter.from_pretrained("monologg/biobert_v1.1_pubmed")
        else:
            return ClinicalBERTAdapter.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_throughput(self, model_adapter, batch_size):
        # Create dummy input
        input_text = ["This is a test sentence."] * batch_size
        with torch.no_grad():
            outputs = model_adapter.preprocess_biomedical_text(input_text)
            model_output = model_adapter(**outputs)
        assert model_output is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_usage(self, model_adapter):
        torch.cuda.empty_cache()
        start_mem = torch.cuda.memory_allocated()
        
        # Process a batch of inputs
        input_text = ["Patient with fever and cough"] * 4
        with torch.no_grad():
            inputs = model_adapter.preprocess_biomedical_text(input_text)
            _ = model_adapter(**inputs)
        
        used_mem = (torch.cuda.memory_allocated() - start_mem) / 1e6  # MB
        assert used_mem > 0
