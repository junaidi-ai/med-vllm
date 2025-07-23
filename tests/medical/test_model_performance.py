import pytest
import torch
from medvllm.models.adapters import BioBERTAdapter, ClinicalBERTAdapter

class TestMedicalModelPerformance:
    @pytest.fixture(params=["biobert", "clinicalbert"])
    def model_adapter(self, request):
        if request.param == "biobert":
            return BioBERTAdapter.from_pretrained("monologg/biobert_v1.1_pubmed")
        return ClinicalBERTAdapter.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_throughput(self, model_adapter, batch_size):
        inputs = torch.randint(0, 1000, (batch_size, 128))
        with torch.no_grad():
            outputs = model_adapter(inputs)
        assert outputs is not None

    def test_memory_usage(self, model_adapter):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            start_mem = torch.cuda.memory_allocated()
            inputs = torch.randint(0, 1000, (1, 256))
            with torch.no_grad():
                _ = model_adapter(inputs)
            used_mem = (torch.cuda.memory_allocated() - start_mem) / 1e6  # MB
            assert used_mem > 0
