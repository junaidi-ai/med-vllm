"""Accuracy testing for medical models against original implementations."""

import torch
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from medvllm.models.adapters import BioBERTAdapter, ClinicalBERTAdapter

class AccuracyTester:
    """Test accuracy of medical model adapters against original models."""
    
    def __init__(self, model_type: str = "biobert"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = model_type.lower()
        self.setup_models()
    
    def setup_models(self):
        """Load both adapter and original models."""
        if self.model_type == "biobert":
            model_name = "monologg/biobert_v1.1_pubmed"
            self.adapter = BioBERTAdapter.from_pretrained(model_name).to(self.device)
        else:  # clinicalbert
            model_name = "emilyalsentzer/Bio_ClinicalBERT"
            self.adapter = ClinicalBERTAdapter.from_pretrained(model_name).to(self.device)
        
        self.adapter.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.original_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        ).to(self.device).eval()
    
    def predict_with_adapter(self, texts):
        with torch.no_grad():
            outputs = self.adapter(texts)
            return torch.softmax(outputs.logits, dim=-1).cpu().numpy()
    
    def predict_with_original(self, texts):
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, 
            max_length=512, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.original_model(**inputs)
            return torch.softmax(outputs.logits, dim=-1).cpu().numpy()

def test_accuracy_comparison():
    """Test accuracy comparison between adapter and original models."""
    print("Running accuracy comparison tests...")
    
    for model_type in ["biobert", "clinicalbert"]:
        print(f"\nTesting {model_type}...")
        tester = AccuracyTester(model_type)
        
        # Test with sample medical texts
        test_texts = [
            "Patient with fever and cough",
            "No significant findings on examination",
            "CT scan shows pneumonia",
            "Vital signs stable"
        ]
        
        # Get predictions
        adapter_preds = tester.predict_with_adapter(test_texts)
        original_preds = tester.predict_with_original(test_texts)
        
        # Check agreement
        agreement = np.mean(
            np.argmax(adapter_preds, axis=1) == np.argmax(original_preds, axis=1)
        )
        print(f"{model_type} prediction agreement: {agreement:.2f}")

if __name__ == "__main__":
    test_accuracy_comparison()
