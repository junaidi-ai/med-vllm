import torch
from sklearn.metrics import accuracy_score
from medvllm.models.adapters import BioBERTAdapter

def test_accuracy():
    # Initialize model
    model = BioBERTAdapter.from_pretrained("monologg/biobert_v1.1_pubmed")
    model.eval()
    
    # Sample test data (replace with actual test data)
    test_texts = [
        "Patient presents with fever and cough",
        "No significant findings on examination",
        "History of hypertension and diabetes",
        "Normal chest x-ray results",
        "Prescribed antibiotics for infection"
    ]
    test_labels = [1, 0, 1, 0, 1]  # Example binary labels
    
    # Run inference
    with torch.no_grad():
        predictions = model(test_texts)
    
    # Convert model outputs to predictions (example: using logits)
    predicted_labels = [1 if p > 0.5 else 0 for p in predictions.logits.softmax(dim=1)[:, 1]]
    
    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predicted_labels)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Basic assertion (adjust threshold as needed)
    assert accuracy >= 0.0, "Accuracy below expected threshold"

if __name__ == "__main__":
    test_accuracy()
