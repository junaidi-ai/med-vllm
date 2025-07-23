"""Accuracy validation for medical models against original implementations."""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from medvllm.models.adapters import BioBERTAdapter, ClinicalBERTAdapter
from medvllm.utils.datasets import MEDICAL_TASKS, load_medical_dataset

# Model mapping between our adapters and original models
MODEL_MAPPING = {
    "biobert": "monologg/biobert_v1.1_pubmed",
    "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
}

# Task-specific configurations
TASK_CONFIGS = {
    "medical_questions_pairs": {
        "dataset": "medical_questions_pairs",
        "metric": "accuracy",
        "text_field": "question",
        "label_field": "label",
        "max_length": 256,
    },
    "i2b2_2010_relations": {
        "dataset": "i2b2_2010_relations",
        "metric": "f1",
        "text_field": "text",
        "label_field": "relation",
        "max_length": 128,
    },
}


def load_original_model(model_name: str, num_labels: int = 2):
    """Load the original model from Hugging Face."""
    model_path = MODEL_MAPPING[model_name]
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def evaluate_model(
    model, tokenizer, dataset, config: Dict[str, str], batch_size: int = 8
) -> Dict[str, float]:
    """Evaluate a model on a given dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Prepare data
    texts = dataset[config["text_field"]]
    labels = dataset[config["label_field"]]
    
    # Tokenize
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=config["max_length"],
        return_tensors="pt",
    )
    
    # Predict in batches
    all_preds = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating"):
        batch = {k: v[i : i + batch_size].to(device) for k, v in encodings.items()}
        with torch.no_grad():
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
    
    # Calculate metrics
    if config["metric"] == "accuracy":
        score = accuracy_score(labels, all_preds)
    else:  # f1
        score = f1_score(labels, all_preds, average="weighted")
    
    return {"score": score, "metric": config["metric"], "predictions": all_preds}


def validate_accuracy(
    model_type: str, task: str, output_dir: str = "results"
) -> Dict[str, Union[float, List[int]]]:
    """Validate accuracy against original model on a specific task."""
    print(f"\n=== Validating {model_type} on {task} ===")
    
    # Load config
    config = TASK_CONFIGS[task]
    
    # Load dataset
    dataset = load_medical_dataset(task, split="test")
    
    # Evaluate our adapter
    print("\nEvaluating our adapter...")
    if model_type == "biobert":
        adapter = BioBERTAdapter.from_pretrained("monologg/biobert_v1.1_pubmed")
    else:  # clinicalbert
        adapter = ClinicalBERTAdapter.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    
    our_results = evaluate_model(adapter, adapter.tokenizer, dataset, config)
    
    # Evaluate original model
    print("\nEvaluating original model...")
    original_model, tokenizer = load_original_model(
        model_type, num_labels=len(set(dataset[config["label_field"]]))
    )
    original_results = evaluate_model(original_model, tokenizer, dataset, config)
    
    # Calculate difference
    diff = our_results["score"] - original_results["score"]
    
    # Save results
    output_path = Path(output_dir) / f"{model_type}_{task}_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        "model": model_type,
        "task": task,
        "our_score": our_results["score"],
        "original_score": original_results["score"],
        "difference": diff,
        "metric": config["metric"],
        "our_predictions": our_results["predictions"],
        "original_predictions": original_results["predictions"],
    }
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print(f"Our {config['metric']}: {our_results['score']:.4f}")
    print(f"Original {config['metric']}: {original_results['score']:.4f}")
    print(f"Difference: {diff:+.4f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate medical model accuracy")
    parser.add_argument("--model", type=str, choices=["biobert", "clinicalbert"], required=True)
    parser.add_argument("--task", type=str, choices=list(TASK_CONFIGS.keys()), required=True)
    parser.add_argument("--output-dir", type=str, default="results")
    
    args = parser.parse_args()
    validate_accuracy(args.model, args.task, args.output_dir)
