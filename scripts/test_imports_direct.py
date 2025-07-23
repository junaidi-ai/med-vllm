import sys
import torch
from transformers import AutoModelForSequenceClassification

def main():
    print("=== Testing Imports ===")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {AutoModelForSequenceClassification.__module__}")
    print("✅ All imports successful!")

if __name__ == "__main__":
    main()
