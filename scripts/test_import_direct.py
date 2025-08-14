import sys
import torch
from transformers import AutoModelForSequenceClassification


def main():
    print("=== Testing Direct Import ===")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {AutoModelForSequenceClassification.__module__}")
    print("✅ Import successful!")


if __name__ == "__main__":
    main()
