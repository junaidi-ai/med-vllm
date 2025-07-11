# Medical Models Integration

This document provides an overview of the medical models integration in Med-vLLM, including how to use, extend, and customize medical models like BioBERT and ClinicalBERT.

## Table of Contents

1. [Overview](#overview)
2. [Available Models](#available-models)
3. [Quick Start](#quick-start)
4. [Advanced Usage](#advanced-usage)
5. [Extending with Custom Models](#extending-with-custom-models)
6. [Configuration Reference](#configuration-reference)
7. [Troubleshooting](#troubleshooting)

## Overview

The medical models integration provides a streamlined way to work with healthcare-specific language models in Med-vLLM. It includes:

- Pre-configured loaders for popular medical models
- Thread-safe model registry with caching
- Support for custom model configurations
- Easy integration with existing pipelines

## Available Models

### BioBERT

- **ID**: `biobert-base`
- **Type**: Biomedical Language Model
- **Base Model**: [dmis-lab/biobert-v1.1](https://huggingface.co/dmis-lab/biobert-v1.1)
- **Description**: A pre-trained biomedical language representation model for biomedical text mining tasks.

### Clinical BERT

- **ID**: `clinical-bert-base`
- **Type**: Clinical Language Model
- **Base Model**: [emilyalsentzer/Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
- **Description**: A pre-trained clinical language model for clinical text mining and NLP tasks.

## Quick Start

### Installation

First, install the required dependencies:

```bash
pip install -r requirements-medical.txt
```

### Basic Usage

```python
from medvllm.engine.model_runner.registry import get_registry

# Get the model registry
registry = get_registry()

# Load BioBERT model
biobert = registry.load_model("biobert-base", device="cuda")

# Load Clinical BERT model
clinical_bert = registry.load_model("clinical-bert-base", device="cuda")
```

## Advanced Usage

### Custom Model Configuration

You can customize model loading with additional parameters:

```python
# Load BioBERT with custom parameters
model = registry.load_model(
    "biobert-base",
    num_labels=3,  # For sequence classification
    output_attentions=True,
    output_hidden_states=True
)
```

### Using the Model Registry

List all available models:

```python
models = registry.list_models()
for model in models:
    print(f"{model['name']}: {model['description']}")
```

Get model metadata:

```python
metadata = registry.get_metadata("biobert-base")
print(f"Model type: {metadata.model_type}")
print(f"Tags: {', '.join(metadata.tags)}")
```

## Extending with Custom Models

### Creating a Custom Loader

Create a new loader by subclassing `MedicalModelLoader`:

```python
from medvllm.models.medical_models import MedicalModelLoader
from transformers import AutoModel, AutoTokenizer

class CustomMedicalLoader(MedicalModelLoader):
    """Loader for a custom medical model."""
    
    MODEL_NAME = "your-model/name"
    MODEL_TYPE = "custom"
    
    @classmethod
    def load_model(cls, **kwargs):
        """Load the custom model and tokenizer."""
        tokenizer = cls.load_tokenizer()
        model = AutoModel.from_pretrained(cls.MODEL_NAME, **kwargs)
        return model, tokenizer
```

### Registering a Custom Model

```python
# Register the custom model
registry.register(
    name="custom-medical-model",
    model_type=ModelType.BIOMEDICAL,  # or ModelType.CLIENT
    description="Custom medical model for specialized tasks",
    tags=["custom", "medical"],
    loader=CustomMedicalLoader,
    custom_param="value"
)

# Load the custom model
model = registry.load_model("custom-medical-model")
```

## Configuration Reference

### Model Configuration

Model configurations are stored in YAML files in `medvllm/configs/models/`. Example:

```yaml
custom_model:
  model_type: biomedical  # or clinical
  name: custom-model
  description: Custom medical model
  tags: [custom, biomedical]
  parameters:
    num_labels: 2
    output_attentions: true
    output_hidden_states: true
```

### Environment Variables

- `MEDVL_MEDICAL_MODELS_DIR`: Custom directory for medical model configurations
- `MEDVL_CACHE_DIR`: Custom cache directory for downloaded models

## Troubleshooting

### Model Loading Issues

- **Error**: "Medical models are not available"
  - **Solution**: Make sure to install the required dependencies from `requirements-medical.txt`

- **Error**: "CUDA out of memory"
  - **Solution**: Try loading the model on CPU or reduce the model size

### Performance Tips

1. **Enable caching**: Loaded models are cached by default
2. **Use appropriate device**: Load models on GPU when available
3. **Batch processing**: Process multiple inputs in batches when possible

## License

This software is licensed under the [MIT License](LICENSE).
