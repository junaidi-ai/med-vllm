# Model Registry

The Model Registry provides a centralized way to manage and load different models in the Med-vLLM framework. It supports model registration, loading, and caching with metadata management.

## Features

- **Model Registration**: Register models with metadata including type, description, and parameters
- **Automatic Loading**: Load models by name with automatic caching
- **Type Safety**: Strong typing support for model classes and configurations
- **Thread Safety**: Thread-safe operations for concurrent access
- **Fallback Support**: Falls back to direct loading if model not in registry

## Basic Usage

### Importing the Registry

```python
from medvllm.engine.model_runner import registry, ModelType
```

### Registering a Model

```python
from transformers import AutoModelForCausalLM, AutoConfig

# Register a new model
registry.register(
    name="my-biobert-model",
    model_type=ModelType.BIOMEDICAL,
    model_class=AutoModelForCausalLM,
    config_class=AutoConfig,
    description="A biomedical language model fine-tuned on clinical data",
    tags=["biomedical", "bert", "clinical"],
    model_path="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
)
```

### Loading a Model

```python
# Load a model (will use cache if available)
model = registry.load_model("my-biobert-model")

# Load with custom parameters
model = registry.load_model(
    "my-biobert-model",
    device="cuda",
    torch_dtype=torch.float16
)
```

### Listing Available Models

```python
# List all models
all_models = registry.list_models()

# Filter by model type
biomed_models = registry.list_models(ModelType.BIOMEDICAL)
```

### Using the Global Registry

The module provides a global registry instance that's pre-configured with common models:

```python
from medvllm.engine.model_runner import registry

# List pre-registered models
print(registry.list_models())

# Load a pre-registered model
model = registry.load_model("biobert-base-cased-v1.2")
```

## Advanced Usage

### Custom Model Classes

You can register custom model classes that inherit from `PreTrainedModel`:

```python
from transformers import PreTrainedModel

class MyCustomModel(PreTrainedModel):
    # Implementation...
    pass

# Register the custom model
registry.register(
    name="my-custom-model",
    model_type=ModelType.CUSTOM,
    model_class=MyCustomModel,
    description="A custom model implementation"
)
```

### Model Cache Management

The registry caches loaded models by default. You can manage the cache:

```python
# Clear the entire cache
registry.clear_cache()

# Check if a model is in cache
if "my-model" in registry._model_cache:
    print("Model is cached")
```

## Pre-registered Models

The registry comes with several pre-registered models:

| Name | Type | Description |
|------|------|-------------|
| biobert-base-cased-v1.2 | BIOMEDICAL | BioBERT base cased v1.2 |
| emilyalsentzer/Bio_ClinicalBERT | CLINICAL | ClinicalBERT model |

## Best Practices

1. **Use Meaningful Names**: Choose descriptive names for your models
2. **Register Early**: Register models at application startup
3. **Reuse Instances**: Use the registry to avoid loading the same model multiple times
4. **Handle Errors**: Always handle potential loading errors
5. **Clean Up**: Clear the cache if you need to free up memory

## API Reference

### `ModelRegistry`

Main registry class for managing models.

#### Methods

- `register(name: str, model_type: ModelType, **kwargs)`: Register a new model
- `load_model(name: str, **kwargs) -> PreTrainedModel`: Load a model by name
- `list_models(model_type: Optional[ModelType] = None) -> Dict[str, ModelMetadata]`: List registered models
- `clear_cache() -> None`: Clear the model cache
- `unregister(name: str) -> None`: Remove a model from the registry
- `get_metadata(name: str) -> ModelMetadata`: Get metadata for a registered model

### `ModelMetadata`

Dataclass containing model metadata.

#### Attributes

- `name`: Unique identifier for the model
- `model_type`: Type of the model (BIOMEDICAL, CLINICAL, etc.)
- `model_class`: The model class to use for instantiation
- `config_class`: The config class to use
- `description`: Human-readable description
- `tags`: List of tags for categorization
- `parameters`: Additional model parameters

### `ModelType`

Enum of supported model types:
- `GENERIC`
- `BIOMEDICAL`
- `CLINICAL`
