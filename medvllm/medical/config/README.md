# Medical Model Configuration

A robust, type-safe configuration system for medical models with support for multiple serialization formats, versioning, and comprehensive validation.

## 🏗️ Package Structure

```
config/
├── __init__.py               # Public API and exports
├── base/                     # Base configuration classes
│   ├── __init__.py
│   ├── base_config.py        # BaseMedicalConfig
│   └── model_config.py       # ModelConfig base
├── constants/                # Configuration constants and enums
│   ├── __init__.py
│   ├── defaults.py           # Default configuration values
│   └── enums.py              # Enumerations for model types, etc.
├── models/                   # Model configurations and schemas
│   ├── __init__.py
│   ├── schema.py             # Pydantic schemas
│   └── medical_config.py     # MedicalModelConfig implementation
├── serialization/            # Serialization framework
│   ├── __init__.py           # High-level serialization API
│   ├── config_serializer.py  # Base ConfigSerializer
│   ├── json_serializer.py    # JSON serialization
│   └── yaml_serializer.py    # YAML serialization (requires PyYAML)
├── types/                    # Type definitions and aliases
│   ├── __init__.py
│   └── common.py             # Common type definitions
├── utils/                    # Utility functions
│   ├── __init__.py
│   ├── type_utils.py         # Type inspection utilities
│   ├── path_utils.py         # Path handling utilities
│   └── misc.py               # Miscellaneous helpers
├── validation/               # Validation framework
│   ├── __init__.py           # Public validation API
│   ├── exceptions.py         # Custom exceptions
│   ├── schema.py             # Schema validation
│   └── validators.py         # Validator implementations
└── versioning/               # Version management
    ├── __init__.py           # Versioning API
    ├── exceptions.py         # Version-related exceptions
    └── migrators/           # Version migration logic
        ├── __init__.py
        └── v1_to_v2.py       # Example migration
```

## ✨ Key Features

- **Type Safety**: Comprehensive type hints with runtime validation using Pydantic
- **Multiple Formats**: Built-in support for JSON and YAML (with PyYAML)
- **Versioning**: Configuration version management with migration support
- **Validation**: Extensive validation with detailed error messages
- **Modular Design**: Clean separation of concerns with a plugin-friendly architecture
- **Documentation**: Complete docstrings with examples and type hints

## 🚀 Quick Start

### Installation

```bash
pip install -e .  # For development
# or
pip install medvllm
```

### Basic Configuration

```python
from medvllm.medical.config import MedicalModelConfig
from medvllm.medical.config.constants import ModelType

# Create a configuration with type hints and validation
config = MedicalModelConfig(
    model_name_or_path="bert-base-uncased",
    model_type=ModelType.BERT,
    tensor_parallel_size=2,
    medical_specialties=["radiology", "cardiology"],
    max_sequence_length=512,
    entity_linking={
        "enabled": True,
        "knowledge_bases": ["umls", "snomed"]
    }
)

# Convert to dictionary (recursively converts all nested models)
config_dict = config.dict()

# Save to file (automatically handles file extension)
config.to_file("config.json")  # or .yaml/.yml

# Load from file (auto-detects format)
loaded_config = MedicalModelConfig.from_file("config.json")
```

### Advanced Usage

#### Serialization

```python
from medvllm.medical.config.serialization import (
    JSONSerializer,
    YAMLSerializer,
    save_config,
    load_config
)

# Using serializer instances
json_serializer = JSONSerializer(indent=2)
json_str = json_serializer.serialize(config)
loaded_config = json_serializer.deserialize(json_str, MedicalModelConfig)

# Using convenience functions
save_config(config, "config.json")  # Auto-detects format
loaded_config = load_config("config.json", MedicalModelConfig)
```

#### Validation

```python
from medvllm.medical.config.validation import (
    validate_config_schema,
    MedicalConfigValidator,
    ValidationError
)

# Schema validation
try:
    validate_config_schema(config_dict, MedicalModelConfig)
    print("Configuration is valid!")
except ValidationError as e:
    print(f"Validation error: {e}")

# Using the validator
validator = MedicalConfigValidator()
validator.validate_medical_parameters(config)

# Access the default validator instance
from medvllm.medical.config.validation import default_validator
default_validator.validate_tensor_parallel_size(4)
```

#### Version Management

```python
from medvllm.medical.config.versioning import ConfigVersioner

# Check version compatibility
version_info = ConfigVersioner.get_version_info("1.0.0")
if version_info.status == "deprecated":
    print(f"Warning: {version_info.message}")

# Migrate configuration to latest version
ConfigVersioner.migrate_config(config)
```

## 🧪 Testing

Run the test suite with:

```bash
pytest tests/medical/config/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) for details.
except ValueError as e:
    print(f"Validation error: {e}")
```

### Versioning

```python
from medvllm.medical.config.versioning import ConfigVersioner

# Check version compatibility
try:
    ConfigVersioner.check_version_compatibility("1.0.0")
except ValueError as e:
    print(f"Version error: {e}")

# Migrate configuration
migrated_config = ConfigVersioner.migrate_config(old_config_dict)
```

## Adding New Configuration Types

1. Create a new class in the appropriate module (or create a new one)
2. Inherit from `BaseMedicalConfig` or `ModelConfig`
3. Define your configuration fields with type hints
4. Add validation logic if needed
5. Update the `__all__` list in the module's `__init__.py`

## Running Tests

```bash
pytest tests/medical/config/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
