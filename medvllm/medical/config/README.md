# Medical Model Configuration

This module provides a flexible and type-safe configuration system for medical models, supporting various serialization formats and versioning.

## Structure

```
config/
├── __init__.py          # Public API and exports
├── base/                # Base configuration classes
│   ├── __init__.py
│   ├── base_config.py   # BaseMedicalConfig
│   └── model_config.py  # ModelConfig base
├── constants/           # Configuration constants
│   ├── __init__.py
│   ├── defaults.py      # Default values
│   └── enums.py         # All enums
├── models/              # Model-specific configurations
│   ├── __init__.py
│   └── medical_config.py # MedicalModelConfig
├── serialization/       # Serialization logic
│   ├── __init__.py
│   ├── base.py          # Base serializers
│   ├── json_serializer.py
│   └── yaml_serializer.py
├── utils/               # Utility functions
│   ├── __init__.py
│   ├── import_utils.py
│   └── validation.py
├── validation/          # Validation logic
│   ├── __init__.py
│   ├── base.py          # Base validators
│   └── medical_validator.py
└── versioning/          # Version management
    ├── __init__.py
    └── config_versioner.py
```

## Key Features

- **Type Safety**: Strong typing with Python type hints and runtime validation
- **Serialization**: Support for JSON and YAML formats
- **Versioning**: Configuration version management with migration support
- **Validation**: Comprehensive validation of configuration values
- **Modular Design**: Easy to extend with new configuration types and validators

## Usage

### Basic Configuration

```python
from medvllm.medical.config import MedicalModelConfig

# Create a new configuration
config = MedicalModelConfig(
    model_name_or_path="bert-base-uncased",
    model_type="bert",
    medical_specialties=["radiology", "cardiology"],
    max_sequence_length=512
)

# Convert to dictionary
config_dict = config.to_dict()

# Save to file
config.to_json("config.json")
config.to_yaml("config.yaml")

# Load from file
loaded_config = MedicalModelConfig.from_dict(config_dict)
```

### Serialization

```python
from medvllm.medical.config.serialization import JSONSerializer, YAMLSerializer

# Create serializers
json_serializer = JSONSerializer()
yaml_serializer = YAMLSerializer()

# Serialize to string
json_str = json_serializer.serialize(config)
yaml_str = yaml_serializer.serialize(config)

# Deserialize from string
loaded_from_json = json_serializer.deserialize(json_str, MedicalModelConfig)
loaded_from_yaml = yaml_serializer.deserialize(yaml_str, MedicalModelConfig)
```

### Validation

```python
from medvllm.medical.config.validation import MedicalConfigValidator

validator = MedicalConfigValidator()

# Validate configuration
try:
    validated_config = validator.validate(config_dict, MedicalModelConfig)
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
