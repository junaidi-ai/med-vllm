"""
Base configuration classes for medical models.

This module provides the BaseMedicalConfig class which serves as the foundation
for all medical model configurations in the medvllm library.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type, TypeVar

from medvllm.config import Config

CONFIG_VERSION = "1.0.0"

# Type variable for class methods that return an instance of the class
T = TypeVar("T", bound="BaseMedicalConfig")

logger = logging.getLogger(__name__)


@dataclass
class BaseMedicalConfig(Config):
    """Base configuration class for medical models.

    This class provides common functionality and fields for medical model
    configurations. It inherits from the base Config class and adds
    medical-specific features.

    Attributes:
        config_version: Version of the configuration schema
        file_path: Path to the config file if loaded from disk
        model_type: Type of the model (default: "base")
    """

    config_version: str = field(default=CONFIG_VERSION, init=True, metadata={"description": "Version of the configuration schema"})
    file_path: Optional[str] = field(
        default=None
    )  # Path to the config file if loaded from disk
    model_type: str = field(default="base")  # Default model type

    @classmethod
    def from_dict(cls, config_dict: dict) -> "BaseMedicalConfig":
        """Create a BaseMedicalConfig instance from a dictionary.
        
        This method creates a new configuration instance from a dictionary,
        properly handling both known and unknown parameters.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            A new BaseMedicalConfig instance with all attributes properly set
        """
        # Make a copy to avoid modifying the input
        config_dict = dict(config_dict)
        
        # Ensure required parameters have defaults
        config_dict.setdefault('model', 'default-model')
        config_dict.setdefault('model_type', 'base')
        
        # Known fields that should be passed to the constructor
        known_fields = {
            "model", "max_num_batched_tokens", "max_num_seqs", "max_model_len",
            "gpu_memory_utilization", "tensor_parallel_size", "enforce_eager",
            "hf_config", "eos", "kvcache_block_size", "num_kvcache_blocks",
            "model_type", "adapter_type", "adapter_config", "use_medical_adapter",
            "memory_efficient", "use_cuda_graphs", "enable_mixed_precision",
            "config_version", "file_path"
        }
        
        # Separate known and unknown parameters
        known_params = {k: v for k, v in config_dict.items() if k in known_fields}
        unknown_params = {k: v for k, v in config_dict.items() if k not in known_fields}
        
        # Create a new instance without calling __init__
        instance = object.__new__(cls)
        
        # Initialize _extra_fields directly on the instance
        object.__setattr__(instance, '_extra_fields', {})
        
        # Set known attributes directly on the instance
        for key, value in known_params.items():
            object.__setattr__(instance, key, value)
        
        # Call __init__ with known parameters
        instance.__init__(**known_params)
        
        # Set unknown parameters as direct attributes and store in _extra_fields
        for key, value in unknown_params.items():
            # Store in _extra_fields for serialization
            instance._extra_fields[key] = value
            # Set as a direct attribute for direct access
            object.__setattr__(instance, key, value)
        
        # Call __post_init__ if it exists
        if hasattr(instance, '__post_init__'):
            instance.__post_init__()
            
            # Ensure dynamic attributes are still accessible after __post_init__
            for key, value in unknown_params.items():
                if not hasattr(instance, key):
                    object.__setattr__(instance, key, value)
        
        return instance

    def __init__(self, *args, **kwargs):
        """Initialize the configuration with support for extra fields.
        
        This method handles the initialization of the configuration object, including:
        1. Separating known parameters (for parent class) from extra fields
        2. Initializing _extra_fields for dynamic attributes
        3. Calling the parent class's __init__ with known parameters
        4. Processing and storing any extra fields as dynamic attributes
        
        Args:
            *args: Positional arguments (not used, for compatibility)
            **kwargs: Keyword arguments for configuration
        """
        print(f"\n[DEBUG] __init__ called with args: {args}, kwargs: {kwargs}")
        
        # Known parameters that should be passed to the parent class
        known_params = {
            "model",
            "max_num_batched_tokens",
            "max_num_seqs",
            "max_model_len",
            "gpu_memory_utilization",
            "tensor_parallel_size",
            "enforce_eager",
            "hf_config",
            "eos",
            "kvcache_block_size",
            "num_kvcache_blocks",
            "model_type",
            "adapter_type",
            "adapter_config",
            "use_medical_adapter",
            "memory_efficient",
            "use_cuda_graphs",
            "enable_mixed_precision",
            "config_version",
            "file_path"
        }
        
        # Separate known params from extra fields
        parent_kwargs = {k: v for k, v in kwargs.items() if k in known_params}
        extra_fields = {k: v for k, v in kwargs.items() if k not in known_params and not k.startswith('_')}
        
        print(f"[DEBUG] Parent kwargs: {parent_kwargs}")
        print(f"[DEBUG] Extra fields: {extra_fields}")
        
        # Initialize _extra_fields before parent __init__
        object.__setattr__(self, '_extra_fields', {})
        print(f"[DEBUG] After _extra_fields init: {self._extra_fields}")
        
        # Store extra fields before calling parent __init__
        for k, v in extra_fields.items():
            print(f"[DEBUG] Pre-setting extra field: {k} = {v}")
            self._extra_fields[k] = v
        
        # Call parent's __init__ with known parameters
        print("[DEBUG] Calling parent __init__")
        super().__init__(**parent_kwargs)
        
        print(f"[DEBUG] After parent __init__, _extra_fields: {getattr(self, '_extra_fields', 'NOT FOUND')}")
        print(f"[DEBUG] After parent __init__, __dict__: {self.__dict__}")
        
        # Process extra fields again to ensure they're set after parent __init__
        print("[DEBUG] Processing extra fields")
        for k, v in extra_fields.items():
            print(f"[DEBUG] Setting extra field: {k} = {v}")
            # Store in _extra_fields for serialization
            self._extra_fields[k] = v
            # Also set as a direct attribute for direct access
            object.__setattr__(self, k, v)
            print(f"[DEBUG] After setting {k}, _extra_fields: {self._extra_fields}")
            print(f"[DEBUG] After setting {k}, hasattr: {hasattr(self, k)}, value: {getattr(self, k, 'NOT FOUND')}")
        
        # Verify all extra fields are accessible
        for k in extra_fields:
            if not hasattr(self, k):
                print(f"[WARNING] Extra field {k} is not accessible after setting")
            else:
                print(f"[DEBUG] Verified access to extra field: {k} = {getattr(self, k, 'NOT FOUND')}")

        # Set other attributes and store in _extra_fields
        for k, v in kwargs.items():
            if k not in known_params and not hasattr(self, k):
                self._extra_fields[k] = v
                # Also set as an attribute for direct access
                object.__setattr__(self, k, v)

        # Ensure model_type is set
        if not hasattr(self, "model_type") or self.model_type is None:
            self.model_type = "base"
            self._extra_fields["model_type"] = "base"

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute with special handling for dynamic fields.
        
        This method is called when an attribute is set on the instance.
        It handles dynamic fields by storing them in the _extra_fields dictionary.
        
        Args:
            name: Name of the attribute to set
            value: Value to set the attribute to
        """
        # Skip debug prints for known attributes to reduce noise
        debug = not name.startswith('__') and name not in ['_extra_fields', 'model_type', 'model']
        
        if debug:
            print(f"\n[DEBUG] __setattr__ called with name: {name}, value: {value}")
        
        # Special handling for _extra_fields to prevent recursion
        if name == '_extra_fields':
            if debug:
                print("[DEBUG] Setting _extra_fields directly")
            object.__setattr__(self, name, value)
            return
        
        # Get all known fields from the class hierarchy
        fields = set()
        for base in self.__class__.__mro__:
            if hasattr(base, '__dataclass_fields__'):
                fields.update(base.__dataclass_fields__.keys())
        
        # Add special fields that should be treated as known
        fields.update({'model', 'model_type'})
        
        if debug:
            print(f"[DEBUG] Known fields: {fields}")
        
        # If this is a known field, set it directly
        if name in fields:
            if debug:
                print(f"[DEBUG] Setting known field: {name} = {value}")
            object.__setattr__(self, name, value)
            return
        
        # Ensure _extra_fields exists
        if not hasattr(self, '_extra_fields'):
            if debug:
                print("[DEBUG] Initializing empty _extra_fields")
            object.__setattr__(self, '_extra_fields', {})
        
        # It's a dynamic field - store it in _extra_fields and as an attribute
        if debug:
            print(f"[DEBUG] Setting dynamic field: {name} = {value}")
        
        # Store in _extra_fields for serialization
        self._extra_fields[name] = value
        
        # Also set as a direct attribute for direct access
        object.__setattr__(self, name, value)
        
        if debug:
            print(f"[DEBUG] After setting {name}, _extra_fields: {self._extra_fields}")
            print(f"[DEBUG] After setting {name}, hasattr({name}): {hasattr(self, name)}")
            print(f"[DEBUG] After setting {name}, getattr({name}): {getattr(self, name, 'NOT FOUND')}")
            print(f"[DEBUG] Instance __dict__: {self.__dict__}")
        
        # Also set as a direct attribute for direct access
        object.__setattr__(self, name, value)
        
    def __getattr__(self, name: str) -> Any:
        """Get attribute, with special handling for dynamic fields.
        
        This method is called when an attribute is not found through normal lookup.
        It checks the following locations in order:
        1. The _extra_fields dictionary (for dynamic attributes)
        2. The instance's __dict__ (for direct attributes)
        3. The class attributes (via object.__getattribute__)
        
        Args:
            name: Name of the attribute to get
            
        Returns:
            The attribute value if found
            
        Raises:
            AttributeError: If the attribute is not found in any of the locations
        """
        # Skip debug prints for known attributes to reduce noise
        debug = not name.startswith('__') and name not in ['_extra_fields']
        
        if debug:
            print(f"\n[DEBUG] __getattr__ called with name: {name}")
        
        # Special handling for _extra_fields to prevent recursion
        if name == '_extra_fields':
            if debug:
                print("[DEBUG] Accessing _extra_fields directly")
            # Initialize _extra_fields if it doesn't exist
            if '_extra_fields' not in self.__dict__:
                if debug:
                    print("[DEBUG] Initializing empty _extra_fields")
                self.__dict__['_extra_fields'] = {}
            return self.__dict__['_extra_fields']
        
        # First, check _extra_fields
        if hasattr(self, '_extra_fields') and name in self._extra_fields:
            if debug:
                print(f"[DEBUG] Found {name} in _extra_fields: {self._extra_fields[name]}")
            return self._extra_fields[name]
        
        # Then check instance __dict__
        if name in self.__dict__:
            if debug:
                print(f"[DEBUG] Found {name} in instance __dict__: {self.__dict__[name]}")
            return self.__dict__[name]
        
        # Try to get the attribute from the class
        try:
            if debug:
                print(f"[DEBUG] Trying to get attribute via object.__getattribute__")
            value = object.__getattribute__(self, name)
            if debug:
                print(f"[DEBUG] Found {name} via object.__getattribute__: {value}")
            return value
        except AttributeError as e:
            # Get all available attributes for a more helpful error message
            available_attrs = set()
            # Get instance attributes
            available_attrs.update(self.__dict__.keys())
            # Get class attributes
            available_attrs.update(dir(self.__class__))
            # Get attributes from _extra_fields if it exists
            if hasattr(self, '_extra_fields'):
                available_attrs.update(self._extra_fields.keys())
            
            # Filter out private attributes and methods
            available_attrs = {a for a in available_attrs if not a.startswith('__') and not callable(getattr(self, a, None))}
            
            if debug:
                print(f"[DEBUG] Attribute {name} not found. Available attributes: {sorted(available_attrs)}")
            
            # Check if the attribute exists in __dict__ but not accessible
            if name in self.__dict__:
                print(f"[WARNING] {name} is in __dict__ but not accessible")
            
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'. "
                f"Available attributes: {', '.join(sorted(available_attrs))}"
            ) from e

    def __post_init__(self):
        """Initialize the configuration with default values and validate.
        
        This method performs the following steps:
        1. Preserve any existing _extra_fields
        2. Call the parent's __post_init__ if it exists
        3. Ensure model_type is set
        4. Initialize _extra_fields if it doesn't exist
        5. Restore any preserved _extra_fields
        6. Validate the configuration
        """
        # Preserve any existing _extra_fields
        extra_fields = getattr(self, '_extra_fields', {})
        
        # Call parent's __post_init__ if it exists
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
            
        # Ensure model_type is set
        if not hasattr(self, "model_type") or self.model_type is None:
            self.model_type = "base"
            extra_fields["model_type"] = "base"
            
        # Initialize _extra_fields if it doesn't exist
        if not hasattr(self, '_extra_fields'):
            self._extra_fields = {}
            
        # Restore any preserved _extra_fields
        self._extra_fields.update(extra_fields)
        
        # Set any extra fields as attributes
        for key, value in self._extra_fields.items():
            if not hasattr(self, key):
                object.__setattr__(self, key, value)
            
        # Validate the configuration
        self.validate()

    def ensure_compatibility(self) -> bool:
        """Ensure the configuration is compatible with the current version."""
        has_version = hasattr(self, "config_version")
        version_matches = has_version and self.config_version == CONFIG_VERSION
        if not has_version or not version_matches:
            self._migrate_config()
            return False
        return True

    def _migrate_config(self) -> None:
        """Migrate the configuration to the latest version.
        
        This method handles migration of configuration from older versions to the current version.
        It preserves the original version in the configuration to maintain backward compatibility.
        
        The method performs the following steps:
        1. Gets the current version or defaults to "0.1.0"
        2. If the version is older than current, performs migration steps
        3. Preserves the original version in a separate field
        4. Updates the config_version to the current version
        """
        version = getattr(self, "config_version", "0.1.0")
        
        # Store the original version before any migration
        if not hasattr(self, '_original_version'):
            self._original_version = version
        
        # Only migrate if version is older than current
        if version == "0.1.0":
            # Migrate from 0.1.0 to 1.0.0
            if hasattr(self, "medical_params") and isinstance(self.medical_params, dict):
                # Copy over any medical parameters to top level
                for key, value in self.medical_params.items():
                    if not hasattr(self, key):
                        setattr(self, key, value)
                delattr(self, "medical_params")
            
            # Update the version to current
            self.config_version = CONFIG_VERSION
        
        # Ensure the config version is set to current if it wasn't set
        if not hasattr(self, 'config_version') or not self.config_version:
            self.config_version = CONFIG_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary.
        
        This method handles conversion of all fields to a serializable format,
        including enums, nested objects, and special types. It ensures that the
        resulting dictionary can be properly serialized to JSON or YAML.
        
        The method preserves the original version if it was set during migration.
        All enum values are converted to their string representations.
        
        Returns:
            Dictionary containing all configuration parameters in a serializable format
            
        Example:
            >>> config = MedicalModelConfig(...)
            >>> config_dict = config.to_dict()
            >>> isinstance(config_dict, dict)
            True
        """
        # Initialize serializing flag if not exists
        if not hasattr(self, '_serializing'):
            self._serializing = False
            
        # Prevent recursive serialization
        if self._serializing:
            return {}
            
        try:
            # Set serializing flag
            self._serializing = True
            
            def convert_value(value):
                """Convert a value to a serializable format."""
                if isinstance(value, (str, int, float, bool)) or value is None:
                    return value
                if hasattr(value, 'to_dict'):
                    return value.to_dict()
                if hasattr(value, 'value'):  # Handle enum values
                    return value.value
                if isinstance(value, (list, tuple, set)):
                    return [convert_value(v) for v in value]
                if isinstance(value, dict):
                    return {k: convert_value(v) for k, v in value.items()}
                return str(value)  # Fallback to string representation
            
            result = {}
            
            # Get all fields including those from parent classes
            for field_name, field_value in self.__dict__.items():
                # Skip private and special attributes
                if field_name.startswith('_') or field_name in {'_extra_fields'}:
                    continue
                    
                # Skip methods and other non-serializable attributes
                if callable(field_value):
                    continue
                    
                # Convert the value to a serializable format
                result[field_name] = convert_value(field_value)
            
            # Include dynamic attributes from _extra_fields
            if hasattr(self, '_extra_fields'):
                for key, value in self._extra_fields.items():
                    if key not in result:
                        result[key] = convert_value(value)
            
            # Ensure model_type is included if not already
            if 'model_type' not in result and hasattr(self, 'model_type'):
                result['model_type'] = self.model_type
                
            # Add original version if it was preserved during migration
            if hasattr(self, '_original_version'):
                result['config_version'] = self._original_version
            # Ensure config_version is included
            elif 'config_version' not in result and hasattr(self, 'config_version'):
                result['config_version'] = self.config_version
                
            return result
        finally:
            # Clean up the serializing flag
            self._serializing = False

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create a configuration from a dictionary.
        
        Handles conversion of string values to appropriate types including enums.
        Preserves the original version if present in the input data.

        Args:
            data: Dictionary containing configuration parameters

        Returns:
            New instance of the configuration with the correct type and field values
        """
        if not isinstance(data, dict):
            raise ValueError(f"Expected dictionary, got {type(data).__name__}")
            
        print(f"\n[DEBUG] from_dict called with data: {data}")
        
        # Create a copy of the data to avoid modifying the input
        data_copy = data.copy()
        
        # Preserve original version if present
        original_version = data_copy.pop("config_version", None)
        
        # Get all dataclass fields including those from parent classes
        fields = set()
        for base in cls.__mro__:
            if hasattr(base, '__dataclass_fields__'):
                fields.update(base.__dataclass_fields__.keys())
        print(f"[DEBUG] All dataclass fields: {fields}")
        
        # Separate known fields from extra fields
        known_fields = {k: v for k, v in data_copy.items() if k in fields}
        extra_fields = {k: v for k, v in data_copy.items() if k not in fields}
        
        print(f"[DEBUG] Known fields: {known_fields}")
        print(f"[DEBUG] Extra fields: {extra_fields}")
        
        # Create instance with known fields
        instance = object.__new__(cls)
        
        # Initialize _extra_fields before setting any attributes
        object.__setattr__(instance, '_extra_fields', {})
        
        # Set known fields using object.__setattr__ to bypass our __setattr__
        for k, v in known_fields.items():
            object.__setattr__(instance, k, v)
        
        # Set extra fields in _extra_fields and as direct attributes
        for k, v in extra_fields.items():
            instance._extra_fields[k] = v
            object.__setattr__(instance, k, v)
        
        # Set config version if provided
        if original_version is not None:
            object.__setattr__(instance, 'config_version', original_version)
        
        # Call __post_init__ if it exists
        if hasattr(instance, '__post_init__'):
            instance.__post_init__()
        
        # Ensure all extra fields are properly set as attributes
        for k in extra_fields.keys():
            if not hasattr(instance, k):
                print(f"[WARNING] Extra field {k} not set as attribute, fixing...")
                object.__setattr__(instance, k, extra_fields[k])
            
        print(f"[DEBUG] Instance after from_dict: {instance}")
        print(f"[DEBUG] Instance _extra_fields: {getattr(instance, '_extra_fields', 'NOT FOUND')}")
        print(f"[DEBUG] Instance __dict__: {instance.__dict__}")
        print(f"[DEBUG] Has custom_field: {hasattr(instance, 'custom_field')}")
        if hasattr(instance, 'custom_field'):
            print(f"[DEBUG] custom_field value: {getattr(instance, 'custom_field', 'NOT FOUND')}")
        
        return instance
        if not hasattr(instance, '_extra_fields'):
            print("[DEBUG] Initializing _extra_fields")
            instance._extra_fields = {}
        
        # Set all fields, including dynamic ones
        print(f"[DEBUG] Processing data_copy items: {data_copy}")
        for k, v in data_copy.items():
            if k not in fields and not k.startswith('_'):
                print(f"[DEBUG] Setting dynamic attribute: {k} = {v}")
                # Store in _extra_fields and set as attribute
                print(f"[DEBUG] Before setting - _extra_fields: {getattr(instance, '_extra_fields', 'NOT FOUND')}")
                print(f"[DEBUG] Before setting - instance.__dict__: {instance.__dict__}")
                
                # Initialize _extra_fields if it doesn't exist
                if not hasattr(instance, '_extra_fields'):
                    print("[DEBUG] Initializing _extra_fields in from_dict")
                    object.__setattr__(instance, '_extra_fields', {})
                
                # Store in _extra_fields
                instance._extra_fields[k] = v
                print(f"[DEBUG] After setting _extra_fields: {instance._extra_fields}")
                
                # Also set as direct attribute
                object.__setattr__(instance, k, v)
                print(f"[DEBUG] After setting {k}, _extra_fields: {getattr(instance, '_extra_fields', 'NOT FOUND')}")
                print(f"[DEBUG] After setting {k}, __dict__: {instance.__dict__}")
                print(f"[DEBUG] Has {k} in __dict__: {k in instance.__dict__}")
                print(f"[DEBUG] Has {k} in _extra_fields: {k in getattr(instance, '_extra_fields', {})}")
        
        print(f"[DEBUG] Final instance state - _extra_fields: {getattr(instance, '_extra_fields', 'NOT FOUND')}")
        print(f"[DEBUG] Final instance state - __dict__: {instance.__dict__}")
        
        # Preserve original version if it was provided
        if original_version is not None and hasattr(instance, '_original_version'):
            instance._original_version = original_version
            
        return instance

    def update_from_dict(self, data: Dict[str, Any]) -> "BaseMedicalConfig":
        """Update the configuration from a dictionary.

        Args:
            data: Dictionary containing configuration parameters

        Returns:
            The updated instance
        """
        for key, value in data.items():
            setattr(self, key, value)
        return self

    def copy(self) -> "BaseMedicalConfig":
        """Create a deep copy of the configuration.
        
        Returns:
            A new BaseMedicalConfig instance with the same values
        """
        import copy
        
        # Get the current state as a dictionary
        state = self.to_dict()
        
        # Ensure required parameters are present
        if 'model' not in state:
            state['model'] = 'default-model'
            
        # Create a new instance using from_dict to ensure proper initialization
        new_instance = self.__class__.from_dict(state)
        
        # Ensure _extra_fields exists in the new instance
        if not hasattr(new_instance, '_extra_fields'):
            new_instance._extra_fields = {}
            
        # Copy any additional attributes and _extra_fields
        for key, value in self.__dict__.items():
            if key == '_extra_fields' and isinstance(value, dict):
                # Copy all dynamic attributes from _extra_fields
                for k, v in value.items():
                    new_instance._extra_fields[k] = copy.deepcopy(v)
                    # Also set as an attribute for direct access
                    setattr(new_instance, k, copy.deepcopy(v))
            elif not key.startswith('_') and not hasattr(new_instance, key):
                # Copy other non-private attributes
                setattr(new_instance, key, copy.deepcopy(value))
                
        return new_instance

    def validate(self) -> None:
        """Validate the configuration."""
        # Check required fields
        required_fields = ["model_type"]
        missing_fields = []
        for field_name in required_fields:
            if not hasattr(self, field_name) or getattr(self, field_name) is None:
                missing_fields.append(field_name)
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        # Try to set up hf_config if not already set
        if (
            hasattr(self, "model")
            and hasattr(self, "hf_config")
            and self.hf_config is None
        ):
            try:
                from transformers import AutoConfig

                self.hf_config = AutoConfig.from_pretrained(self.model)
                # Set max_model_len from model config if available
                if hasattr(self, "hf_config") and self.hf_config is not None:
                    try:
                        # Use getattr with a default to safely access the attribute
                        max_pos_embeddings = getattr(
                            self.hf_config, "max_position_embeddings", None
                        )
                        if max_pos_embeddings is not None and hasattr(
                            self, "max_model_len"
                        ):
                            self.max_model_len = min(
                                self.max_model_len, max_pos_embeddings
                            )
                    except (AttributeError, TypeError, ValueError) as e:
                        # Log the error but don't fail
                        logger.debug(
                            "Could not set max_model_len from hf_config: %s",
                            str(e),
                            exc_info=True,
                        )
            except (AttributeError, TypeError, ValueError) as e:
                # Skip setting max_model_len if we can't access the config attributes
                logger.debug(
                    "Could not set max_model_len from hf_config: %s",
                    str(e),
                    exc_info=True,
                )

        # Call custom validation if implemented
        if hasattr(self, "_validate_custom"):
            self._validate_custom()
