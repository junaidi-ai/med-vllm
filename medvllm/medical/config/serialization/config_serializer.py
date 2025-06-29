"""
Base serializer for medical model configurations.
"""

from typing import Any, Dict, Optional, Type, TypeVar, Union

from ..base import BaseMedicalConfig

T = TypeVar('T', bound='MedicalModelConfig')

class ConfigSerializer:
    """Base class for configuration serializers."""
    
    @classmethod
    def to_dict(cls, config: BaseMedicalConfig) -> Dict[str, Any]:
        """Convert configuration to a dictionary.
        
        Args:
            config: Configuration to serialize
            
        Returns:
            Dictionary representation of the configuration
        """
        output = {}

        # Get base config parameters
        if hasattr(super(BaseMedicalConfig, config), "to_dict"):
            base_dict = super(BaseMedicalConfig, config).to_dict()
            if isinstance(base_dict, dict):
                output.update(base_dict)

        # Add medical-specific fields
        medical_fields = {
            "config_version": getattr(config, "config_version", None),
            "model": getattr(config, "model", None),
            "model_type": getattr(config, "model_type", None),
            "medical_specialties": getattr(config, "medical_specialties", None),
            "anatomical_regions": getattr(config, "anatomical_regions", None),
            "imaging_modalities": getattr(config, "imaging_modalities", None),
            "clinical_metrics": getattr(config, "clinical_metrics", None),
            "regulatory_compliance": getattr(config, "regulatory_compliance", None),
            "use_crf": getattr(config, "use_crf", None),
            "do_lower_case": getattr(config, "do_lower_case", None),
            "preserve_case_for_abbreviations": getattr(
                config, "preserve_case_for_abbreviations", None
            ),
            "domain_adaptation": getattr(config, "domain_adaptation", None),
            "domain_adaptation_lambda": getattr(
                config, "domain_adaptation_lambda", None
            ),
            "entity_types": getattr(config, "entity_types", None),
            "document_types": getattr(config, "document_types", None),
            "section_headers": getattr(config, "section_headers", None),
            "knowledge_bases": getattr(config, "knowledge_bases", None),
            "uncertainty_threshold": getattr(config, "uncertainty_threshold", None),
            "max_entity_span": getattr(config, "max_entity_span", None),
            "max_retries": getattr(config, "max_retries", None),
            "request_timeout": getattr(config, "request_timeout", None),
        }
        
        # Add non-None fields to output
        for key, value in medical_fields.items():
            if value is not None:
                output[key] = value
                
        return output
    
    @classmethod
    def from_dict(
        cls, 
        config_dict: Dict[str, Any], 
        config_class: Type[T],
        **kwargs: Any
    ) -> T:
        """Create a configuration object from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            config_class: Configuration class to instantiate
            **kwargs: Additional keyword arguments to pass to the config class
            
        Returns:
            Instantiated configuration object
        """
        # Create a copy to avoid modifying the input
        config_dict = config_dict.copy()
        
        # Remove any None values to use defaults
        config_dict = {k: v for k, v in config_dict.items() if v is not None}
        
        # Update with any additional keyword arguments
        config_dict.update(kwargs)
        
        return config_class(**config_dict)
