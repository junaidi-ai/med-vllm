from dataclasses import dataclass, field
from typing import Optional, Dict, List, Union
from pathlib import Path
from ..config import Config

@dataclass
class MedicalModelConfig(Config):
    """
    Extended configuration class for medical language models.
    
    This class extends the base Config class with medical domain-specific parameters
    and functionality for handling medical NLP tasks.
    """
    # Model type specification (e.g., 'biobert', 'clinicalbert', 'bioclinicalbert')
    model_type: str = "biobert"
    
    # Medical domain specific parameters
    max_medical_seq_length: int = 512
    
    # Task-specific parameters
    num_medical_labels: int = 0  # Will be set based on the task
    
    # Model head configuration
    use_crf: bool = False  # Whether to use CRF layer for sequence tagging
    
    # Medical tokenizer settings
    do_lower_case: bool = True
    
    # Domain adaptation parameters
    domain_adaptation: bool = False
    domain_adaptation_lambda: float = 1.0
    
    # Pretrained model paths
    pretrained_model_name_or_path: Optional[str] = None
    
    # Medical vocabulary settings
    medical_vocab_file: Optional[str] = None
    
    # Model-specific parameters
    dropout_prob: float = 0.1
    
    def __post_init__(self):
        # First call parent's __post_init__
        super().__post_init__()
        
        # Medical model specific validations
        assert self.model_type in ["biobert", "clinicalbert", "bioclinicalbert"], \
            f"Unsupported model type: {self.model_type}"
            
        # Set default pretrained paths if not specified
        if self.pretrained_model_name_or_path is None:
            if self.model_type == "biobert":
                self.pretrained_model_name_or_path = "dmis-lab/biobert-v1.1"
            elif self.model_type == "clinicalbert":
                self.pretrained_model_name_or_path = "emilyalsentzer/Bio_ClinicalBERT"
            elif self.model_type == "bioclinicalbert":
                self.pretrained_model_name_or_path = "emilyalsentzer/Bio_ClinicalBERT"
        
        # Update model path to use the pretrained model if not specified
        if not Path(self.model).exists() and self.pretrained_model_name_or_path:
            self.model = self.pretrained_model_name_or_path

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """
        Create a MedicalModelConfig from a pretrained model.
        
        Args:
            model_name_or_path: Name or path of the pretrained model
            **kwargs: Additional arguments to override the default configuration
            
        Returns:
            MedicalModelConfig: Configured instance
        """
        # Determine model type from name if not specified
        if "model_type" not in kwargs:
            if "biobert" in model_name_or_path.lower():
                kwargs["model_type"] = "biobert"
            elif "clinical" in model_name_or_path.lower():
                kwargs["model_type"] = "clinicalbert"
        
        # Create config with the model path and any overrides
        config = cls(model=model_name_or_path, **kwargs)
        return config

    def to_dict(self) -> Dict:
        """Convert configuration to a dictionary."""
        output = super().to_dict()
        # Add medical-specific fields
        medical_fields = {
            "model_type": self.model_type,
            "max_medical_seq_length": self.max_medical_seq_length,
            "num_medical_labels": self.num_medical_labels,
            "use_crf": self.use_crf,
            "do_lower_case": self.do_lower_case,
            "domain_adaptation": self.domain_adaptation,
            "domain_adaptation_lambda": self.domain_adaptation_lambda,
            "pretrained_model_name_or_path": self.pretrained_model_name_or_path,
            "medical_vocab_file": self.medical_vocab_file,
            "dropout_prob": self.dropout_prob,
        }
        output.update(medical_fields)
        return output
