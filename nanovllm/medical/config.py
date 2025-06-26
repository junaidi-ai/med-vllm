import os
import json
from dataclasses import dataclass, field, fields
from typing import Optional, Dict, List, Union, Any, Tuple, Type, TypeVar
from pathlib import Path
from ..config import Config

# Type variable for the class itself (for type hints in class methods)
T = TypeVar('T', bound='MedicalModelConfig')

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
    
    # Medical specialties and domains
    medical_specialties: List[str] = field(
        default_factory=lambda: [
            # Primary Care
            'family_medicine', 'internal_medicine', 'pediatrics', 'geriatrics',
            'obstetrics_gynecology', 'preventive_medicine',
            
            # Medical Specialties
            'allergy_immunology', 'anesthesiology', 'cardiology', 'dermatology',
            'endocrinology', 'gastroenterology', 'hematology', 'infectious_disease',
            'nephrology', 'pulmonology', 'rheumatology',
            
            # Surgical Specialties
            'general_surgery', 'cardiac_surgery', 'neurosurgery', 'orthopedic_surgery',
            'plastic_surgery', 'thoracic_surgery', 'transplant_surgery', 'urology',
            'vascular_surgery',
            
            # Diagnostic Specialties
            'pathology', 'clinical_pathology', 'anatomic_pathology', 'radiology',
            'diagnostic_radiology', 'interventional_radiology', 'nuclear_medicine',
            
            # Hospital-based Specialties
            'emergency_medicine', 'critical_care_medicine', 'hospital_medicine',
            'palliative_care', 'pain_medicine', 'sleep_medicine',
            
            # Mental Health
            'psychiatry', 'child_psychiatry', 'addiction_psychiatry',
            'forensic_psychiatry', 'geriatric_psychiatry',
            
            # Other Specialties
            'dermatology', 'neurology', 'neurosurgery', 'ophthalmology',
            'otolaryngology', 'physical_medicine_rehab', 'radiation_oncology',
            'reproductive_endocrinology', 'sports_medicine', 'wound_care'
        ]
    )
    
    # Anatomical regions for NER and other tasks, organized by body systems
    anatomical_regions: List[str] = field(
        default_factory=lambda: [
            # Head and Neck
            'head', 'skull', 'face', 'forehead', 'temple', 'scalp',
            'eye', 'eyebrow', 'eyelid', 'conjunctiva', 'cornea', 'retina',
            'ear', 'auricle', 'external_auditory_canal', 'tympanic_membrane',
            'nose', 'nasal_cavity', 'paranasal_sinuses',
            'mouth', 'oral_cavity', 'lips', 'tongue', 'palate', 'pharynx',
            'neck', 'larynx', 'thyroid', 'parathyroid', 'trachea',
            
            # Thorax (Chest)
            'thorax', 'chest', 'thoracic_wall', 'ribs', 'sternum',
            'pleura', 'pleural_cavity', 'mediastinum',
            'lungs', 'bronchi', 'bronchioles', 'alveoli',
            'heart', 'pericardium', 'myocardium', 'endocardium',
            'esophagus', 'thymus',
            
            # Abdomen and Pelvis
            'abdomen', 'abdominal_wall', 'peritoneum', 'peritoneal_cavity',
            'stomach', 'small_intestine', 'duodenum', 'jejunum', 'ileum',
            'large_intestine', 'cecum', 'appendix', 'colon', 'rectum', 'anus',
            'liver', 'gallbladder', 'biliary_tract',
            'pancreas', 'spleen', 'kidneys', 'ureters',
            'pelvis', 'pelvic_cavity', 'urinary_bladder', 'urethra',
            'prostate', 'seminal_vesicles', 'testes', 'epididymis',
            'ovaries', 'fallopian_tubes', 'uterus', 'cervix', 'vagina', 'vulva',
            
            # Back and Spine
            'back', 'spine', 'vertebral_column', 'cervical_spine', 'thoracic_spine',
            'lumbar_spine', 'sacrum', 'coccyx', 'intervertebral_discs',
            'spinal_cord', 'meninges', 'cauda_equina',
            
            # Upper Limbs
            'upper_limb', 'shoulder', 'axilla', 'arm', 'upper_arm', 'humerus',
            'elbow', 'forearm', 'radius', 'ulna', 'wrist', 'hand',
            'carpal_bones', 'metacarpals', 'phalanges', 'fingers', 'thumb',
            
            # Lower Limbs
            'lower_limb', 'hip', 'thigh', 'femur', 'knee', 'patella',
            'leg', 'tibia', 'fibula', 'ankle', 'foot',
            'tarsal_bones', 'calcaneus', 'talus', 'metatarsals', 'toes',
            
            # Other
            'skin', 'subcutaneous_tissue', 'fascia', 'muscles', 'tendons',
            'ligaments', 'joints', 'bursae', 'nerves', 'blood_vessels',
            'lymph_nodes', 'lymphatic_vessels',
            
            # General/Whole Body
            'whole_body', 'bilateral', 'unilateral', 'proximal', 'distal',
            'anterior', 'posterior', 'medial', 'lateral', 'superior', 'inferior'
        ]
    )
    
    # Medical imaging modalities
    imaging_modalities: List[str] = field(
        default_factory=lambda: [
            'xray', 'ct', 'mri', 'ultrasound', 'pet', 'mammography',
            'fluoroscopy', 'angiography'
        ]
    )
    
    # Clinical metrics including vital signs, lab tests, and clinical scores
    clinical_metrics: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            'vital_signs': {
                'temperature': {'unit': '°C', 'normal_range': (36.1, 37.2), 'category': 'general'},
                'blood_pressure_systolic': {'unit': 'mmHg', 'normal_range': (90, 120), 'category': 'cardiovascular'},
                'blood_pressure_diastolic': {'unit': 'mmHg', 'normal_range': (60, 80), 'category': 'cardiovascular'},
                'heart_rate': {'unit': 'bpm', 'normal_range': (60, 100), 'category': 'cardiovascular'},
                'respiratory_rate': {'unit': 'breaths/min', 'normal_range': (12, 20), 'category': 'respiratory'},
                'oxygen_saturation': {'unit': '%', 'normal_range': (95, 100), 'category': 'respiratory'},
                'pain_score': {'unit': '0-10', 'normal_range': (0, 3), 'category': 'general'},
                'height': {'unit': 'cm', 'normal_range': (150, 190), 'category': 'anthropometric'},
                'weight': {'unit': 'kg', 'normal_range': (50, 100), 'category': 'anthropometric'},
                'bmi': {'unit': 'kg/m²', 'normal_range': (18.5, 24.9), 'category': 'anthropometric'},
                'blood_glucose': {'unit': 'mg/dL', 'normal_range': (70, 140), 'category': 'metabolic'},
                'gcs': {'unit': '3-15', 'normal_range': (15, 15), 'category': 'neurological'},
                'capillary_refill': {'unit': 'seconds', 'normal_range': (0, 2), 'category': 'cardiovascular'}
            },
            'lab_tests': {
                # Hematology
                'hemoglobin': {'unit': 'g/dL', 'normal_range': (12.0, 16.0), 'category': 'hematology'},
                'hematocrit': {'unit': '%', 'normal_range': (36, 48), 'category': 'hematology'},
                'wbc_count': {'unit': '10³/µL', 'normal_range': (4.5, 11.0), 'category': 'hematology'},
                'platelet_count': {'unit': '10³/µL', 'normal_range': (150, 450), 'category': 'hematology'},
                
                # Chemistry
                'sodium': {'unit': 'mEq/L', 'normal_range': (135, 145), 'category': 'electrolytes'},
                'potassium': {'unit': 'mEq/L', 'normal_range': (3.5, 5.1), 'category': 'electrolytes'},
                'creatinine': {'unit': 'mg/dL', 'normal_range': (0.6, 1.2), 'category': 'renal'},
                'bun': {'unit': 'mg/dL', 'normal_range': (7, 20), 'category': 'renal'},
                'glucose': {'unit': 'mg/dL', 'normal_range': (70, 100), 'category': 'metabolic'},
                'calcium': {'unit': 'mg/dL', 'normal_range': (8.5, 10.2), 'category': 'electrolytes'},
                
                # Liver Function
                'ast': {'unit': 'U/L', 'normal_range': (10, 40), 'category': 'liver'},
                'alt': {'unit': 'U/L', 'normal_range': (7, 56), 'category': 'liver'},
                'alkaline_phosphatase': {'unit': 'U/L', 'normal_range': (44, 147), 'category': 'liver'},
                'bilirubin_total': {'unit': 'mg/dL', 'normal_range': (0.3, 1.2), 'category': 'liver'},
                'albumin': {'unit': 'g/dL', 'normal_range': (3.5, 5.0), 'category': 'liver'},
                
                # Cardiac Markers
                'troponin': {'unit': 'ng/mL', 'normal_range': (0, 0.04), 'category': 'cardiac'},
                'ck_mb': {'unit': 'ng/mL', 'normal_range': (0, 5), 'category': 'cardiac'},
                'bnp': {'unit': 'pg/mL', 'normal_range': (0, 100), 'category': 'cardiac'},
                
                # Coagulation
                'pt': {'unit': 'seconds', 'normal_range': (11, 13.5), 'category': 'coagulation'},
                'inr': {'unit': 'ratio', 'normal_range': (0.9, 1.1), 'category': 'coagulation'},
                'ptt': {'unit': 'seconds', 'normal_range': (25, 35), 'category': 'coagulation'},
                'd_dimer': {'unit': 'µg/mL', 'normal_range': (0, 0.5), 'category': 'coagulation'}
            },
            'scores': {
                # Critical Care
                'apache_ii': {'range': (0, 71), 'higher_worse': True, 'category': 'critical_care'},
                'saps_ii': {'range': (0, 163), 'higher_worse': True, 'category': 'critical_care'},
                'sofa': {'range': (0, 24), 'higher_worse': True, 'category': 'critical_care'},
                
                # Sepsis
                'qsofa': {'range': (0, 3), 'higher_worse': True, 'category': 'sepsis'},
                'sirs': {'range': (0, 4), 'higher_worse': True, 'category': 'sepsis'},
                
                # Pain
                'visual_analog_scale': {'range': (0, 10), 'higher_worse': True, 'category': 'pain'},
                'numeric_rating_scale': {'range': (0, 10), 'higher_worse': True, 'category': 'pain'},
                
                # Functional Status
                'karnofsky': {'range': (0, 100), 'higher_worse': False, 'category': 'functional'},
                'ecog': {'range': (0, 5), 'higher_worse': True, 'category': 'functional'},
                
                # Psychiatric
                'phq9': {'range': (0, 27), 'higher_worse': True, 'category': 'psychiatric'},
                'gad7': {'range': (0, 21), 'higher_worse': True, 'category': 'psychiatric'},
                'mmse': {'range': (0, 30), 'higher_worse': False, 'category': 'neurological'}
            }
        }
    )
    
    # Regulatory compliance flags
    regulatory_compliance: Dict[str, bool] = field(
        default_factory=lambda: {
            'hipaa_compliant': True,
            'gdpr_compliant': True,
            'fda_cleared': False,
            'ce_marked': False,
            'hippa_compliant': True
        }
    )
    
    # Task-specific parameters
    num_medical_labels: int = 0  # Will be set based on the task
    
    # Model head configuration
    use_crf: bool = False  # Whether to use CRF layer for sequence tagging
    
    # Medical tokenizer settings
    do_lower_case: bool = True
    preserve_case_for_abbreviations: bool = True
    
    # Domain adaptation parameters
    domain_adaptation: bool = False
    domain_adaptation_lambda: float = 1.0
    domain_specific_vocab: bool = True
    
    # NER and Entity Recognition
    ner_confidence_threshold: float = 0.7  # Minimum confidence score for entity recognition
    max_entity_span_length: int = 10  # Maximum number of tokens in a single entity
    entity_linking_enabled: bool = True  # Whether to link entities to knowledge bases
    entity_linking_knowledge_bases: List[str] = field(
        default_factory=lambda: ['UMLS', 'SNOMED_CT', 'ICD10', 'RxNorm', 'LOINC']
    )
    
    # Clinical Document Processing
    document_types: List[str] = field(
        default_factory=lambda: [
            'progress_note', 'discharge_summary', 'consult_note', 'radiology_report',
            'pathology_report', 'operative_note', 'emergency_note', 'admission_note'
        ]
    )
    section_headers: Dict[str, List[str]] = field(
        default_factory=lambda: {
            'history': ['history of present illness', 'hpi', 'history'],
            'examination': ['physical exam', 'examination', 'pe'],
            'assessment': ['assessment', 'impression', 'diagnosis'],
            'plan': ['plan', 'recommendations', 'treatment plan']
        }
    )
    
    # Model Behavior
    enable_uncertainty_estimation: bool = True
    uncertainty_threshold: float = 0.3
    max_retries: int = 3
    request_timeout: int = 30  # seconds
    batch_size: int = 32
    
    # Performance Optimization
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour in seconds
    max_cache_size: int = 10000  # Maximum number of items to cache
    
    # Pretrained model paths
    pretrained_model_name_or_path: Optional[str] = None
    
    # Medical vocabulary settings
    medical_vocab_file: Optional[str] = None
    medical_entity_types: List[str] = field(
        default_factory=lambda: [
            'disease', 'drug', 'procedure', 'anatomy', 'symptom',
            'lab_value', 'vital_sign', 'device', 'biomarker', 'gene',
            'mutation', 'cell_type', 'cell_line', 'dna', 'rna', 'protein',
            'pathogen', 'tumor', 'tumor_marker', 'clinical_trial', 'age',
            'gender', 'race', 'ethnicity', 'family_history', 'social_history'
        ]
    )
    
    # Model-specific parameters
    dropout_prob: float = 0.1
    
    # Clinical context window
    context_window_size: int = 3  # Number of sentences to consider as context
    
    # Privacy and security
    deidentify_text: bool = True
    deidentification_method: str = 'ner_based'  # Options: 'ner_based', 'rule_based', 'hybrid'
    
    # Clinical validation
    require_clinical_validation: bool = True
    validation_frequency: int = 1000  # Number of steps between validations
    
    def __post_init__(self):
        """
        Initialize the configuration and validate all parameters.

        This method is automatically called after the dataclass is initialized.
        It performs validation of all configuration parameters and sets default values.

        Note:
            We call the parent's __post_init__ first to ensure base validation runs.
            Then we apply medical-specific validation and defaults.
        """
        # Store original model path before parent validation might modify it
        original_model = getattr(self, 'model', None)

        try:
            # Validate tensor_parallel_size first to provide better error messages
            if hasattr(self, 'tensor_parallel_size') and self.tensor_parallel_size is not None:
                if not (1 <= self.tensor_parallel_size <= 8):
                    raise ValueError(
                        f"tensor_parallel_size must be between 1 and 8, got {self.tensor_parallel_size}"
                    )
            
            # Call parent's __post_init__ for base validation
            super().__post_init__()
                
        except Exception as e:
            # If validation fails but we have a pretrained model, try with that
            if hasattr(self, 'pretrained_model_name_or_path') and self.pretrained_model_name_or_path:
                original_model = self.model  # Save the original model path
                try:
                    self.model = self.pretrained_model_name_or_path
                    # Re-validate tensor_parallel_size for the pretrained model
                    if hasattr(self, 'tensor_parallel_size') and self.tensor_parallel_size is not None:
                        if not (1 <= self.tensor_parallel_size <= 8):
                            raise ValueError(
                                f"tensor_parallel_size must be between 1 and 8, got {self.tensor_parallel_size}"
                            )
                    super().__post_init__()  # Call the parent's __post_init__ with parentheses
                except Exception as e2:
                    self.model = original_model  # Restore original model path on failure
                    if 'tensor_parallel_size' in str(e2).lower():
                        raise ValueError(f"Invalid tensor_parallel_size: {e2}")
                    raise ValueError(f"Invalid configuration with pretrained model: {str(e2)}")
            else:
                if 'tensor_parallel_size' in str(e).lower():
                    raise ValueError(f"Invalid tensor_parallel_size: {e}")
                raise ValueError(f"Invalid configuration: {str(e)}. Please provide a valid model path or set pretrained_model_name_or_path.")
        
        # Restore original model path if it was set and different
        if original_model and original_model != self.model:
            self.model = original_model
            
        # Apply medical-specific validations
        self._validate_model_type()
        self._validate_medical_parameters()
        self._validate_entity_linking()
        self._set_default_pretrained_paths()
        
        # Update model path to use the pretrained model if specified and local path doesn't exist
        if not Path(str(self.model)).exists() and hasattr(self, 'pretrained_model_name_or_path') and self.pretrained_model_name_or_path:
            self.model = self.pretrained_model_name_or_path
    
    def _validate_model_type(self) -> None:
        """
        Validate the model type.
        
        Raises:
            ValueError: If the model type is not supported
        """
        supported_models = [
            'biobert', 'clinicalbert', 'bioclinicalbert', 'bluebert', 'pubmedbert',
            'clinicalcovidbert', 'scibert', 'biomed_roberta', 'gpt2_medical',
            'gpt_neo_medical', 'gptj_medical', 'gpt_neox_medical', 'llama_medical'
        ]
        
        if self.model_type.lower() not in supported_models:
            raise ValueError(
                f"Unsupported model type: {self.model_type}. "
                f"Supported types are: {', '.join(supported_models)}"
            )
    
    def _validate_medical_parameters(self) -> None:
        """
        Validate medical-specific parameters.
        
        Raises:
            ValueError: If any medical parameter fails validation
        """
        # Validate medical specialties
        if not self.medical_specialties:
            raise ValueError("At least one medical specialty must be specified")
            
        # Validate anatomical regions
        if not self.anatomical_regions:
            raise ValueError("At least one anatomical region must be specified")
            
        # Validate NER parameters
        if not 0 <= self.ner_confidence_threshold <= 1.0:
            raise ValueError("ner_confidence_threshold must be between 0 and 1")
            
        if self.max_entity_span_length <= 0:
            raise ValueError("max_entity_span_length must be positive")
            
        # Validate document processing
        if not self.document_types:
            raise ValueError("At least one document type must be specified")
            
        if not self.section_headers:
            raise ValueError("Section headers must be specified")
            
        # Validate model behavior
        if not 0 <= self.uncertainty_threshold <= 1.0:
            raise ValueError("uncertainty_threshold must be between 0 and 1")
            
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
            
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
            
        # Validate performance settings
        if self.enable_caching and self.cache_ttl <= 0:
            raise ValueError("cache_ttl must be positive when caching is enabled")
            
        if self.enable_caching and self.max_cache_size <= 0:
            raise ValueError("max_cache_size must be positive when caching is enabled")
            
        # Validate sequence length
        if not 64 <= self.max_medical_seq_length <= 4096:
            raise ValueError("max_medical_seq_length must be between 64 and 4096")
            
        # Validate context window size
        if not 0 <= self.context_window_size <= 10:
            raise ValueError("context_window_size must be between 0 and 10")
            
        # Validate deidentification method
        valid_deid_methods = ['ner_based', 'rule_based', 'hybrid']
        if self.deidentification_method not in valid_deid_methods:
            raise ValueError(
                f"Invalid deidentification_method: {self.deidentification_method}. "
                f"Must be one of: {', '.join(valid_deid_methods)}"
            )
            
        # Validate clinical metrics structure
        for metric_type, metrics in self.clinical_metrics.items():
            if not isinstance(metrics, dict):
                raise ValueError(f"{metric_type} in clinical_metrics must be a dictionary")
                
        # Validate regulatory compliance flags
        for flag_name, flag_value in self.regulatory_compliance.items():
            if not isinstance(flag_value, bool):
                raise ValueError(f"Regulatory flag {flag_name} must be a boolean")
    
    def _set_default_pretrained_paths(self) -> None:
        """
        Set default pretrained model paths based on model type.
        
        Sets appropriate default model paths for common medical model types if not specified.
        """
        if not self.pretrained_model_name_or_path:
            if self.model_type == 'biobert':
                self.pretrained_model_name_or_path = 'dmis-lab/biobert-v1.1'
            elif self.model_type == 'clinicalbert':
                self.pretrained_model_name_or_path = 'emilyalsentzer/Bio_ClinicalBERT'
                
    def _validate_entity_linking(self) -> None:
        """
        Validate entity linking configuration.
        
        Raises:
            ValueError: If entity linking is enabled but no knowledge bases are specified
        """
        if self.entity_linking_enabled and not self.entity_linking_knowledge_bases:
            raise ValueError("At least one knowledge base must be specified when entity linking is enabled")

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """
        Create a MedicalModelConfig from a pretrained model.
        
        Args:
            model_name_or_path: Name or path of the pretrained model
            **kwargs: Additional arguments to override the default configuration
            
        Returns:
            MedicalModelConfig: Configured instance
            
        Raises:
            ValueError: If the configuration file is not found or is invalid
        """
        # If the path exists and is a directory, try to load config from it
        if os.path.isdir(model_name_or_path):
            config_path = os.path.join(model_name_or_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config_dict = json.load(f)
                # Update with any explicit kwargs
                config_dict.update(kwargs)
                return cls.from_dict(config_dict)
        
        # If not loading from a directory or config not found, create new config
        # Determine model type from name if not specified
        if "model_type" not in kwargs:
            if "biobert" in model_name_or_path.lower():
                kwargs["model_type"] = "biobert"
            elif "clinical" in model_name_or_path.lower():
                kwargs["model_type"] = "clinicalbert"
        
        # Create config with the model path and any overrides
        config = cls(model=model_name_or_path, **kwargs)
        return config
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MedicalModelConfig':
        """
        Create a MedicalModelConfig from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            MedicalModelConfig: Configured instance
            
        Note:
            Handles both base Config parameters and medical-specific parameters.
            Preserves backward compatibility with the base Config class.
            
        Raises:
            ValueError: If required parameters are missing or invalid
        """
        if not isinstance(config_dict, dict):
            raise ValueError("Input must be a dictionary")
            
        # Create a copy to avoid modifying the input
        config_dict = config_dict.copy()
        
        # Extract base config parameters that are valid for the parent class
        base_config = {}
        base_fields = {f.name for f in fields(Config) if f.init}
        
        for field_name in base_fields:
            if field_name in config_dict:
                base_config[field_name] = config_dict.pop(field_name)
        
        # Handle model path - prioritize pretrained model if specified
        if 'pretrained_model_name_or_path' in config_dict and not base_config.get('model'):
            base_config['model'] = config_dict.pop('pretrained_model_name_or_path')
        
        # Create instance with base config
        try:
            if base_config:
                config = cls(**base_config)
            else:
                # If no base config provided, use defaults
                config = cls(model=config_dict.get('model', ''))
        except Exception as e:
            raise ValueError(f"Failed to initialize base config: {str(e)}")
        
        # Update with remaining parameters (medical-specific)
        for key, value in config_dict.items():
            # Only set attributes that are defined in the class
            if hasattr(config, key) or key in cls.__annotations__:
                try:
                    setattr(config, key, value)
                except Exception as e:
                    import warnings
                    warnings.warn(f"Could not set {key}: {str(e)}")
        
        # Re-validate the configuration
        try:
            config.__post_init__()
        except Exception as e:
            import warnings
            warnings.warn(f"Configuration validation warning: {str(e)}")
                
        return config
        
    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the configuration to a directory.
        
        Args:
            save_directory: Directory to save the configuration to
            
        Raises:
            ValueError: If the directory cannot be created or is not writable
        """
        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)
        
        # Check if directory is writable
        if not os.access(save_directory, os.W_OK):
            raise ValueError(f"Directory {save_directory} is not writable")
        
        # Save config as JSON
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
            
        # If there's a medical vocab file, copy it to the target directory
        if self.medical_vocab_file and os.path.isfile(self.medical_vocab_file):
            import shutil
            vocab_filename = os.path.basename(self.medical_vocab_file)
            target_path = os.path.join(save_directory, vocab_filename)
            if not os.path.exists(target_path):
                shutil.copy2(self.medical_vocab_file, target_path)
            self.medical_vocab_file = vocab_filename

    def to_dict(self) -> Dict:
        """
        Convert configuration to a dictionary.
        
        Returns:
            Dict: A dictionary containing all configuration parameters
            
        Note:
            First gets all base class parameters using super().to_dict(),
            then updates with medical-specific parameters.
            This ensures we don't miss any base class parameters.
        """
        # Get base config parameters
        output = {}
        try:
            # Try to get base class dict if available
            if hasattr(super(), 'to_dict'):
                output = super().to_dict()
            else:
                # Fallback: Get all fields from base class
                base_fields = {}
                for field in fields(Config):
                    if hasattr(self, field.name):
                        base_fields[field.name] = getattr(self, field.name)
                output.update(base_fields)
        except Exception as e:
            # If base class to_dict fails, collect fields manually
            import traceback
            print(f"Warning: Could not get base config: {str(e)}\n{traceback.format_exc()}")
            
        # Add medical-specific fields
        medical_fields = {
            "model_type": self.model_type,
            "max_medical_seq_length": self.max_medical_seq_length,
            "medical_specialties": self.medical_specialties,
            "anatomical_regions": self.anatomical_regions,
            "imaging_modalities": self.imaging_modalities,
            "clinical_metrics": self.clinical_metrics,
            "regulatory_compliance": self.regulatory_compliance,
            "num_medical_labels": self.num_medical_labels,
            "use_crf": self.use_crf,
            "do_lower_case": self.do_lower_case,
            "preserve_case_for_abbreviations": self.preserve_case_for_abbreviations,
            "domain_adaptation": self.domain_adaptation,
            "domain_adaptation_lambda": self.domain_adaptation_lambda,
            "domain_specific_vocab": self.domain_specific_vocab,
            "pretrained_model_name_or_path": self.pretrained_model_name_or_path,
            "medical_vocab_file": self.medical_vocab_file,
            "medical_entity_types": self.medical_entity_types,
            "ner_confidence_threshold": self.ner_confidence_threshold,
            "max_entity_span_length": self.max_entity_span_length,
            "entity_linking_enabled": self.entity_linking_enabled,
            "entity_linking_knowledge_bases": self.entity_linking_knowledge_bases,
            "document_types": self.document_types,
            "section_headers": self.section_headers,
            "enable_uncertainty_estimation": self.enable_uncertainty_estimation,
            "uncertainty_threshold": self.uncertainty_threshold,
            "max_retries": self.max_retries,
            "request_timeout": self.request_timeout,
            "batch_size": self.batch_size,
            "enable_caching": self.enable_caching,
            "cache_ttl": self.cache_ttl,
            "max_cache_size": self.max_cache_size,
        }
        
        # Only update with non-None values to avoid overwriting valid base config
        for k, v in medical_fields.items():
            if v is not None:
                output[k] = v
                
        return output
        
    def to_json(self, file_path: Optional[str] = None, indent: int = 2) -> Optional[str]:
        """
        Convert configuration to a JSON string or file.
        
        Args:
            file_path: If provided, saves the JSON to this file
            indent: Number of spaces for indentation in the output
            
        Returns:
            Optional[str]: JSON string if file_path is None, else None
            
        Raises:
            ValueError: If the file cannot be written
        """
        def json_serializable(obj):
            """Convert objects to JSON serializable format."""
            if isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            elif isinstance(obj, (list, tuple)):
                return [json_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {str(k): json_serializable(v) for k, v in obj.items()}
            elif hasattr(obj, 'to_json_string'):
                return json.loads(obj.to_json_string())
            elif hasattr(obj, '__dict__'):
                return json_serializable(obj.__dict__)
            else:
                return str(obj)
        
        config_dict = self.to_dict()
        serializable_dict = json_serializable(config_dict)
        
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_dict, f, indent=indent, ensure_ascii=False)
            return None
        else:
            return json.dumps(serializable_dict, indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_input: Union[str, os.PathLike, Dict]) -> 'MedicalModelConfig':
        """
        Create a MedicalModelConfig from a JSON string, file, or dictionary.
        
        Args:
            json_input: JSON string, path to a JSON file, or a dictionary
            
        Returns:
            MedicalModelConfig: A new instance configured from the input
            
        Raises:
            ValueError: If the input cannot be parsed or is invalid
            TypeError: If the input type is not supported
        """
        if isinstance(json_input, dict):
            config_dict = json_input
        elif isinstance(json_input, (str, os.PathLike)):
            if os.path.isfile(json_input):
                try:
                    with open(json_input, 'r', encoding='utf-8') as f:
                        config_dict = json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    raise ValueError(f"Failed to load configuration from {json_input}: {str(e)}")
            else:
                try:
                    config_dict = json.loads(json_input)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON string: {str(e)}")
        else:
            raise TypeError("Input must be a JSON string, file path, or dictionary")
            
        if not isinstance(config_dict, dict):
            raise ValueError("JSON must contain an object at the top level")
            
        return cls.from_dict(config_dict)
