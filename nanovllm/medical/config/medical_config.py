"""
Medical model configuration.

This module contains the main MedicalModelConfig class that brings together
all the configuration components for medical models.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Type, TypeVar
import os
from pathlib import Path

from nanovllm.config import Config
from .base import BaseMedicalConfig
from .validation import MedicalConfigValidator
from .serialization import ConfigSerializer
from .versioning import ConfigVersioner

T = TypeVar('T', bound='MedicalModelConfig')

@dataclass
class MedicalModelConfig(BaseMedicalConfig):
    """Configuration class for medical model parameters.
    
    This class extends the base configuration with medical-specific parameters
    and validation logic.
    """
    # Model parameters
    model_type: str = "medical"
    pretrained_model_name_or_path: Optional[str] = None
    
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
    num_medical_labels: int = 50
    medical_vocab_file: Optional[str] = None
    
    # NER and entity linking
    medical_entity_types: List[str] = field(
        default_factory=lambda: [
            "DISEASE", "SYMPTOM", "TREATMENT", "MEDICATION", "LAB_TEST", 
            "ANATOMY", "PROCEDURE", "FINDING", "OBSERVATION", "FAMILY_HISTORY"
        ]
    )
    ner_confidence_threshold: float = 0.85
    max_entity_span_length: int = 10
    entity_linking_enabled: bool = False
    entity_linking_knowledge_bases: List[str] = field(
        default_factory=lambda: ["umls", "snomed_ct", "rxnorm", "loinc", "icd10", "hpo"]
    )
    
    # Document processing
    document_types: List[str] = field(
        default_factory=lambda: [
            "clinical_notes", "radiology_reports", "discharge_summaries",
            "progress_notes", "surgical_reports", "pathology_reports"
        ]
    )
    section_headers: List[str] = field(
        default_factory=lambda: [
            "history", "findings", "impression", "assessment", "plan",
            "medications", "allergies", "procedures", "family_history", "social_history"
        ]
    )
    
    # Clinical NLP specific
    enable_uncertainty_estimation: bool = True
    uncertainty_threshold: float = 0.3
    
    # API and request handling
    max_retries: int = 3
    request_timeout: int = 30  # seconds
    
    # Performance and optimization
    batch_size: int = 32
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour in seconds
    max_cache_size: int = 1000  # Max number of items in cache
    
    # Advanced parameters
    use_crf: bool = True
    do_lower_case: bool = True
    preserve_case_for_abbreviations: bool = True
    domain_adaptation: bool = False
    domain_adaptation_lambda: float = 0.1
    domain_specific_vocab: Optional[Dict[str, List[str]]] = None
    
    # Regulatory compliance
    regulatory_compliance: List[str] = field(
        default_factory=lambda: ["hipaa", "gdpr", "hl7", "fda_510k", "ce_mark"]
    )
    
    # NER and entity linking
    medical_entity_types: List[str] = field(
        default_factory=lambda: [
            "DISEASE", "SYMPTOM", "TREATMENT", "MEDICATION",
            "LAB_TEST", "ANATOMY", "PROCEDURE", "FINDING"
        ]
    )
    ner_confidence_threshold: float = 0.85
    max_entity_span_length: int = 10
    entity_linking_enabled: bool = False
    entity_linking_knowledge_bases: List[str] = field(
        default_factory=lambda: ["umls", "snomed_ct", "rxnorm", "loinc"]
    )
    
    # Document processing
    document_types: List[str] = field(
        default_factory=lambda: ["clinical_notes", "radiology_reports", "discharge_summaries"]
    )
    section_headers: List[str] = field(
        default_factory=lambda: [
            "history", "findings", "impression",
            "assessment", "plan", "medications"
        ]
    )
    
    # Performance and optimization
    batch_size: int = 32
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour in seconds
    max_cache_size: int = 1000  # Max number of items in cache
    
    # Advanced parameters
    use_crf: bool = True
    do_lower_case: bool = True
    preserve_case_for_abbreviations: bool = True
    domain_adaptation: bool = False
    domain_adaptation_lambda: float = 0.1
    domain_specific_vocab: Optional[Dict[str, List[str]]] = None
    
    def __post_init__(self):
        """Initialize the configuration with validation."""
        # First ensure compatibility and run base validation
        super().__post_init__()
        
        # Validate medical-specific parameters
        MedicalConfigValidator.validate_medical_parameters(self)
        
        # Set default pretrained paths if needed
        self._set_default_pretrained_paths()
    
    def _set_default_pretrained_paths(self):
        """Set default pretrained model paths if not specified."""
        if not self.pretrained_model_name_or_path and hasattr(self, 'model'):
            self.pretrained_model_name_or_path = self.model
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> 'MedicalModelConfig':
        """Create a config from a pretrained model."""
        config_dict = {"pretrained_model_name_or_path": model_name_or_path}
        config_dict.update(kwargs)
        return cls(**config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MedicalModelConfig':
        """Create a config from a dictionary."""
        return ConfigSerializer.from_dict(cls, config_dict)
    
    @classmethod
    def from_json(
        cls, 
        json_input: Union[str, os.PathLike, Dict]
    ) -> 'MedicalModelConfig':
        """Create a config from a JSON string, file, or dictionary."""
        return ConfigSerializer.from_json(cls, json_input)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary."""
        return ConfigSerializer.to_dict(self)
    
    def to_json(
        self, 
        file_path: Optional[Union[str, os.PathLike]] = None, 
        indent: int = 2
    ) -> Optional[str]:
        """Convert the config to a JSON string or file."""
        return ConfigSerializer.to_json(self, file_path, indent)
