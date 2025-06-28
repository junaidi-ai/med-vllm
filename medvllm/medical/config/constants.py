"""
Default values and constants for medical model configurations.
"""

# Model configuration defaults
DEFAULT_MODEL_TYPE = "medical_bert"
DEFAULT_MAX_SEQ_LENGTH = 512
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_LABELS = 50
DEFAULT_NER_THRESHOLD = 0.85
DEFAULT_MAX_ENTITY_SPAN = 10
DEFAULT_UNCERTAINTY_THRESHOLD = 0.3
DEFAULT_MAX_RETRIES = 3
DEFAULT_REQUEST_TIMEOUT = 30
DEFAULT_DOMAIN_ADAPTATION_LAMBDA = 0.1
CONFIG_VERSION = "1.0.0"

# Supported model types
SUPPORTED_MODEL_TYPES = {
    "bert",
    "roberta",
    "gpt2",
    "t5",
    "medical_bert",
    "biobert",
    "clinical_bert",
    "pubmed_bert",
    "bluebert",
}

# Medical specialties
DEFAULT_MEDICAL_SPECIALTIES = [
    # Primary Care
    "family_medicine",
    "internal_medicine",
    "pediatrics",
    "geriatrics",
    "obstetrics_gynecology",
    "preventive_medicine",
    # Medical Specialties
    "allergy_immunology",
    "anesthesiology",
    "cardiology",
    "dermatology",
    "endocrinology",
    "gastroenterology",
    "hematology",
    "infectious_disease",
    "nephrology",
    "pulmonology",
    "rheumatology",
    # Surgical Specialties
    "general_surgery",
    "cardiac_surgery",
    "neurosurgery",
    "orthopedic_surgery",
    "plastic_surgery",
    "thoracic_surgery",
    "transplant_surgery",
    "urology",
    "vascular_surgery",
    # Diagnostic Specialties
    "pathology",
    "clinical_pathology",
    "anatomic_pathology",
    "radiology",
    "diagnostic_radiology",
    "interventional_radiology",
    "nuclear_medicine",
    # Hospital-based Specialties
    "emergency_medicine",
    "critical_care_medicine",
    "hospital_medicine",
    "palliative_care",
    "pain_medicine",
    "sleep_medicine",
    # Mental Health
    "psychiatry",
    "child_psychiatry",
    "addiction_psychiatry",
    "forensic_psychiatry",
    "geriatric_psychiatry",
    # Other Specialties
    "dermatology",
    "neurology",
    "neurosurgery",
    "ophthalmology",
    "otolaryngology",
    "physical_medicine_rehab",
    "radiation_oncology",
    "reproductive_endocrinology",
    "sports_medicine",
    "wound_care",
]

# Anatomical regions
DEFAULT_ANATOMICAL_REGIONS = [
    # Head and Neck
    "head",
    "skull",
    "face",
    "forehead",
    "temple",
    "scalp",
    "eye",
    "eyebrow",
    "eyelid",
    "conjunctiva",
    "cornea",
    "retina",
    "ear",
    "auricle",
    "external_auditory_canal",
    "tympanic_membrane",
    "nose",
    "nasal_cavity",
    "paranasal_sinuses",
    "mouth",
    "oral_cavity",
    "lips",
    "tongue",
    "palate",
    "pharynx",
    "neck",
    "larynx",
    "thyroid",
    "parathyroid",
    "trachea",
    # Thorax
    "thorax",
    "chest",
    "thoracic_wall",
    "ribs",
    "sternum",
    "pleura",
    "pleural_cavity",
    "mediastinum",
    "lungs",
    "bronchi",
    "bronchioles",
    "alveoli",
    "heart",
    "pericardium",
    "myocardium",
    "endocardium",
    "esophagus",
    "thymus",
    # Abdomen and Pelvis
    "abdomen",
    "abdominal_wall",
    "peritoneum",
    "peritoneal_cavity",
    "stomach",
    "small_intestine",
    "duodenum",
    "jejunum",
    "ileum",
    "large_intestine",
    "cecum",
    "appendix",
    "colon",
    "rectum",
    "anus",
    "liver",
    "gallbladder",
    "biliary_tract",
    "pancreas",
    "spleen",
    "kidneys",
    "ureters",
    "pelvis",
    "pelvic_cavity",
    "urinary_bladder",
    "urethra",
    "prostate",
    "seminal_vesicles",
    "testes",
    "epididymis",
    "ovaries",
    "fallopian_tubes",
    "uterus",
    "cervix",
    "vagina",
    "vulva",
    # Back and Spine
    "back",
    "spine",
    "vertebral_column",
    "cervical_spine",
    "thoracic_spine",
    "lumbar_spine",
    "sacrum",
    "coccyx",
    "intervertebral_discs",
    "spinal_cord",
    "meninges",
    "cauda_equina",
    # Upper Limbs
    "upper_limb",
    "shoulder",
    "axilla",
    "arm",
    "upper_arm",
    "humerus",
    "elbow",
    "forearm",
    "radius",
    "ulna",
    "wrist",
    "hand",
    "carpal_bones",
    "metacarpals",
    "phalanges",
    "fingers",
    "thumb",
    # Lower Limbs
    "lower_limb",
    "hip",
    "thigh",
    "femur",
    "knee",
    "patella",
    "leg",
    "tibia",
    "fibula",
    "ankle",
    "foot",
    "tarsal_bones",
    "calcaneus",
    "talus",
    "metatarsals",
    "toes",
    # Other
    "skin",
    "subcutaneous_tissue",
    "fascia",
    "muscles",
    "tendons",
    "ligaments",
    "joints",
    "bursae",
    "nerves",
    "blood_vessels",
    "lymph_nodes",
    "lymphatic_vessels",
    # General/Whole Body
    "whole_body",
    "bilateral",
    "unilateral",
    "proximal",
    "distal",
    "anterior",
    "posterior",
    "medial",
    "lateral",
    "superior",
    "inferior",
]

# Imaging modalities
DEFAULT_IMAGING_MODALITIES = [
    "xray",
    "ct",
    "mri",
    "ultrasound",
    "pet",
    "mammography",
    "fluoroscopy",
    "angiography",
]

# Clinical entity types
DEFAULT_ENTITY_TYPES = [
    "DISEASE",
    "SYMPTOM",
    "TREATMENT",
    "MEDICATION",
    "LAB_TEST",
    "ANATOMY",
    "PROCEDURE",
    "FINDING",
    "OBSERVATION",
    "FAMILY_HISTORY",
]

# Document types
DEFAULT_DOCUMENT_TYPES = [
    "clinical_notes",
    "radiology_reports",
    "discharge_summaries",
    "progress_notes",
    "surgical_reports",
    "pathology_reports",
]

# Section headers
DEFAULT_SECTION_HEADERS = [
    "history",
    "findings",
    "impression",
    "assessment",
    "plan",
    "medications",
    "allergies",
    "procedures",
    "family_history",
    "social_history",
]

# Knowledge bases for entity linking
DEFAULT_KNOWLEDGE_BASES = ["umls", "snomed_ct", "rxnorm", "loinc", "icd10", "hpo"]

# Regulatory compliance standards
DEFAULT_REGULATORY_STANDARDS = ["hipaa", "gdpr", "hl7", "fda_510k", "ce_mark"]
