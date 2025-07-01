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

# Default medical specialties
DEFAULT_MEDICAL_SPECIALTIES = [
    "cardiology",
    "dermatology",
    "endocrinology",
    # Changed from general_practice to match MedicalSpecialty enum
    "family_medicine",
    "gastroenterology",
    "hematology",
    "internal_medicine",
    "neurology",
    "oncology",
    "ophthalmology",
    "orthopedics",
    "pediatrics",
    "psychiatry",
    "pulmonology",
    "radiology",
    "urology",
]

# Default anatomical regions
DEFAULT_ANATOMICAL_REGIONS = [
    "head",
    "neck",
    "chest",
    "abdomen",
    "pelvis",
    "upper_limb",
    "lower_limb",
    "back",
    "thorax",
    "spine",
]

# Default imaging modalities
DEFAULT_IMAGING_MODALITIES = [
    "xray",
    "ct",
    "mri",
    "ultrasound",
    "pet",
    "mammography",
    "fluoroscopy",
]

# Default knowledge bases for entity linking
DEFAULT_KNOWLEDGE_BASES = ["umls", "snomed_ct", "loinc", "rxnorm", "icd10"]

# Default document types - must match DocumentType enum values
DEFAULT_DOCUMENT_TYPES = [
    "clinical_notes",
    "radiology_reports",
    "discharge_summaries",
    "progress_notes",
    "surgical_reports",
    "pathology_reports",
    "consult_notes",
    "emergency_notes",
    "admission_notes",
]

# Default section headers in clinical documents
DEFAULT_SECTION_HEADERS = [
    "chief_complaint",
    "history_of_present_illness",
    "past_medical_history",
    "family_history",
    "social_history",
    "review_of_systems",
    "physical_exam",
    "assessment_and_plan",
]

# Default regulatory standards - must match RegulatoryStandard enum values
DEFAULT_REGULATORY_STANDARDS = [
    "hipaa",
    "gdpr",
    "hl7",
    "fda_510k",
    "ce_mark",
    "hitech",
    "hitrust",
    "nist",
    "iso_13485",
    "iso_14971",
]

# Clinical entity types
DEFAULT_ENTITY_TYPES = [
    "DISEASE",
    "SYMPTOM",
    "TREATMENT",
    "TEST",
    "ANATOMY",
    "GENDER",
    "AGE",
    "DATE",
    "DURATION",
    "FREQUENCY",
    "SEVERITY",
    "ROUTE",
    "DOSAGE",
    "FORM",
    "STRENGTH",
    "FREQUENCY",
    "LAB_VALUE",
    "LAB_UNIT",
    "LAB_INTERPRETATION",
    "PROCEDURE",
    "OCCUPATION",
    "LIVING_STATUS",
    "RACE",
    "ETHNICITY",
    "LANGUAGE",
    "RELIGION",
    "ADDRESS",
    "PHONE",
    "EMAIL",
    "URL",
    "ID",
    "SSN",
    "MEDICAL_RECORD_NUMBER",
    "HEALTH_PLAN_BENEFICIARY_NUMBER",
    "ACCOUNT_NUMBER",
    "LICENSE",
    "VEHICLE",
    "DEVICE",
    "BIOID",
    "HOSPITAL",
    "ORGANIZATION",
    "PROFESSION",
    "CITY",
    "STATE",
    "COUNTRY",
    "ZIPCODE",
    "LOCATION_OTHER",
]
