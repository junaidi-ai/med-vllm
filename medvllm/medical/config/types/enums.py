"""
Enum types for medical model configuration.
"""

from enum import Enum

class MedicalSpecialty(str, Enum):
    """Enumeration of medical specialties."""
    FAMILY_MEDICINE = "family_medicine"
    INTERNAL_MEDICINE = "internal_medicine"
    PEDIATRICS = "pediatrics"
    CARDIOLOGY = "cardiology"
    DERMATOLOGY = "dermatology"
    ENDOCRINOLOGY = "endocrinology"
    GASTROENTEROLOGY = "gastroenterology"
    HEMATOLOGY = "hematology"
    NEUROLOGY = "neurology"
    ONCOLOGY = "oncology"
    OPHTHALMOLOGY = "ophthalmology"
    ORTHOPEDICS = "orthopedics"
    PSYCHIATRY = "psychiatry"
    PULMONOLOGY = "pulmonology"
    RADIOLOGY = "radiology"
    UROLOGY = "urology"


class AnatomicalRegion(str, Enum):
    """Enumeration of anatomical regions."""
    HEAD = "head"
    THORAX = "thorax"
    ABDOMEN = "abdomen"
    PELVIS = "pelvis"
    UPPER_LIMB = "upper_limb"
    LOWER_LIMB = "lower_limb"
    NECK = "neck"
    BACK = "back"
    SPINE = "spine"
    CHEST = "chest"


class ImagingModality(str, Enum):
    """Enumeration of medical imaging modalities."""
    XRAY = "xray"
    CT = "ct"
    MRI = "mri"
    ULTRASOUND = "ultrasound"
    PET = "pet"
    MAMMOGRAPHY = "mammography"
    FLUOROSCOPY = "fluoroscopy"
    ANGIOGRAPHY = "angiography"


class EntityType(str, Enum):
    """Types of medical entities."""
    DISEASE = "DISEASE"
    SYMPTOM = "SYMPTOM"
    TREATMENT = "TREATMENT"
    MEDICATION = "MEDICATION"
    LAB_TEST = "LAB_TEST"
    ANATOMY = "ANATOMY"
    PROCEDURE = "PROCEDURE"
    FINDING = "FINDING"
    OBSERVATION = "OBSERVATION"
    FAMILY_HISTORY = "FAMILY_HISTORY"
    ALLERGY = "ALLERGY"
    VITAL_SIGN = "VITAL_SIGN"
    DEVICE = "DEVICE"
    BIOPSY = "BIOPSY"
    IMAGING_FINDING = "IMAGING_FINDING"
    PATHOLOGY_FINDING = "PATHOLOGY_FINDING"


class DocumentType(str, Enum):
    """Types of clinical documents."""
    CLINICAL_NOTES = "clinical_notes"
    RADIOLOGY_REPORTS = "radiology_reports"
    DISCHARGE_SUMMARIES = "discharge_summaries"
    PROGRESS_NOTES = "progress_notes"
    SURGICAL_REPORTS = "surgical_reports"
    PATHOLOGY_REPORTS = "pathology_reports"
    CONSULT_NOTES = "consult_notes"
    EMERGENCY_NOTES = "emergency_notes"
    ADMISSION_NOTES = "admission_notes"


class RegulatoryStandard(str, Enum):
    """Regulatory compliance standards."""
    HIPAA = "hipaa"
    GDPR = "gdpr"
    HL7 = "hl7"
    FDA_510K = "fda_510k"
    CE_MARK = "ce_mark"
    HITECH = "hitech"
    HITRUST = "hitrust"
    NIST = "nist"
    ISO_13485 = "iso_13485"
    ISO_14971 = "iso_14971"
