"""Task-specific implementations for medical NLP.

This module keeps imports light to avoid pulling heavy dependencies (e.g.,
transformers-backed tasks) during test collection or when only NER is used.
"""

from .ner_processor import NERProcessor, NERResult

__all__ = ["NERProcessor", "NERResult"]

# Optional exports: only if dependencies are available
try:  # clinical notes generation (may require transformers)
    from .clinical_notes import ClinicalNotesGenerator  # type: ignore

    __all__.append("ClinicalNotesGenerator")
except Exception:  # pragma: no cover - optional dependency not present
    ClinicalNotesGenerator = None  # type: ignore

try:  # medical NER task wrapper (may require model deps)
    from .medical_ner import MedicalNER  # type: ignore

    __all__.append("MedicalNER")
except Exception:  # pragma: no cover
    MedicalNER = None  # type: ignore

try:  # medical QA task (may require transformers)
    from .medical_qa import MedicalQA  # type: ignore

    __all__.append("MedicalQA")
except Exception:  # pragma: no cover
    MedicalQA = None  # type: ignore
