"""Configuration and fixtures for unit tests."""

from unittest.mock import MagicMock, patch

import pytest

# Common fixtures for unit tests


@pytest.fixture
def mock_medical_config():
    """Return a mock MedicalModelConfig with default values."""
    return {
        "model": "test-model",
        "medical_specialties": ["cardiology"],
        "anatomical_regions": ["head"],
        "max_seq_len": 1024,
        "dtype": "float16",
    }


@pytest.fixture
def patch_imports():
    """Patch common imports to avoid loading heavy dependencies."""
    with patch.dict(
        "sys.modules",
        {
            "torch": MagicMock(),
            "torch.cuda": MagicMock(is_available=lambda: False),
            "flash_attn": MagicMock(),
            "transformers": MagicMock(),
        },
    ):
        yield
