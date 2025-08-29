"""Configuration and fixtures for unit tests."""

from unittest.mock import patch

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
    """No-op fixture retained for backward compatibility.

    Historically this fixture injected MagicMock versions of heavy deps like
    torch/transformers into sys.modules. That leaked MagicMocks across the
    test session and broke tests that assert tensor/device semantics or rely
    on real module metadata. We now do nothing here and rely on the project
    level mocks in `tests/conftest.py`.
    """
    yield
