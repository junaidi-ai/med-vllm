"""Pytest configuration and fixtures."""

import importlib
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Import mock field function to handle description parameter
from tests.mock_field import field  # noqa: F401

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Mock flash_attn and other CUDA-related modules
class MockFlashAttn:
    @staticmethod
    def flash_attn_varlen_func(*args, **kwargs):
        return MagicMock()

    @staticmethod
    def flash_attn_with_kvcache(*args, **kwargs):
        return MagicMock()


# Mock torch and CUDA-related modules
class MockTorch:
    class cuda:
        @staticmethod
        def is_available():
            return False

        class Stream:
            pass

    class multiprocessing:
        def __init__(self):
            pass

        def get_context(self, *args, **kwargs):
            return self

        def set_sharing_strategy(self, *args, **kwargs):
            pass

    class Tensor:
        pass

    class nn:
        class Module:
            pass

    @staticmethod
    def manual_seed(*args, **kwargs):
        pass

    @property
    def float16(self):
        return "float16"


# Apply mocks
sys.modules["flash_attn"] = MockFlashAttn()
torch_mock = MockTorch()
sys.modules["torch"] = torch_mock
sys.modules["torch.cuda"] = torch_mock.cuda
sys.modules["torch.multiprocessing"] = torch_mock.multiprocessing

# Skip tests that require CUDA if not available
HAS_CUDA = False


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "cuda: mark test as requiring CUDA")
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (slow, requires external services)",
    )


# Skip tests that require CUDA if not available
def pytest_runtest_setup(item):
    """Skip tests that require CUDA if not available."""
    if not HAS_CUDA and any(item.iter_markers(name="cuda")):
        pytest.skip("Test requires CUDA")


# Add a command line option to run CUDA tests
def pytest_addoption(parser):
    """Add command line options for pytest."""
    parser.addoption(
        "--run-cuda",
        action="store_true",
        default=False,
        help="run tests that require CUDA",
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow tests",
    )
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests",
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options."""
    skip_cuda = not config.getoption("--run-cuda")
    skip_slow = not config.getoption("--run-slow")
    skip_integration = not config.getoption("--run-integration")

    for item in items:
        if skip_cuda and "cuda" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="need --run-cuda option to run"))
        if skip_slow and "slow" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="need --run-slow option to run"))
        if skip_integration and "integration" in item.keywords:
            item.add_marker(
                pytest.mark.skip(reason="need --run-integration option to run")
            )
