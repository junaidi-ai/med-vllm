"""
Basic test suite for core functionality.
This file consolidates minimal tests for the development environment and core dependencies.
"""

import sys
import unittest

import torch
import torch.nn as nn


class TestEnvironment(unittest.TestCase):
    """Test the basic development environment."""

    def test_python_version(self):
        """Verify Python version is 3.8+."""
        self.assertGreaterEqual(sys.version_info, (3, 8))

    def test_imports(self):
        """Verify core dependencies can be imported."""
        import numpy as np  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401


class TestPyTorch(unittest.TestCase):
    """Test PyTorch core functionality."""

    def test_tensor_creation(self):
        """Test tensor creation and properties."""
        # Test CPU tensor
        cpu_tensor = torch.ones(3, 4)
        self.assertEqual(cpu_tensor.device.type, "cpu")
        self.assertEqual(cpu_tensor.dtype, torch.float32)
        self.assertEqual(cpu_tensor.shape, (3, 4))

        # Test CUDA tensor if available
        if torch.cuda.is_available():
            cuda_tensor = torch.ones(3, 4, device="cuda")
            self.assertEqual(cuda_tensor.device.type, "cuda")
            self.assertEqual(cuda_tensor.dtype, torch.float32)
            self.assertEqual(cuda_tensor.shape, (3, 4))

    def test_nn_module(self):
        """Test basic neural network functionality."""
        # Test embedding layer
        embedding = nn.Embedding(10, 5)
        input_ids = torch.tensor([[1, 2, 3]])
        output = embedding(input_ids)
        self.assertEqual(output.shape, (1, 3, 5))

        # Test linear layer
        linear = nn.Linear(5, 2)
        output = linear(output)
        self.assertEqual(output.shape, (1, 3, 2))


class TestPackageImports(unittest.TestCase):
    """Test that all package modules can be imported."""

    def test_import_package_modules(self):
        """Test that all modules in the package can be imported."""
        import importlib
        import pkgutil

        import medvllm

        package = medvllm
        prefix = f"{package.__name__}."

        for _, name, _ in pkgutil.walk_packages(package.__path__, prefix):
            with self.subTest(module=name):
                try:
                    importlib.import_module(name)
                except ImportError as e:
                    self.fail(f"Failed to import {name}: {e}")


if __name__ == "__main__":
    unittest.main()
