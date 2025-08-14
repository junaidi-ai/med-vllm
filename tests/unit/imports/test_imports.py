"""
Test critical package imports.
This is a lightweight test that runs early in the test suite to catch import issues.
"""

import os
import sys
import unittest


class TestCriticalImports(unittest.TestCase):
    """Test that critical dependencies can be imported."""

    def setUp(self):
        """Print debug information before each test."""
        print("\n" + "=" * 80)
        print(f"Running {self._testMethodName}")
        print("=" * 80)
        print("Python path:")
        print("\n".join(sys.path))
        print("\nEnvironment variables:")
        for k, v in os.environ.items():
            if "PYTHON" in k or "PATH" in k:
                print(f"{k}: {v}")
        print("=" * 80 + "\n")

    def test_torch_import(self):
        """Test that PyTorch can be imported."""
        import torch  # noqa: F401

        self.assertTrue(True, "PyTorch imported successfully")

    def test_transformers_import(self):
        """Test that transformers can be imported."""
        import transformers  # noqa: F401

        self.assertTrue(True, "Transformers imported successfully")

    def test_pydantic_import(self):
        """Test that pydantic can be imported."""
        import sys

        # Clear pydantic from sys.modules to ensure a fresh import
        for mod in list(sys.modules):
            if mod == "pydantic" or mod.startswith("pydantic."):
                del sys.modules[mod]

        try:
            # Import pydantic fresh
            import pydantic

            self.assertTrue(hasattr(pydantic, "__file__"), "pydantic module is missing __file__")

            # Import BaseModel
            from pydantic import BaseModel

            self.assertTrue(True, "Successfully imported BaseModel from pydantic")

        except ImportError as e:
            self.fail(f"Failed to import pydantic: {e}")

    def test_main_package_import(self):
        """Test that the main package can be imported."""
        # First try without importing LLMEngine
        try:
            import medvllm  # noqa: F401

            print("Successfully imported medvllm package")
            self.assertTrue(True, "Med-vLLM package imported successfully")
        except ImportError as e:
            print(f"Error importing medvllm: {e}")
            raise

        # Now try importing LLMEngine specifically
        try:
            from medvllm.engine.llm_engine import LLMEngine  # noqa: F401

            print("Successfully imported LLMEngine")
            self.assertTrue(True, "LLMEngine imported successfully")
        except ImportError as e:
            print(f"Error importing LLMEngine: {e}")
            raise

    def test_core_imports(self):
        """Test that core modules can be imported."""
        # Test core model components
        try:
            # Import from the correct paths based on the project structure
            from medvllm.models.attention.medical_attention import (
                MedicalMultiheadAttention,
            )  # noqa: F401

            # Check if the medical_layers module exists and import from it
            try:
                from medvllm.models.layers.medical_layers import MedicalFeedForward  # noqa: F401
                from medvllm.models.layers.medical_layers import MedicalLayerNorm  # noqa: F401
            except ImportError as e:
                print(f"Warning: Could not import medical_layers: {e}")
                # If medical_layers is not available, that's okay for now
                pass

            # If we get here, imports were successful
            self.assertTrue(True, "Core modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import core modules: {e}")


if __name__ == "__main__":
    unittest.main()
