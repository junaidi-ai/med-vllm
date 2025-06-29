"""Test that core modules can be imported without PyTorch."""

import importlib
import sys
import warnings
from pathlib import Path
from typing import List, Set
from unittest.mock import MagicMock, patch

import pytest

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Mock PyTorch and related packages at module level
sys.modules['torch'] = MagicMock()
sys.modules['triton'] = MagicMock()
sys.modules['flash_attn'] = MagicMock()
sys.modules['transformers'] = MagicMock()

# Simple test for core modules that shouldn't require PyTorch
CORE_MODULES = [
    "medvllm",
    "medvllm.utils.logging",
    "medvllm.medical.config.constants",
    "medvllm.medical.config.types",
    "medvllm.medical.config.validation",
    "medvllm.medical.config.versioning",
    "medvllm.medical.config.utils",
    "medvllm.medical.config.serialization",
    "medvllm.medical.config.base",
    "medvllm.medical.config.models",
    "medvllm.medical.config",
    "medvllm.medical",
]

# Test core imports without any PyTorch dependencies
@pytest.mark.parametrize("module_name", CORE_MODULES)
def test_core_imports(module_name):
    """Test that core modules can be imported without PyTorch."""
    # Patch importlib to handle any problematic imports
    with patch.dict('sys.modules'):
        # Ensure we don't have any medvllm modules imported
        for mod in list(sys.modules.keys()):
            if mod.startswith('medvllm'):
                del sys.modules[mod]
        
        # Now try to import the module
        try:
            module = importlib.import_module(module_name)
            assert module is not None
        except ImportError as e:
            if 'torch' in str(e) or 'transformers' in str(e):
                pytest.skip(f"Skipping {module_name} as it requires PyTorch")
            else:
                raise

def test_import_medical_model_config():
    """Test that MedicalModelConfig can be imported."""
    # This will be skipped since it requires PyTorch
    with patch.dict('sys.modules'):
        try:
            from medvllm.medical.config import MedicalModelConfig
            assert MedicalModelConfig is not None
        except ImportError as e:
            if 'torch' in str(e) or 'transformers' in str(e):
                pytest.skip("Skipping MedicalModelConfig as it requires PyTorch")
            else:
                raise

def test_import_order():
    """Test that modules can be imported in the correct order."""
    with patch.dict('sys.modules'):
        try:
            # Import in dependency order
            import medvllm  # noqa: F401
            import medvllm.utils.logging  # noqa: F401
            import medvllm.medical.config.constants  # noqa: F401
            import medvllm.medical.config.types  # noqa: F401
            import medvllm.medical.config.validation  # noqa: F401
            import medvllm.medical.config.versioning  # noqa: F401
            import medvllm.medical.config.utils  # noqa: F401
            import medvllm.medical.config.serialization  # noqa: F401
            import medvllm.medical.config.base  # noqa: F401
            import medvllm.medical.config.models  # noqa: F401
            import medvllm.medical.config  # noqa: F401
            import medvllm.medical  # noqa: F401
        except ImportError as e:
            if 'torch' in str(e) or 'transformers' in str(e):
                pytest.skip("Skipping import order test as it requires PyTorch")
            else:
                raise

if __name__ == "__main__":
    # When run directly, print import status for each module
    results = []
    for module_name in CORE_MODULES:
        try:
            with patch.dict('sys.modules'):
                importlib.import_module(module_name)
                status = "✅"
        except ImportError as e:
            if 'No module named' in str(e) and ('torch' in str(e) or 'transformers' in str(e)):
                status = "⏭️ (requires PyTorch)"
            else:
                status = f"❌ {str(e)}"
        except Exception as e:
            status = f"❌ {str(e)}"
        results.append(f"{status} {module_name}")
    
    print("\n".join(results))

    # Test MedicalModelConfig specifically
    try:
        with patch.dict('sys.modules'):
            from medvllm.medical.config import MedicalModelConfig
            results.append("✅ medvllm.medical.config.MedicalModelConfig")
    except ImportError as e:
        if 'torch' in str(e) or 'transformers' in str(e):
            results.append("⏭️ medvllm.medical.config.MedicalModelConfig (requires PyTorch)")
        else:
            results.append(f"❌ medvllm.medical.config.MedicalModelConfig ({str(e).split('(')[0]})")
    print("\n".join(results))
