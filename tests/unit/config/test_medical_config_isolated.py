"""Isolated tests for MedicalModelConfig with minimal dependencies."""
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# Mock the imports that would normally be provided by the application
class MockBaseConfig:
    pass

# Create a mock for the module
sys.modules['medvllm.medical.config.base'] = MagicMock()
sys.modules['medvllm.medical.config.base'].BaseMedicalConfig = MockBaseConfig

# Now import the class we want to test
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from medvllm.medical.config.medical_config import MedicalModelConfig

class TestMedicalModelConfigIsolated:
    """Test MedicalModelConfig with isolated dependencies."""
    
    def test_medical_model_config_creation(self):
        """Test basic MedicalModelConfig creation."""
        config = MedicalModelConfig(
            model="test-model",
            medical_specialties=["cardiology"],
            anatomical_regions=["head"]
        )
        
        assert config.model == "test-model"
        assert config.medical_specialties == ["cardiology"]
        assert config.anatomical_regions == ["head"]

    def test_medical_specialties_validation(self):
        """Test validation of medical_specialties field."""
        # Test with string
        config = MedicalModelConfig(medical_specialties="cardiology, neurology")
        assert config.medical_specialties == ["cardiology", "neurology"]
        
        # Test with list
        config = MedicalModelConfig(medical_specialties=["cardiology", "neurology"])
        assert config.medical_specialties == ["cardiology", "neurology"]
        
        # Test with None
        config = MedicalModelConfig(medical_specialties=None)
        assert config.medical_specialties == []

    def test_anatomical_regions_validation(self):
        """Test validation of anatomical_regions field."""
        # Test with string
        config = MedicalModelConfig(anatomical_regions="head, chest")
        assert config.anatomical_regions == ["head", "chest"]
        
        # Test with list
        config = MedicalModelConfig(anatomical_regions=["head", "chest"])
        assert config.anatomical_regions == ["head", "chest"]
        
        # Test with None
        config = MedicalModelConfig(anatomical_regions=None)
        assert config.anatomical_regions == []

    @patch('os.makedirs')
    def test_model_dir_creation(self, mock_makedirs):
        """Test that model directory is created if it doesn't exist."""
        test_path = "/tmp/test_model_dir"
        config = MedicalModelConfig(model=test_path)
        mock_makedirs.assert_called_once_with(test_path, exist_ok=True)

if __name__ == "__main__":
    import sys
    pytest.main(sys.argv)
