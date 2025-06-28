"""Simple script to verify MedicalModelConfig functionality."""

import os
import sys
from pathlib import Path

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath('.'))

try:
    from medvllm.medical.config.medical_config import MedicalModelConfig
    from medvllm.medical.config.serialization import ConfigSerializer
    
    # Test creating a config with string medical_specialties and anatomical_regions
    config_dict = {
        "model_type": "bert",
        "model": "./test_model_dir",
        "config_version": "0.1.0",
        "medical_specialties": "cardiology, neurology, radiology",
        "anatomical_regions": "head, neck, chest",
        "max_medical_seq_length": 512,
    }
    
    print("Creating MedicalModelConfig from dict:")
    print("Input:", config_dict)
    
    # Test direct instantiation
    config = MedicalModelConfig(**config_dict)
    print("\nDirect instantiation successful!")
    print(f"Model path: {config.model}")
    print(f"Medical specialties: {config.medical_specialties}")
    print(f"Anatomical regions: {config.anatomical_regions}")
    
    # Test from_dict
    config_from_dict = MedicalModelConfig.from_dict(config_dict)
    print("\nfrom_dict() instantiation successful!")
    
    # Test serialization roundtrip
    config_dict_roundtrip = config_from_dict.to_dict()
    print("\nSerialized to dict:")
    print(config_dict_roundtrip)
    
    # Clean up
    if os.path.exists("./test_model_dir"):
        import shutil
        shutil.rmtree("./test_model_dir")
    
    print("\nAll tests passed!")
    
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
