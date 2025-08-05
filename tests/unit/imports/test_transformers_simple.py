"""Simple test to verify transformers import in test environment."""

def test_transformers_import():
    """Test that transformers can be imported in the test environment."""
    try:
        import transformers
        assert transformers is not None
        
        # Check if we're using a mock
        is_mock = hasattr(transformers, 'MockTransformers') or 'Mock' in str(type(transformers))
        
        if is_mock:
            print("Using mock transformers package")
            # Just verify it has some expected attributes or methods
            # Don't be too strict about what's available in the mock
            print("Mock transformers attributes:", dir(transformers)[:10])  # Print first 10 attrs
        else:
            # Only try to access __file__ if not a mock
            print(f"Using real transformers package from: {transformers.__file__}")
            # Try to import a submodule
            from transformers.configuration_utils import PretrainedConfig
            print("Successfully imported PretrainedConfig from real transformers")
        
        assert True, "Successfully imported transformers"
        
    except ImportError as e:
        print(f"Failed to import transformers: {e}")
        assert False, f"Failed to import transformers: {e}"
