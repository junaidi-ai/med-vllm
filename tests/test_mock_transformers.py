"""Test file that works with the mocked transformers package."""

def test_mock_transformers():
    """Test that the mock transformers package is working as expected."""
    import sys
    
    # Check if we're using the mock
    if 'transformers' in sys.modules and hasattr(sys.modules['transformers'], 'MockTransformers'):
        print("Using mock transformers package")
        from transformers import MockTransformers
        
        # Test that we can create a mock model
        mock_model = MockTransformers.MockPreTrainedModel()
        assert mock_model is not None
        
        # Test that we can create a mock tokenizer
        mock_tokenizer = MockTransformers.MockTokenizer()
        assert mock_tokenizer is not None
        assert hasattr(mock_tokenizer, 'pad_token_id')
        assert hasattr(mock_tokenizer, 'eos_token_id')
        
        # Test that we can create a config
        config = MockTransformers.Qwen3Config()
        assert config is not None
        assert hasattr(config, 'vocab_size')
        
        print("All mock transformers tests passed!")
        assert True
    else:
        print("Using real transformers package")
        # If we're not using mocks, skip this test
        import pytest
        pytest.skip("Not using mock transformers, skipping mock tests")

def test_import_medvllm():
    """Test that we can import medvllm with the mock transformers."""
    import sys
    
    # Check if we're using the mock
    if 'transformers' in sys.modules and hasattr(sys.modules['transformers'], 'MockTransformers'):
        print("Testing medvllm import with mock transformers")
        
        # Import medvllm - this should work with the mocks
        try:
            import medvllm
            print("Successfully imported medvllm with mock transformers")
            
            # If we can import LLM, test it
            if hasattr(medvllm, 'LLM'):
                print("LLM class is available")
                llm = medvllm.LLM()
                assert llm is not None
                
            assert True
            
        except ImportError as e:
            print(f"Failed to import medvllm: {e}")
            assert False, f"Failed to import medvllm: {e}"
    else:
        print("Using real transformers, skipping mock import test")
        import pytest
        pytest.skip("Not using mock transformers, skipping mock import test")
