#!/usr/bin/env python3
"""Minimal test case for pydantic import issue."""

def test_pydantic_import():
    """Test that pydantic can be imported and has __file__ attribute."""
    import pydantic
    
    # Check basic attributes
    assert hasattr(pydantic, '__file__'), "pydantic module is missing __file__"
    assert pydantic.__file__.endswith('__init__.py'), "pydantic.__file__ doesn't point to __init__.py"
    
    # Try to import BaseModel
    from pydantic import BaseModel
    assert BaseModel is not None, "Failed to import BaseModel from pydantic"
    
    # Create a simple model
    class TestModel(BaseModel):
        name: str
        value: int
    
    # Test the model
    instance = TestModel(name="test", value=42)
    assert instance.name == "test"
    assert instance.value == 42

def test_pydantic_import_isolated():
    """Test pydantic import in an isolated environment."""
    import sys
    import importlib
    
    # Save current modules
    saved_modules = {k: v for k, v in sys.modules.items() if 'pydantic' in k}
    
    # Clear pydantic from sys.modules
    for mod in list(sys.modules):
        if 'pydantic' in mod:
            del sys.modules[mod]
    
    try:
        # Import pydantic fresh
        import pydantic
        assert hasattr(pydantic, '__file__'), "Fresh import: pydantic missing __file__"
        print(f"pydantic.__file__: {pydantic.__file__}")
        
        # Import BaseModel
        from pydantic import BaseModel
        assert BaseModel is not None, "Failed to import BaseModel"
        
    finally:
        # Restore original modules
        for mod in list(sys.modules):
            if 'pydantic' in mod:
                del sys.modules[mod]
        sys.modules.update(saved_modules)

if __name__ == "__main__":
    print("Running tests directly...")
    test_pydantic_import()
    test_pydantic_import_isolated()
    print("All tests passed!")
