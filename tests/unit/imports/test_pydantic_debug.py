#!/usr/bin/env python3
"""Debug pytest import behavior for pydantic."""

def test_import_chain():
    """Test the import chain for pydantic."""
    import sys
    import importlib
    
    # Print Python version and pytest version
    print("\n=== Environment ===")
    import sys, pytest
    print(f"Python: {sys.version}")
    print(f"Pytest: {pytest.__version__}")
    
    # Print import hooks
    print("\n=== Import hooks ===")
    for i, hook in enumerate(sys.meta_path):
        print(f"{i}: {hook}")
    
    # Check if pydantic is already in sys.modules
    print("\n=== sys.modules entries before import ===")
    pydantic_mods = {k: v for k, v in sys.modules.items() if 'pydantic' in k}
    for k, v in pydantic_mods.items():
        print(f"{k}: {v}")
    
    # Try to import pydantic
    print("\n=== Importing pydantic ===")
    try:
        import pydantic
        print(f"Imported pydantic: {pydantic}")
        print(f"pydantic.__file__: {getattr(pydantic, '__file__', 'NOT FOUND')}")
        print(f"pydantic.__spec__: {getattr(pydantic, '__spec__', 'NOT FOUND')}")
        print(f"pydantic.__loader__: {getattr(pydantic, '__loader__', 'NOT FOUND')}")
        
        # Try to import BaseModel
        try:
            from pydantic import BaseModel
            print("Successfully imported BaseModel")
        except ImportError as e:
            print(f"Failed to import BaseModel: {e}")
            
    except ImportError as e:
        print(f"Failed to import pydantic: {e}")
    
    # Check sys.modules after import
    print("\n=== sys.modules entries after import ===")
    pydantic_mods = {k: v for k, v in sys.modules.items() if 'pydantic' in k}
    for k, v in pydantic_mods.items():
        print(f"{k}: {v}")

def test_isolated_import():
    """Test importing pydantic with a clean environment."""
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
        print("\n=== Fresh import of pydantic ===")
        import pydantic
        print(f"Imported pydantic: {pydantic}")
        print(f"pydantic.__file__: {getattr(pydantic, '__file__', 'NOT FOUND')}")
        print(f"pydantic.__spec__: {getattr(pydantic, '__spec__', 'NOT FOUND')}")
        print(f"pydantic.__loader__: {getattr(pydantic, '__loader__', 'NOT FOUND')}")
        
        # Try to import BaseModel
        try:
            from pydantic import BaseModel
            print("Successfully imported BaseModel")
            
            # Test creating a model
            class TestModel(BaseModel):
                name: str
                value: int
            
            instance = TestModel(name="test", value=42)
            print(f"Created test model: {instance}")
            
        except ImportError as e:
            print(f"Failed to import BaseModel: {e}")
            
    finally:
        # Restore original modules
        for mod in list(sys.modules):
            if 'pydantic' in mod:
                del sys.modules[mod]
        sys.modules.update(saved_modules)

if __name__ == "__main__":
    print("Running tests directly...")
    test_import_chain()
    test_isolated_import()
    print("\n=== Tests complete ===")
