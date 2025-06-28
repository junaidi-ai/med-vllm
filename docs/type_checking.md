# Type Checking Standards and Plan

## Overview
This document outlines our approach to type checking in the codebase using mypy. Our goal is to gradually improve type safety while maintaining development velocity.

## Type Checking Rules

### Required for All New Code
1. All function definitions must have complete type annotations
2. Use `Optional[Type]` instead of implicit `Type = None`
3. Avoid `Any` unless absolutely necessary
4. Use type aliases for complex types
5. Document non-obvious type decisions with comments

### Type Checking Configuration
- Python 3.12+ type system features are enabled
- Strict mode is enabled by default
- Per-module configurations can relax rules where necessary

## Incremental Type Checking Plan

### Phase 1: Core Configuration (Current)
- [x] Set up minimal mypy configuration
- [ ] Fix type issues in critical path modules
- [ ] Add type stubs for external dependencies

### Phase 2: Core Modules
- [ ] `medvllm/medical/config/`
  - [ ] `medical_config.py`
  - [ ] `serialization.py`
  - [ ] `schema.py`

### Phase 3: Engine Components
- [ ] `medvllm/engine/`
  - [ ] `model_runner.py`
  - [ ] `scheduler.py`
  - [ ] `sequence.py`

### Phase 4: Remaining Modules
- [ ] `medvllm/layers/`
- [ ] `medvllm/utils/`
- [ ] Tests

## How to Add Type Annotations

### Function Signatures
```python
def process_data(data: list[dict[str, Any]], timeout: int = 30) -> ProcessResult:
    """Process input data with a timeout.
    
    Args:
        data: List of data dictionaries to process
        timeout: Maximum time to wait in seconds
        
    Returns:
        ProcessResult containing the processed data
    """
    ...
```

### Type Aliases
```python
from typing import TypedDict, Sequence

class ModelConfig(TypedDict):
    model_type: str
    hidden_size: int
    num_layers: int

ConfigSequence = Sequence[ModelConfig]
```

### Handling Optional Values
```python
def get_user(id: int) -> Optional[User]:
    """Return user if found, None otherwise."""
    ...

# Usage
user = get_user(123)
if user is not None:
    print(user.name)
```

## Common Patterns

### Type Ignore
Only use `# type: ignore` when:
1. There's a false positive that can't be fixed otherwise
2. The issue is in a third-party library
3. You've added a comment explaining why it's needed

### Type Casting
```python
from typing import cast

value = some_untyped_function()
# When you're certain about the type
result = cast(ExpectedType, value)
```

## Running Type Checks

```bash
# Check everything
mypy .

# Check specific module
mypy medvllm/medical/config/

# Run with more verbose output
mypy --pretty .
```

## IDE Integration
- VS Code: Use Pylance with "python.analysis.typeCheckingMode": "basic"
- PyCharm: Built-in type checking works well with mypy
- Vim/Neovim: Use ALE or coc.nvim with coc-tsserver

## Review Process
1. All PRs must pass type checking for modified files
2. New files must be fully typed
3. When modifying existing files, improve type annotations if possible
