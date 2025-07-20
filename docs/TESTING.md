# Testing Guide for Medical Model Adapters

This document describes the test structure and how to run tests for the medical model adapter system.

## Test Structure

The adapter system has comprehensive test coverage organized into unit tests and integration tests:

### Unit Tests (`tests/unit/`)

#### `test_adapters.py`
- Basic adapter functionality tests
- Tests for `MedicalModelAdapter`, `BioBERTAdapter`, `ClinicalBERTAdapter`
- Factory function testing (`create_medical_adapter`)
- Mock-based tests that don't require actual models

#### `test_adapter_manager.py`
- Tests for the `AdapterManager` class
- Model type detection tests
- Configuration management tests
- Adapter creation workflow tests
- Error handling and fallback mechanisms

#### `models/test_medical_adapters.py`
- Comprehensive adapter tests
- Abstract base class testing
- Device management tests
- KV caching behavior tests
- Forward pass testing

### Integration Tests (`tests/integration/`)

#### `test_adapter_integration.py`
- End-to-end adapter workflow tests
- Config system integration tests
- ModelManager integration tests
- Device management integration
- Error handling in real scenarios

#### `test_biobert_adapter.py`
- BioBERT-specific integration tests
- Tests with actual model loading (when available)
- KV caching with real models
- BioBERT loader integration

## Running Tests

### Run All Adapter Tests
```bash
# Run all adapter-related tests
pytest tests/ -k "adapter" -v

# Run only unit tests
pytest tests/unit/ -k "adapter" -v

# Run only integration tests
pytest tests/integration/ -k "adapter" -v
```

### Run Specific Test Files
```bash
# Unit tests
pytest tests/unit/test_adapters.py -v
pytest tests/unit/test_adapter_manager.py -v
pytest tests/unit/models/test_medical_adapters.py -v

# Integration tests
pytest tests/integration/test_adapter_integration.py -v
pytest tests/integration/test_biobert_adapter.py -v
```

### Run Tests with Coverage
```bash
pytest tests/ -k "adapter" --cov=medvllm.models --cov-report=html
```

## Test Categories

### 1. Unit Tests
- **Purpose**: Test individual components in isolation
- **Characteristics**: 
  - Use mocked dependencies
  - Fast execution
  - No external model downloads required
  - Test specific functionality and edge cases

### 2. Integration Tests
- **Purpose**: Test component interactions and end-to-end workflows
- **Characteristics**:
  - May use real models (when available)
  - Test adapter integration with the engine
  - Verify configuration system integration
  - Test device management and memory handling

## Test Coverage

The test suite covers:

✅ **Core Adapter Functionality**
- Abstract base class behavior
- Concrete adapter implementations
- Factory pattern functionality
- Device management
- KV caching behavior

✅ **Adapter Manager**
- Model type detection
- Configuration management
- Adapter creation workflows
- Error handling and fallbacks

✅ **Integration Points**
- Config system integration
- ModelManager integration
- Engine integration points
- End-to-end workflows

✅ **Error Handling**
- Invalid model types
- Missing dependencies
- Device errors
- Configuration errors

## Mock Strategy

Tests use comprehensive mocking to avoid dependencies:

```python
# Mock transformers to avoid import issues
import sys
import types

mock_transformers = types.ModuleType('transformers')
mock_transformers.AutoConfig = MagicMock()
sys.modules['transformers'] = mock_transformers
```

This allows tests to run without:
- Downloading large models
- GPU requirements
- External dependencies

## Continuous Integration

Tests are designed to run in CI environments:
- No external model downloads required for unit tests
- Mocked dependencies for isolation
- Fast execution times
- Clear error messages

## Adding New Tests

When adding new adapter types or functionality:

1. **Add Unit Tests**: Test the new component in isolation
2. **Add Integration Tests**: Test integration with existing systems
3. **Update Mock Strategy**: Ensure mocks cover new dependencies
4. **Document Test Cases**: Add descriptions of what's being tested

### Example Test Structure
```python
def test_new_adapter_functionality():
    """Test description of what this test verifies."""
    # Arrange
    mock_model = MagicMock()
    adapter = NewAdapter(mock_model, {})
    
    # Act
    result = adapter.some_method()
    
    # Assert
    assert result is not None
    mock_model.some_call.assert_called_once()
```

## Troubleshooting Tests

### Common Issues

1. **Import Errors**: Ensure transformers is properly mocked
2. **CUDA Errors**: Tests should work on CPU-only systems
3. **Model Loading**: Unit tests shouldn't require real models

### Debug Mode
```bash
# Run tests with verbose output and no capture
pytest tests/unit/test_adapters.py -v -s

# Run single test with debugging
pytest tests/unit/test_adapters.py::test_specific_function -v -s --pdb
```

## Test Data

Tests use minimal synthetic data:
- Mock tensors for input/output
- Simple configuration dictionaries
- Predefined model names for detection testing

No large test datasets or models are required for the core test suite.

---

This testing strategy ensures the adapter system is robust, maintainable, and works correctly across different environments while keeping test execution fast and reliable.
