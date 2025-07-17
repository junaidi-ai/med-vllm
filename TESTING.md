# Testing Guide for Med vLLM

This document provides an overview of the testing strategy and instructions for running tests in the Med vLLM project.

## Test Structure

### Unit Tests
- Location: `tests/unit/`
- Purpose: Test individual components in isolation using mocks
- Dependencies: Minimal, only test dependencies (pytest, pytest-mock)
- Run with: `pytest tests/unit/`

### Integration Tests
- Location: `tests/integration/`
- Purpose: Test component interactions with real dependencies
- Dependencies: May require external services or data
- Marked with: `@pytest.mark.integration`
- Run with: `pytest -m integration tests/integration/`

### Performance Tests
- Location: `tests/performance/`
- Purpose: Measure and monitor performance characteristics
- Run with: `pytest tests/performance/`

## Running Tests

### Basic Test Commands

```bash
# Run all tests
pytest

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest -m integration tests/integration/

# Run tests with coverage report
pytest --cov=medvllm tests/

# Run a specific test file
pytest tests/unit/engine/test_model_registry.py -v
```

### Test Markers

- `@pytest.mark.unit`: Basic unit tests (default)
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.performance`: Performance tests
- `@pytest.mark.slow`: Long-running tests (excluded by default)

## Writing Tests

### Unit Test Guidelines
- Use mocks to isolate components
- Keep tests small and focused
- Follow the Arrange-Act-Assert pattern
- Name tests descriptively (test_<method>_<scenario>_<expected>)

### Fixtures
Common test fixtures are defined in:
- `tests/conftest.py`: Base fixtures
- `tests/conftest_override.py`: Test overrides and mocks

## CI/CD Integration

Tests are automatically run on:
- Push to main branch
- Pull requests
- Scheduled runs

## Troubleshooting

### Common Issues

1. **Tests are being skipped**
   - Check if you need to set up environment variables
   - Verify test markers are correct

2. **Dependency issues**
   - Make sure all test dependencies are installed
   - Run `pip install -e .[test]`

3. **Test failures**
   - Run with `-v` for more verbose output
   - Use `--pdb` to drop into debugger on failure

## Adding New Tests

1. Place tests in the appropriate directory
2. Use descriptive names
3. Add appropriate markers
4. Include docstrings explaining the test case
5. Update this document if adding new test categories or patterns
