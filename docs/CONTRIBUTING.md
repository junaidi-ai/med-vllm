# Contributing to Med vLLM

Thank you for your interest in contributing to Med vLLM! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork the repository and clone your fork:
   ```bash
   git clone https://github.com/your-username/med-vllm.git
   cd med-vllm
   ```

2. Set up a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package in development mode with test dependencies:
   ```bash
   pip install -e ".[test]"
   ```

## Running Tests

Run the test suite with:

```bash
# Run all tests
pytest tests/unit/ -v

# Run tests with coverage report
pytest --cov=medvllm tests/unit/ -v

# Run a specific test file
pytest tests/unit/test_medical_adapters.py -v

# Run a specific test class
pytest tests/unit/test_medical_adapters.py::TestBioBERTAdapter -v

# Run a specific test method
pytest tests/unit/test_medical_adapters.py::TestBioBERTAdapter::test_biomedical_text_processing -v
```

## Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for static type checking

Run the following commands to check your code:

```bash
# Format code with Black
black .

# Sort imports with isort
isort .

# Check for style issues with flake8
flake8 .

# Run static type checking with mypy
mypy .
```

## Pull Requests

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them with a descriptive message:
   ```bash
   git commit -m "Add your feature description"
   ```

3. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Open a pull request against the `main` branch of the main repository.

## Writing Tests

- Write unit tests for all new functionality
- Ensure tests are isolated and don't depend on external services
- Use descriptive test method names
- Follow the existing test structure
- Aim for high test coverage

## Reporting Issues

When reporting issues, please include:

- A clear description of the problem
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, etc.)
- Any relevant error messages or logs

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.
