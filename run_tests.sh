#!/bin/bash

# Exit on error
set -e

echo "=== Setting up test environment ==="
python -m pip install -r requirements-test.txt

echo -e "\n=== Running unit tests ==="
python -m pytest tests/ -v --cov=medvllm --cov-report=term-missing

echo -e "\n=== Running accuracy validation ==="
python tests/validate_accuracy.py

echo -e "\n=== Running benchmarks ==="
if [ "$1" == "--benchmark" ]; then
    python benchmarks/run_benchmarks.py
else
    echo "Skipping benchmarks (use --benchmark to include)"
fi

echo -e "\n=== All tests completed successfully! ==="
