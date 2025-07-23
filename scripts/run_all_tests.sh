#!/bin/bash

# Run all tests and benchmarks
set -e

# Create timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="test_results_${TIMESTAMP}"
BENCHMARK_DIR="${OUTPUT_DIR}/benchmarks"

# Create output directories
mkdir -p "${BENCHMARK_DIR}"

echo "Running tests and benchmarks..."
echo "Output will be saved to: ${OUTPUT_DIR}"

# Run unit tests
echo -e "\n=== Running unit tests ==="
python -m pytest tests/unit/ -v --junitxml="${OUTPUT_DIR}/unit_tests.xml"

# Run medical model tests
echo -e "\n=== Running medical model tests ==="
python -m pytest tests/medical/test_medical_models.py -v --junitxml="${OUTPUT_DIR}/medical_tests.xml"

# Run accuracy comparison
echo -e "\n=== Running accuracy comparison ==="
python -m pytest tests/medical/test_accuracy.py -v --junitxml="${OUTPUT_DIR}/accuracy_tests.xml"

# Run benchmarks
echo -e "\n=== Running benchmarks ==="
python benchmarks/benchmark_medical.py --output-dir "${BENCHMARK_DIR}"

# Generate report
echo -e "\n=== Generating report ==="
python benchmarks/generate_report.py --results-dir "${BENCHMARK_DIR}" --output "${OUTPUT_DIR}/benchmark_report.md"

echo -e "\n=== All tests completed ==="
echo "Results saved to: ${OUTPUT_DIR}"
# Make the script executable
