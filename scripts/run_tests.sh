#!/bin/bash

# Set up environment
echo "Setting up test environment..."
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Create results directory
mkdir -p results/accuracy

# Install test requirements if not already installed
pip install -q pytest pytest-cov

# Run tests
echo "Running tests..."
python -m pytest tests/medical/ -v

# Check if we have any results
if [ -n "$(ls -A results/accuracy/ 2>/dev/null)" ]; then
    echo -e "\nTest results:"
    ls -l results/accuracy/
else
    echo -e "\nNo test results were generated. Check for errors above."
fi
