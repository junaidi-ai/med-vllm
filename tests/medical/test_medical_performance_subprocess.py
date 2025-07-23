import os
import subprocess
import sys
import pytest
import torch

def skip_if_no_gpu(func):
    """Decorator to skip tests if no GPU is available."""
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA is not available"
    )(func)

def run_test_in_subprocess(test_file, test_name=None):
    """Run a test in a subprocess and return the result."""
    cmd = [sys.executable, "-m", "pytest", test_file, "-v", "-s"]
    if test_name:
        cmd.append(f"::{test_name}")
    
    result = subprocess.run(
        cmd,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        capture_output=True,
        text=True
    )
    return result

@skip_if_no_gpu
def test_medical_performance_throughput():
    """Test medical model throughput in a subprocess."""
    result = run_test_in_subprocess(
        "tests/medical/test_medical_performance.py",
        "test_medical_model_throughput"
    )
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    assert result.returncode == 0, f"Test failed: {result.stderr}"

@skip_if_no_gpu
def test_medical_performance_memory_usage():
    """Test medical model memory usage in a subprocess."""
    result = run_test_in_subprocess(
        "tests/medical/test_medical_performance.py",
        "test_medical_model_memory_usage"
    )
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    assert result.returncode == 0, f"Test failed: {result.stderr}"

@skip_if_no_gpu
def test_kv_cache_impact():
    """Test KV cache impact in a subprocess."""
    result = run_test_in_subprocess(
        "tests/medical/test_medical_performance.py",
        "test_kv_cache_impact"
    )
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    assert result.returncode == 0, f"Test failed: {result.stderr}"
