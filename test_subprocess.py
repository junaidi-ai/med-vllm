import subprocess
import sys

def test_import_transformers():
    result = subprocess.run(
        [sys.executable, "-c", "import transformers; from transformers.tokenization_utils_base import PreTrainedTokenizerBase; print('Imports successful')"],
        capture_output=True,
        text=True
    )
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    assert result.returncode == 0, f"Import failed: {result.stderr}"
