import os
import subprocess
import sys
from typing import Optional, Tuple


def run_cli(
    args, input_text: Optional[str] = None, timeout: int = 20
) -> Tuple[int, str, str, subprocess.CompletedProcess]:
    """Run the CLI as a subprocess: python -m medvllm.cli <args>.
    Returns (returncode, stdout, stderr, completed_process).
    """
    cmd = [sys.executable, "-m", "medvllm.cli", *args]
    env = os.environ.copy()
    # Keep output deterministic for tests
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Use fake engine for generation to keep tests lightweight
    env.setdefault("MEDVLLM_TEST_FAKE_ENGINE", "1")
    cp = subprocess.run(
        cmd,
        input=(input_text.encode("utf-8") if input_text is not None else None),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        env=env,
    )
    return (
        cp.returncode,
        cp.stdout.decode("utf-8", errors="replace"),
        cp.stderr.decode("utf-8", errors="replace"),
        cp,
    )
