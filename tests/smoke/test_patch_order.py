import sys
from unittest.mock import MagicMock


def test_patch_transformers_applied_before_tests_run():
    # pytest_configure should have run patch_transformers before any tests
    transformers_mod = sys.modules.get("transformers")
    assert transformers_mod is not None, "transformers module should be present in sys.modules"
    assert isinstance(
        transformers_mod, MagicMock
    ), "transformers should be a MagicMock set by patch_transformers()"

    # Importing medvllm.config should not replace the mocked transformers
    from medvllm import config as _  # noqa: F401

    assert sys.modules["transformers"] is transformers_mod
