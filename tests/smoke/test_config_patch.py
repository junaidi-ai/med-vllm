import pytest


def test_config_patch_has_hf_config():
    # Under pytest, tests/conftest.py applies patch_transformers() and
    # patches medvllm.config.Config to set a minimal hf_config.
    from medvllm.config import Config

    c = Config(model="dummy-model")
    assert hasattr(c, "hf_config"), "Config should have hf_config set by pytest patches"
    mpe = getattr(getattr(c, "hf_config", None), "max_position_embeddings", None)
    assert mpe is not None, "hf_config.max_position_embeddings should be populated"
