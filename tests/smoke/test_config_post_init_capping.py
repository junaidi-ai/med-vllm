import sys
from unittest.mock import MagicMock


def test_config_post_init_capping_and_no_hf_call(monkeypatch):
    # Arrange: spy on AutoConfig.from_pretrained
    transformers = sys.modules["transformers"]
    assert hasattr(transformers, "AutoConfig")

    # Replace with a spy MagicMock
    spy = MagicMock(name="AutoConfig.from_pretrained")
    # Some mocks make AutoConfig a MagicMock itself; handle accordingly
    auto_config = transformers.AutoConfig
    original = getattr(auto_config, "from_pretrained", None)
    monkeypatch.setattr(auto_config, "from_pretrained", spy, raising=False)

    # Act: construct Config with very large max_model_len to test capping
    from medvllm.config import Config

    cfg = Config(model="dummy-model", max_model_len=999_999)

    # Assert: AutoConfig.from_pretrained was NOT called by patched __post_init__
    spy.assert_not_called()

    # Assert: hf_config is present and has max_position_embeddings
    assert hasattr(cfg, "hf_config"), "hf_config should be set by patch"
    mpe = getattr(cfg.hf_config, "max_position_embeddings", None)
    assert mpe is not None and mpe > 0

    # Assert: max_model_len was capped to hf_config.max_position_embeddings
    assert cfg.max_model_len == mpe

    # Cleanup: restore original if it existed (monkeypatch will undo automatically at test end)
    if original is not None:
        monkeypatch.setattr(auto_config, "from_pretrained", original, raising=False)
