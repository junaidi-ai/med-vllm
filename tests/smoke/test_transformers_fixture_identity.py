import sys


def test_mock_transformers_fixture_identity(mock_transformers):
    # Fixture should return the live transformers module from sys.modules
    assert (
        mock_transformers is sys.modules["transformers"]
    ), "Fixture must return live sys.modules['transformers']"

    # Basic attributes should be present
    assert hasattr(mock_transformers, "AutoConfig"), "Mock transformers must expose AutoConfig"

    # AutoConfig.from_pretrained should return a config object with max_position_embeddings
    cfg = mock_transformers.AutoConfig.from_pretrained("any-model")
    assert hasattr(cfg, "max_position_embeddings") and cfg.max_position_embeddings is not None
