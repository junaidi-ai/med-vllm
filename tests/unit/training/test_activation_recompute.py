import torch
import pytest

from medvllm.training.trainer import MedicalModelTrainer, TrainerConfig


class IdentityModule(torch.nn.Module):
    def forward(self, *args, **kwargs):  # type: ignore[override]
        # Return first arg if present, else zero scalar tensor
        if args:
            return args[0]
        return torch.tensor(0.0)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Name includes 'attention' so default patterns will match
        self.attention = IdentityModule()
        self.head = IdentityModule()

    def forward(self, x):
        x = self.attention(x)
        y = self.head(x)
        # Return a dict with a scalar-like for compatibility; value unused in this test
        return {"loss": torch.as_tensor(0.0)}


def test_activation_recompute_wraps_default_pattern():
    pytest.importorskip("torch.utils.checkpoint")
    model = TinyModel()
    cfg = TrainerConfig(
        device="cpu",
        activation_recompute=True,
        gradient_checkpointing=False,
        num_epochs=1,
        batch_size=1,
    )
    trainer = MedicalModelTrainer(model, cfg)
    # Call the targeted wrapper directly to avoid full training setup
    trainer._maybe_enable_activation_recompute(
        trainer.model,
        patterns=["attention", "conv3d"],
    )

    # After prepare, the 'attention' submodule should be wrapped with CheckpointWrapper
    wrapped = getattr(trainer.model, "attention")
    # Identify wrapper via sentinel attribute added in wrapper class
    assert isinstance(wrapped, torch.nn.Module)
    assert hasattr(wrapped, "inner"), "Expected CheckpointWrapper with 'inner' attribute"

    # Do not execute forward; checkpoint requires tensor semantics not guaranteed in mock torch


def test_activation_recompute_wraps_custom_pattern():
    pytest.importorskip("torch.utils.checkpoint")

    class BlockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Name includes 'block' for custom match
            self.block = IdentityModule()
            self.head = IdentityModule()

        def forward(self, x):
            x = self.block(x)
            _ = self.head(x)
            return {"loss": torch.as_tensor(0.0)}

    model = BlockModel()
    cfg = TrainerConfig(
        device="cpu",
        activation_recompute=True,
        activation_recompute_patterns=["block"],
        num_epochs=1,
        batch_size=1,
    )
    trainer = MedicalModelTrainer(model, cfg)
    trainer._maybe_enable_activation_recompute(
        trainer.model,
        patterns=["block"],
    )

    wrapped = getattr(trainer.model, "block")
    assert isinstance(wrapped, torch.nn.Module)
    assert hasattr(wrapped, "inner"), "Expected CheckpointWrapper on custom pattern"

    # Do not execute forward in mocked torch environment


def test_activation_recompute_does_not_wrap_non_matching():
    pytest.importorskip("torch.utils.checkpoint")

    class NoMatchModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Names do not include default patterns 'attention' or 'conv3d'
            self.encoder = IdentityModule()
            self.head = IdentityModule()

        def forward(self, x):
            x = self.encoder(x)
            _ = self.head(x)
            return {"loss": torch.as_tensor(0.0)}

    model = NoMatchModel()
    cfg = TrainerConfig(
        device="cpu",
        activation_recompute=True,
        # Use default patterns
    )
    trainer = MedicalModelTrainer(model, cfg)
    trainer._maybe_enable_activation_recompute(trainer.model, patterns=["attention", "conv3d"])

    # Ensure non-matching modules remain unwrapped (no 'inner' sentinel)
    assert not hasattr(getattr(trainer.model, "encoder"), "inner")
    assert not hasattr(getattr(trainer.model, "head"), "inner")
