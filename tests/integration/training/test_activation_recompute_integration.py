import pytest
import torch

from medvllm.training.trainer import MedicalModelTrainer, TrainerConfig


class TinyAttnModel(torch.nn.Module):
    def __init__(self, in_dim=8, hid=8):
        super().__init__()
        # Named 'attention' to match default patterns
        self.attention = torch.nn.Linear(in_dim, hid)
        self.head = torch.nn.Linear(hid, 1)

    def forward(self, x):
        h = self.attention(x)
        y = self.head(torch.tanh(h))
        # Return dict with loss as required by trainer
        return {"loss": y.mean()}


@pytest.mark.integration
def test_activation_recompute_integration_train_step():
    # Skip if checkpointing or Linear not available in this environment
    ckpt = pytest.importorskip("torch.utils.checkpoint")
    if not hasattr(torch.nn, "Linear"):
        pytest.skip("torch.nn.Linear not available in this environment")
    # Also require torch.utils.data.Dataset and DataLoader
    if not hasattr(torch, "utils") or not hasattr(torch.utils, "data"):
        pytest.skip("torch.utils.data not available in this environment")
    if not hasattr(torch.utils.data, "Dataset") or not hasattr(torch.utils.data, "DataLoader"):
        pytest.skip("torch.utils.data.* not available in this environment")

    class TinyDataset(torch.utils.data.Dataset):
        def __init__(self, n=4, d=8):
            self.n = n
            self.d = d

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            # Trainer expects dict so model(**batch) works
            return {"x": torch.randn(self.d)}

    model = TinyAttnModel()
    cfg = TrainerConfig(
        device="cpu",
        activation_recompute=True,
        # default patterns cover 'attention'
        num_epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        scheduler="none",
        enable_tensorboard=False,
        enable_wandb=False,
        channels_last=False,
    )

    trainer = MedicalModelTrainer(model, cfg)

    # Prepare and verify wrapping occurred
    trainer.prepare_for_training()
    assert hasattr(trainer.model.attention, "inner"), "Expected attention to be checkpoint-wrapped"

    # Build a tiny dataset and run a brief train loop (one epoch)
    ds = TinyDataset(n=4, d=8)
    trainer.train(ds, eval_dataset=None, output_dir="./toy_cli_run2")
