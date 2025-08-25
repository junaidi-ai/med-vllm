from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

import click
import torch
from torch import nn
from torch.utils.data import Dataset

from medvllm.cli.utils import console
from medvllm.training import MedicalModelTrainer, TrainerConfig


class _ToyDataset(Dataset):
    """Simple 2D points with linear separability.

    y = 1 if w.x + b + noise > 0 else 0
    """

    def __init__(self, n: int = 512, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.X = torch.randn(n, 2, generator=g)
        w = torch.tensor([1.5, -0.75])
        b = 0.1
        logits = self.X @ w + b + 0.1 * torch.randn(n, generator=g)
        self.y = (logits > 0).long()

    def __len__(self) -> int:  # type: ignore[override]
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        x = self.X[idx]
        y = self.y[idx]
        return {"input": x, "labels": y}


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self, input: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:  # type: ignore[override]
        logits = self.net(input)
        out: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            out["loss"] = self.loss_fn(logits, labels)
        return out


@click.group(name="training")
def training_group() -> None:
    """Training utilities (experimental)."""


@training_group.command(name="train")
@click.option("--epochs", type=int, default=1, show_default=True, help="Number of epochs")
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--lr", type=float, default=5e-3, show_default=True, help="Learning rate")
@click.option(
    "--amp/--no-amp", default=False, show_default=True, help="Use mixed precision (CUDA only)"
)
@click.option(
    "--output", type=click.Path(file_okay=False), default="./toy_finetune", show_default=True
)
@click.option("--toy", is_flag=True, help="Run a built-in toy model/dataset example")
def train_cmd(epochs: int, batch_size: int, lr: float, amp: bool, output: str, toy: bool) -> None:
    """Train a model. Currently supports a built-in toy example via --toy.

    Example:
      python -m medvllm.cli training train --toy --epochs 3 --batch-size 32 --lr 1e-3
    """
    if not toy:
        raise click.UsageError("Only --toy mode is implemented in this minimal training CLI.")

    console.print("[bold]Running toy training example[/bold]")

    # Build toy data/model
    train_ds = _ToyDataset(n=1024, seed=0)
    eval_ds = _ToyDataset(n=256, seed=1)
    model = _ToyModel()

    # Trainer config
    cfg = TrainerConfig(
        learning_rate=lr,
        num_epochs=epochs,
        batch_size=batch_size,
        eval_batch_size=batch_size,
        use_amp=amp,
        log_every=20,
        save_every_epochs=epochs,  # save at end only by default
    )

    console.print(f"config: {asdict(cfg)}")

    trainer = MedicalModelTrainer(model, cfg)
    trainer.train(train_ds, eval_dataset=eval_ds, output_dir=os.fspath(output))

    console.print(f"[green]Done. Saved artifacts under[/green] {output}")


def register_commands(cli: Any) -> None:
    cli.add_command(training_group)
