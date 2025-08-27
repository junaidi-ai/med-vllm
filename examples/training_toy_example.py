"""Tiny example: fine-tune a toy classifier on synthetic data with MedicalModelTrainer.

Run:
  python examples/training_toy_example.py --epochs 2 --batch-size 32 --lr 1e-3 --output ./toy_run

You can also use the CLI equivalent:
  python -m medvllm.cli training train --toy --epochs 2 --batch-size 32 --lr 1e-3 --output ./toy_run
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import torch
from torch import nn
from torch.utils.data import Dataset

from medvllm.training import MedicalModelTrainer, TrainerConfig


class ToyDataset(Dataset):
    def __init__(self, n: int = 1024, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.X = torch.randn(n, 2, generator=g)
        w = torch.tensor([1.5, -0.75])
        b = 0.1
        logits = self.X @ w + b + 0.1 * torch.randn(n, generator=g)
        self.y = (logits > 0).long()

    def __len__(self) -> int:  # type: ignore[override]
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {"input": self.X[idx], "labels": self.y[idx]}


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input: torch.Tensor, labels: torch.Tensor | None = None):  # type: ignore[override]
        logits = self.net(input)
        out: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            out["loss"] = self.loss_fn(logits, labels)
        return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output", type=str, default="./local_runs/toy_finetune")
    args = parser.parse_args()

    train_ds = ToyDataset(n=1024, seed=0)
    eval_ds = ToyDataset(n=256, seed=1)
    model = ToyModel()

    cfg = TrainerConfig(
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        use_amp=args.amp,
        log_every=20,
        save_every_epochs=args.epochs,
    )

    print("config:", asdict(cfg))
    trainer = MedicalModelTrainer(model, cfg)
    trainer.train(train_ds, eval_dataset=eval_ds, output_dir=args.output)
    print("Done. Saved artifacts under", args.output)


if __name__ == "__main__":
    main()
