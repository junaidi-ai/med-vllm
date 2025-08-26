#!/usr/bin/env python3
"""
Minimal examples for using medvllm.training.MedicalModelTrainer with:
- Single-GPU AMP (fp16/bf16)
- DDP via torchrun
- DeepSpeed (optional)

This uses a tiny synthetic dataset and a tiny model that returns a loss.
No external datasets or model checkpoints are required.

Example runs:

1) Single GPU, BF16 AMP (recommended on modern GPUs):
   python3 examples/trainer_usage_example.py --use-amp --amp-dtype bf16

2) Single GPU, FP16 AMP:
   python3 examples/trainer_usage_example.py --use-amp --amp-dtype fp16

3) Multi-GPU DDP (2 GPUs example):
   torchrun --nproc_per_node=2 examples/trainer_usage_example.py --use-ddp

4) DeepSpeed single GPU:
   python3 examples/trainer_usage_example.py --deepspeed

5) DeepSpeed multi-GPU (2 GPUs example):
   deepspeed --num_gpus=2 examples/trainer_usage_example.py --deepspeed

Optional flags:
  --gradient-checkpointing  Enable gradient checkpointing
  --channels-last           Use channels_last memory format (helpful for imaging conv nets)

"""

from __future__ import annotations

import argparse
from typing import Dict, Any
import os
import sys

# Ensure repository root is on sys.path when running from source tree
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from torch import nn
from torch.utils.data import Dataset

from medvllm.training import MedicalModelTrainer, TrainerConfig


class TinyDataset(Dataset):
    def __init__(self, length: int = 256, dim: int = 128, num_classes: int = 4):
        self.length = length
        self.x = torch.randn(length, dim)
        self.y = torch.randint(0, num_classes, (length,))

    def __len__(self) -> int:  # type: ignore[override]
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # type: ignore[override]
        return {"input": self.x[idx], "label": self.y[idx]}


class TinyModel(nn.Module):
    def __init__(self, dim: int = 128, num_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input: torch.Tensor, label: torch.Tensor) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        logits = self.net(input)
        loss = self.loss_fn(logits, label)
        return {"loss": loss}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trainer usage example")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=32)

    p.add_argument("--use-amp", action="store_true")
    p.add_argument(
        "--amp-dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="AMP dtype to use when --use-amp is set",
    )

    p.add_argument("--use-ddp", action="store_true", help="Enable DDP (torchrun)")
    p.add_argument("--deepspeed", action="store_true", help="Enable DeepSpeed (optional)")

    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--channels-last", action="store_true")

    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--log-every", type=int, default=20)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    cfg = TrainerConfig(
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        log_every=args.log_every,
        use_amp=args.use_amp,
        amp_dtype=amp_dtype,
        use_ddp=args.use_ddp,
        deepspeed=args.deepspeed,
        gradient_accumulation_steps=2,
        gradient_checkpointing=args.gradient_checkpointing,
        channels_last=args.channels_last,
        cudnn_benchmark=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        save_every_epochs=1,
    )

    model = TinyModel()

    # toy data
    train_ds = TinyDataset(length=512)
    eval_ds = TinyDataset(length=128)

    trainer = MedicalModelTrainer(model, cfg)
    trainer.train(train_ds, eval_ds, output_dir="./tiny_trained_model")


if __name__ == "__main__":
    main()
