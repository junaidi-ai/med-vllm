from __future__ import annotations

import importlib
import importlib.util
import json
import os
from dataclasses import asdict
from typing import Any, Callable, Tuple

import click
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

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
    "--device",
    type=click.Choice(["auto", "cpu", "cuda", "xla"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Device to train on. 'xla' requires torch_xla installed.",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False),
    default="./local_runs/toy_finetune",
    show_default=True,
)
@click.option("--toy", is_flag=True, help="Run the built-in toy example")
@click.option(
    "--entrypoint",
    type=str,
    default=None,
    help="Python entrypoint in form 'module.sub:func' or '/path/to/file.py:func' that returns (model, train_ds, eval_ds_or_none)",
)
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Optional JSON config passed to the entrypoint function as a dict",
)
# Checkpointing / resume
@click.option("--save-every-steps", type=int, default=None, help="Save checkpoint every N steps")
@click.option(
    "--resume-from",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to a checkpoint .pt to resume from",
)
# Export
@click.option(
    "--export-torchscript/--no-export-torchscript",
    default=False,
    show_default=True,
    help="Export TorchScript after training",
)
@click.option(
    "--export-onnx/--no-export-onnx",
    default=False,
    show_default=True,
    help="Export ONNX after training",
)
@click.option("--onnx-opset", type=int, default=13, show_default=True, help="ONNX opset version")
@click.option(
    "--export-auto-example/--no-export-auto-example",
    default=True,
    show_default=True,
    help="Auto-derive export input example from the first training batch (or toy example)",
)
# Versioning
@click.option("--no-versioning", is_flag=True, default=False, help="Disable model versioning")
def train_cmd(
    epochs: int,
    batch_size: int,
    lr: float,
    amp: bool,
    device: str,
    output: str,
    toy: bool,
    entrypoint: str | None,
    config: str | None,
    save_every_steps: int | None,
    resume_from: str | None,
    export_torchscript: bool,
    export_onnx: bool,
    onnx_opset: int,
    export_auto_example: bool,
    no_versioning: bool,
) -> None:
    """Train a model via either --toy or a custom --entrypoint.

    Examples:
      python -m medvllm.cli training train --toy --epochs 3 --batch-size 32 --lr 1e-3
      python -m medvllm.cli training train --entrypoint mypkg.mymod:build --config config.json --epochs 1
    """
    if not toy and not entrypoint:
        raise click.UsageError("Provide either --toy or --entrypoint.")

    if toy and entrypoint:
        raise click.UsageError("Choose only one of --toy or --entrypoint.")

    # Build data/model
    export_input_example: dict[str, Any] | None = None
    if toy:
        console.print("[bold]Running toy training example[/bold]")
        train_ds = _ToyDataset(n=1024, seed=0)
        eval_ds = _ToyDataset(n=256, seed=1)
        model = _ToyModel()
        if export_auto_example and (export_torchscript or export_onnx):
            # Toy shapes: input: (B,2), labels: (B,)
            export_input_example = {"input": torch.randn(1, 2), "labels": torch.tensor([1])}
    else:
        cfg_payload: dict[str, Any] = {}
        if config is not None:
            with open(config, "r", encoding="utf-8") as f:
                cfg_payload = json.load(f)
        fn = _resolve_entrypoint(entrypoint)  # type: ignore[arg-type]
        result = fn(cfg_payload)
        try:
            model, train_ds, eval_ds = result
        except Exception as e:  # pragma: no cover - helpful error message
            raise click.ClickException(
                "Entrypoint must return a tuple (model, train_dataset, eval_dataset_or_none)."
            ) from e

        # Auto derive export example: take first batch from train_ds
        if export_auto_example and (export_torchscript or export_onnx):
            try:
                loader = DataLoader(train_ds, batch_size=1, shuffle=False)
                first_batch = next(iter(loader))
                if isinstance(first_batch, dict):
                    export_input_example = {
                        k: (v if torch.is_tensor(v) else torch.as_tensor(v))
                        for k, v in first_batch.items()
                    }
                else:
                    console.print(
                        "[yellow]Auto example: first batch is not a dict; skipping.[/yellow]"
                    )
            except Exception as e:
                console.print(f"[yellow]Auto example extraction failed: {e}[/yellow]")

    # Trainer config
    cfg = TrainerConfig(
        learning_rate=lr,
        num_epochs=epochs,
        batch_size=batch_size,
        eval_batch_size=batch_size,
        use_amp=amp,
        log_every=20,
        save_every_epochs=epochs,  # save at end only by default
        save_every_steps=save_every_steps,
        resume_from=resume_from,
        export_torchscript=export_torchscript,
        export_onnx=export_onnx,
        onnx_opset=onnx_opset,
        export_input_example=export_input_example,
        model_versioning=not no_versioning,
        device=None if device.lower() == "auto" else device.lower(),
    )

    console.print(f"config: {asdict(cfg)}")

    trainer = MedicalModelTrainer(model, cfg)
    trainer.train(train_ds, eval_dataset=eval_ds, output_dir=os.fspath(output))

    console.print(f"[green]Done. Saved artifacts under[/green] {output}")


def _resolve_entrypoint(spec: str) -> Callable[[dict[str, Any]], Tuple[Any, Any, Any | None]]:
    if ":" not in spec:
        raise click.UsageError("--entrypoint must be in the form 'module.or.path:func'")
    mod_part, func_name = spec.split(":", 1)

    # If it's a filesystem path
    if os.path.exists(mod_part):
        module_name = os.path.splitext(os.path.basename(mod_part))[0]
        spec_obj = importlib.util.spec_from_file_location(module_name, mod_part)
        if spec_obj is None or spec_obj.loader is None:
            raise click.ClickException(f"Failed to load module from path: {mod_part}")
        module = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(module)
    else:
        module = importlib.import_module(mod_part)

    fn = getattr(module, func_name, None)
    if not callable(fn):
        raise click.ClickException(f"Function '{func_name}' not found/callable in '{mod_part}'.")
    return fn  # type: ignore[return-value]


def register_commands(cli: Any) -> None:
    cli.add_command(training_group)
