import math
from typing import Dict, Any, List, Tuple
import pytest

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from medvllm.training.trainer import MedicalModelTrainer, TrainerConfig

# Minimal fallbacks for environments lacking some nn modules
try:
    Identity = nn.Identity  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover

    class Identity(nn.Module):
        def forward(self, x):
            return x


try:
    Dropout = nn.Dropout  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover

    class Dropout(nn.Module):
        def __init__(self, p: float = 0.5):
            super().__init__()
            self.p = p

        def forward(self, x):
            return x


Dropout2d = getattr(nn, "Dropout2d", Dropout)
Dropout3d = getattr(nn, "Dropout3d", Dropout)


class TinyClsDataset(Dataset):
    def __init__(self, n: int = 32, in_dim: int = 8, num_classes: int = 2, seed: int = 0):
        try:
            torch.manual_seed(seed)
        except Exception:
            pass
        self.x = torch.randn(n, in_dim)
        try:
            self.y = torch.tensor([i % num_classes for i in range(n)], dtype=torch.long)
        except Exception:
            # Last-resort: create a Python list; trainer will convert as needed
            ys = [i % num_classes for i in range(n)]
            self.y = torch.LongTensor(ys) if hasattr(torch, 'LongTensor') else torch.tensor(ys)

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"x": self.x[idx], "y": self.y[idx]}


class TinyModel(nn.Module):
    def __init__(
        self, in_dim: int = 8, hidden: int = 8, num_classes: int = 2, dropout: float = 0.0
    ):
        super().__init__()
        # Keep attribute names for layer-wise LR matching
        # encoder layers
        self.encoder_lin1 = nn.Linear(in_dim, hidden)
        # use functional relu to avoid dependency on nn.ReLU class
        if hasattr(nn, "Dropout"):
            self.encoder_dropout = Dropout(p=dropout)
        else:
            self.encoder_dropout = Identity()
        self.encoder_lin2 = nn.Linear(hidden, hidden)
        # use functional relu
        # classifier layers
        self.classifier_lin = nn.Linear(hidden, num_classes)

    # Provide properties to expose 'encoder' and 'classifier' in parameter names
    @property
    def encoder(self):  # type: ignore[override]
        # a dummy object exposing named parameters prefix 'encoder'
        # Not used directly; names include 'encoder_' via layer attrs
        return self

    @property
    def classifier(self):  # type: ignore[override]
        return self

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
        z = self.encoder_lin1(x)
        z = F.relu(z)
        z = self.encoder_dropout(z)
        z = self.encoder_lin2(z)
        z = F.relu(z)
        logits = self.classifier_lin(z)
        loss = nn.functional.cross_entropy(logits, y)
        return {"loss": loss, "logits": logits}

    # Provide named_parameters for minimal environments
    def named_parameters(self, prefix: str = "", recurse: bool = True):  # type: ignore[override]
        # encoder params
        yield ("encoder.lin1.weight", self.encoder_lin1.weight)
        yield ("encoder.lin1.bias", self.encoder_lin1.bias)
        yield ("encoder.lin2.weight", self.encoder_lin2.weight)
        yield ("encoder.lin2.bias", self.encoder_lin2.bias)
        # classifier params
        yield ("classifier.lin.weight", self.classifier_lin.weight)
        yield ("classifier.lin.bias", self.classifier_lin.bias)


def _build_trainer(
    optimizer: str, scheduler: str
) -> Tuple[MedicalModelTrainer, TrainerConfig, TinyModel, TinyClsDataset]:
    model = TinyModel()
    ds = TinyClsDataset(n=32)

    cfg = TrainerConfig(
        learning_rate=1e-3,
        weight_decay=0.01,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_steps=2,
        total_steps=10,
        step_size=3,
        gamma=0.9,
        t_max=10,
        eta_min=0.0,
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
        batch_size=8,
        num_epochs=1,
        device="cpu",
        use_amp=False,
        grad_clip_mode="norm",
        max_grad_norm=0.1,
        # layer-wise LR rules to exercise grouping
        layer_wise_lrs=[("encoder", 0.5), ("classifier", 2.0)],
    )
    trainer = MedicalModelTrainer(model, cfg)
    return trainer, cfg, model, ds


def test_param_groups_and_dropout_override():
    trainer, cfg, model, _ = _build_trainer("adamw", "none")
    # Set dropout override and prepare
    setattr(cfg, "dropout_prob", 0.1)
    # Environment guard: need named_parameters to build param groups
    if not hasattr(trainer.model, "named_parameters"):
        pytest.skip("Minimal torch environment: named_parameters unavailable")
    # Also require tensors with requires_grad attribute
    try:
        any(next(iter(trainer.model.named_parameters()))[1].requires_grad for _ in [0])  # type: ignore[attr-defined]
    except Exception:
        pytest.skip("Minimal torch environment: parameters lack requires_grad")
    trainer.prepare_for_training()

    # Verify dropout override applied if Dropout exists in this env
    if hasattr(nn, "Dropout"):
        drops = [m for m in model.modules() if isinstance(m, (Dropout, Dropout2d, Dropout3d))]
        assert drops, "No dropout modules found to override"
        for d in drops:
            assert math.isclose(d.p, 0.1, rel_tol=0, abs_tol=1e-6)

    # Inspect optimizer param groups
    opt = trainer.optimizer
    assert opt is not None

    # Collect unique (lr, weight_decay) pairs
    pairs = {(pg.get("lr"), pg.get("weight_decay")) for pg in opt.param_groups}
    lrs = {pg.get("lr") for pg in opt.param_groups}

    # Expected LR multipliers present
    assert any(abs(lr - 0.0005) < 1e-9 for lr in lrs), "encoder 0.5x lr not found"
    assert any(abs(lr - 0.002) < 1e-9 for lr in lrs), "classifier 2.0x lr not found"

    # Expect at least one group with weight_decay == 0.0 (bias/no-norm)
    assert any(abs((wd or 0.0) - 0.0) < 1e-12 for (_, wd) in pairs)

    # Ensure some groups have non-zero weight decay
    assert any((wd or 0.0) > 0 for (_, wd) in pairs)


def test_optimizers_and_schedulers_and_clipping_runs():
    # Try a small subset to keep smoke fast
    cases: List[Tuple[str, str]] = [
        ("adamw", "linear_warmup"),
        ("adam", "step"),
        ("sgd", "cosine"),
        ("adamw", "one_cycle"),
        ("adamw", "none"),
    ]
    for opt_name, sched_name in cases:
        trainer, cfg, model, ds = _build_trainer(opt_name, sched_name)
        if not hasattr(trainer.model, "named_parameters"):
            pytest.skip("Minimal torch environment: named_parameters unavailable")
        try:
            any(next(iter(trainer.model.named_parameters()))[1].requires_grad for _ in [0])  # type: ignore[attr-defined]
        except Exception:
            pytest.skip("Minimal torch environment: parameters lack requires_grad")
        trainer.train(ds, None, output_dir="./toy_cli_run2")

        # After training one epoch, ensure scheduler (if any) advanced some steps
        if trainer.scheduler is not None:
            # Many schedulers expose last_epoch; ensure it progressed
            last_epoch = getattr(trainer.scheduler, "last_epoch", None)
            if last_epoch is not None:
                assert last_epoch >= 0

        # Basic sanity: a forward pass still works
        model.eval()
        with torch.no_grad():
            batch = ds[0]
            out = model(**batch)
            assert isinstance(out, dict) and "loss" in out
