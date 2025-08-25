from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, Optional

import torch
from torch.utils.data import DataLoader

# Optional XLA (TPU) support
try:  # pragma: no cover - only active when torch_xla is installed
    import torch_xla.core.xla_model as xm  # type: ignore
except Exception:  # noqa: BLE001 - broad for optional dep
    xm = None  # type: ignore[assignment]


@dataclass
class TrainerConfig:
    # optimization
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    max_grad_norm: Optional[float] = 1.0
    gradient_accumulation_steps: int = 1

    # training loop
    num_epochs: int = 1
    batch_size: int = 8
    eval_batch_size: int = 8
    log_every: int = 50

    # mixed precision
    use_amp: bool = False
    amp_dtype: torch.dtype = torch.float16

    # scheduler (linear warmup decay minimal)
    warmup_steps: int = 0
    total_steps: Optional[int] = None  # if None, computed from dataset

    # device
    device: Optional[str] = None  # "cuda", "cpu"; if None auto

    # checkpointing
    save_every_epochs: int = 1
    save_optimizer: bool = True
    save_scheduler: bool = True


@dataclass
class EvalResult:
    loss: float
    metrics: Dict[str, float]


class MedicalModelTrainer:
    """
    A lightweight, dependency-minimal Trainer for fine-tuning.

    Expects the model to follow a standard PyTorch interface where
    model(**batch) returns either a dict with 'loss' or a loss tensor directly.
    """

    def __init__(self, model: torch.nn.Module, config: TrainerConfig):
        self.model = model
        self.config = config

        # Resolve device, including optional XLA
        resolved = self._resolve_device()
        if resolved == "xla":
            if xm is None:
                raise RuntimeError(
                    "XLA device requested but torch_xla is not available. Install torch-xla to use TPU."
                )
            self.device = xm.xla_device()
        else:
            self.device = torch.device(resolved)

        # AMP: CUDA GradScaler only; XLA uses torch.autocast("xla") instead
        self.scaler: Optional[torch.cuda.amp.GradScaler] = (
            torch.cuda.amp.GradScaler()
            if (
                self.config.use_amp
                and isinstance(self.device, torch.device)
                and self.device.type == "cuda"
            )
            else None
        )

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None

    def _resolve_device(self) -> str:
        if self.config.device is not None:
            # allow explicit 'xla' to select TPU when available
            dev = self.config.device.lower()
            if dev.startswith("xla"):
                return "xla"
            return dev
        # auto: prefer CUDA, else CPU (do not auto-pick XLA)
        return "cuda" if torch.cuda.is_available() else "cpu"

    def prepare_for_training(self) -> None:
        self.model.to(self.device)
        self.model.train()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas,
            eps=self.config.eps,
        )

    def _maybe_build_scheduler(self, total_steps: int) -> None:
        if self.config.warmup_steps == 0 and not self.config.total_steps:
            self.scheduler = None
            return
        # simple linear warmup, then constant; minimal and dependency-free
        warmup = max(0, self.config.warmup_steps)
        t_total = self.config.total_steps or total_steps

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return float(step) / float(max(1, warmup))
            return 1.0

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)  # type: ignore[arg-type]

    def train(
        self,
        train_dataset: Iterable[Dict[str, Any]] | torch.utils.data.Dataset,
        eval_dataset: Optional[Iterable[Dict[str, Any]] | torch.utils.data.Dataset] = None,
        output_dir: str = "./fine_tuned_model",
    ) -> None:
        self.prepare_for_training()

        train_loader = (
            train_dataset
            if isinstance(train_dataset, DataLoader)
            else DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        )
        eval_loader: Optional[DataLoader] = None
        if eval_dataset is not None:
            eval_loader = (
                eval_dataset
                if isinstance(eval_dataset, DataLoader)
                else DataLoader(eval_dataset, batch_size=self.config.eval_batch_size)
            )

        total_steps = self.config.total_steps or (len(train_loader) * self.config.num_epochs)
        self._maybe_build_scheduler(total_steps)

        global_step = 0
        accum_steps = max(1, self.config.gradient_accumulation_steps)

        for epoch in range(self.config.num_epochs):
            self.model.train()
            running_loss = 0.0

            for step, batch in enumerate(train_loader, start=1):
                batch = self._move_to_device(batch)
                loss = self._forward_loss(batch) / accum_steps

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if step % accum_steps == 0:
                    if self.config.max_grad_norm is not None:
                        if self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)  # type: ignore[arg-type]
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )

                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)  # type: ignore[arg-type]
                        self.scaler.update()
                    else:
                        if (
                            xm is not None
                            and isinstance(self.device, torch.device)
                            and self.device.type == "xla"
                        ):
                            # XLA requires optimizer_step for correct graph execution
                            xm.optimizer_step(self.optimizer, barrier=True)  # type: ignore[arg-type]
                        else:
                            self.optimizer.step()  # type: ignore[union-attr]

                    if self.scheduler is not None:
                        self.scheduler.step()  # type: ignore[union-attr]

                    self.optimizer.zero_grad(set_to_none=True)  # type: ignore[union-attr]
                    global_step += 1

                running_loss += loss.item() * accum_steps
                if global_step % max(1, self.config.log_every) == 0:
                    self._log(
                        {
                            "epoch": epoch,
                            "step": global_step,
                            "loss": running_loss / max(1, step),
                            "lr": self.optimizer.param_groups[0]["lr"] if self.optimizer else None,
                        }
                    )

            # end epoch: evaluate + save checkpoint
            if eval_loader is not None:
                eval_res = self.evaluate(eval_loader)
                self._log({"epoch": epoch, "eval_loss": eval_res.loss, **eval_res.metrics})

            if (epoch + 1) % max(1, self.config.save_every_epochs) == 0:
                self._save_checkpoint(output_dir, f"checkpoint-epoch{epoch+1}")

        # final save
        self._save_model(output_dir)

    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader) -> EvalResult:
        self.model.eval()
        total_loss = 0.0
        count = 0
        for batch in eval_loader:
            batch = self._move_to_device(batch)
            loss = self._forward_loss(batch)
            total_loss += float(loss.item())
            count += 1
        avg_loss = total_loss / max(1, count)
        return EvalResult(loss=avg_loss, metrics={})

    @torch.no_grad()
    def inference(self, inputs: Dict[str, Any]) -> Any:
        self.model.eval()
        inputs = self._move_to_device(inputs)
        return self.model(**inputs)

    def _forward_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        if self.config.use_amp:
            # CUDA autocast
            if isinstance(self.device, torch.device) and self.device.type == "cuda":
                with torch.cuda.amp.autocast(dtype=self.config.amp_dtype):
                    out = self.model(**batch)
            # XLA autocast (bfloat16 recommended)
            elif (
                xm is not None
                and isinstance(self.device, torch.device)
                and self.device.type == "xla"
            ):
                # torch.autocast supports device_type="xla" when torch_xla is installed
                dtype = (
                    torch.bfloat16
                    if self.config.amp_dtype not in (torch.bfloat16, torch.float16)
                    else self.config.amp_dtype
                )
                with torch.autocast(device_type="xla", dtype=dtype):
                    out = self.model(**batch)
            else:
                out = self.model(**batch)
        else:
            out = self.model(**batch)

        if isinstance(out, dict):
            loss = out.get("loss")
            if loss is None:
                raise ValueError("Model output dict must contain 'loss' during training.")
            return loss
        if torch.is_tensor(out):
            return out
        raise TypeError("Model forward must return a loss tensor or a dict with 'loss'.")

    def _move_to_device(self, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: self._move_to_device(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return type(data)(self._move_to_device(v) for v in data)
        if torch.is_tensor(data):
            return data.to(self.device)
        return data

    def _save_checkpoint(self, output_dir: str, name: str) -> None:
        import os

        os.makedirs(output_dir, exist_ok=True)
        ckpt_path = os.path.join(output_dir, f"{name}.pt")
        payload: Dict[str, Any] = {
            "model_state": self.model.state_dict(),
            "config": asdict(self.config),
        }
        if self.config.save_optimizer and self.optimizer is not None:
            payload["optimizer_state"] = self.optimizer.state_dict()
        if self.config.save_scheduler and self.scheduler is not None:
            payload["scheduler_state"] = self.scheduler.state_dict()
        if self.scaler is not None:
            payload["scaler_state"] = self.scaler.state_dict()
        torch.save(payload, ckpt_path)

    def _save_model(self, output_dir: str) -> None:
        import os

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    def _log(self, data: Dict[str, Any]) -> None:
        # minimal logging; can be replaced by structured logging later
        msg = " ".join(f"{k}={v}" for k, v in data.items())
        print(f"[trainer] {msg}")
