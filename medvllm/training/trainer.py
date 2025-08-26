from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
import warnings
import torch.distributed as dist

# Optional FSDP support (PyTorch native)
try:  # pragma: no cover - only active when torch.distributed.fsdp is available
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        StateDictType,
        FullStateDictConfig,
    )
except Exception:  # noqa: BLE001 - broad for optional dep
    FSDP = None  # type: ignore[assignment]
    StateDictType = None  # type: ignore[assignment]
    FullStateDictConfig = None  # type: ignore[assignment]

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
    amp_dtype: torch.dtype = torch.float16  # torch.float16 or torch.bfloat16

    # scheduler (linear warmup decay minimal)
    warmup_steps: int = 0
    total_steps: Optional[int] = None  # if None, computed from dataset

    # device
    device: Optional[str] = None  # "cuda", "cpu", "xla"; if None auto

    # distributed
    use_ddp: bool = False  # torch.distributed DistributedDataParallel
    ddp_backend: str = "nccl"  # or gloo for CPU
    deepspeed: bool = False  # initialize with deepspeed if available
    deepspeed_config: Optional[Any] = None  # dict or path-like to DS config (e.g., ZeRO stages)
    ddp_find_unused_parameters: bool = False

    # checkpointing
    save_every_epochs: int = 1
    save_optimizer: bool = True
    save_scheduler: bool = True

    # gradient checkpointing
    gradient_checkpointing: bool = False

    # FSDP
    fsdp: bool = False  # enable PyTorch FSDP wrapping

    # dataloader performance
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    # performance tweaks (esp. for imaging models)
    channels_last: bool = False
    cudnn_benchmark: bool = True


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

        # AMP: use GradScaler only for fp16 on CUDA; not for bf16
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        if (
            self.config.use_amp
            and isinstance(self.device, torch.device)
            and self.device.type == "cuda"
            and self.config.amp_dtype == torch.float16
        ):
            self.scaler = torch.cuda.amp.GradScaler()

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self._wrapped_with_ddp = False
        self._ds_engine = None

        # cuDNN perf
        try:
            torch.backends.cudnn.benchmark = bool(self.config.cudnn_benchmark)
        except Exception:  # pragma: no cover
            pass

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
        # gradient checkpointing best-effort enable
        if self.config.gradient_checkpointing:
            self._maybe_enable_gradient_checkpointing(self.model)

        # channels last for conv-heavy models
        if (
            self.config.channels_last
            and isinstance(self.device, torch.device)
            and self.device.type == "cuda"
        ):
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
            except Exception:  # pragma: no cover
                warnings.warn("channels_last requested but not supported; continuing.")

        self.model.to(self.device)
        self.model.train()

        # Distributed wrapping priority: DeepSpeed > FSDP > DDP
        if self._should_use_deepspeed():
            # Optimizer must exist for deepspeed.initialize
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=self.config.betas,
                eps=self.config.eps,
            )
            self._init_deepspeed()
        else:
            # Initialize process group if DDP or FSDP is requested
            if self._should_use_ddp() or self._should_use_fsdp():
                self._init_distributed()
            if self._should_use_fsdp():
                # Wrap model with FSDP
                self._wrap_model_fsdp()
                # Build optimizer on sharded parameters
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                    betas=self.config.betas,
                    eps=self.config.eps,
                )
            else:
                # Possibly wrap with DDP
                if self._should_use_ddp():
                    self._wrap_model_ddp()
                # Build optimizer on (wrapped or bare) model
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

        try:
            train_loader, train_sampler = self._build_dataloader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=not self._is_distributed(),
            )
            eval_loader: Optional[DataLoader] = None
            eval_sampler = None
            if eval_dataset is not None:
                eval_loader, eval_sampler = self._build_dataloader(
                    eval_dataset,
                    batch_size=self.config.eval_batch_size,
                    shuffle=False,
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

                    if self._ds_engine is not None:
                        # DeepSpeed handles scaling/backward internally
                        self.model.backward(loss)
                    elif self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    if step % accum_steps == 0:
                        if self._ds_engine is None and self.config.max_grad_norm is not None:
                            if self.scaler is not None:
                                self.scaler.unscale_(self.optimizer)  # type: ignore[arg-type]
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.config.max_grad_norm
                            )

                        if self._ds_engine is not None:
                            # DeepSpeed engine performs step and zero grad
                            self.model.step()
                        elif self.scaler is not None:
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

                        if self._ds_engine is None:
                            self.optimizer.zero_grad(set_to_none=True)  # type: ignore[union-attr]
                        global_step += 1

                    running_loss += loss.item() * accum_steps
                    if self._is_rank0() and global_step % max(1, self.config.log_every) == 0:
                        self._log(
                            {
                                "epoch": epoch,
                                "step": global_step,
                                "loss": running_loss / max(1, step),
                                "lr": self.optimizer.param_groups[0]["lr"]
                                if self.optimizer
                                else None,
                            }
                        )

                # end epoch: evaluate + save checkpoint
                if eval_loader is not None:
                    if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                        train_sampler.set_epoch(epoch + 1)
                    if eval_sampler is not None and hasattr(eval_sampler, "set_epoch"):
                        eval_sampler.set_epoch(epoch + 1)
                    eval_res = self.evaluate(eval_loader)
                    if self._is_rank0():
                        self._log({"epoch": epoch, "eval_loss": eval_res.loss, **eval_res.metrics})

                if self._is_rank0() and (epoch + 1) % max(1, self.config.save_every_epochs) == 0:
                    self._save_checkpoint(output_dir, f"checkpoint-epoch{epoch+1}")

            # final save
            if self._is_rank0():
                self._save_model(output_dir)
        finally:
            # Ensure distributed resources are properly released
            self._shutdown_distributed()

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

        # all-reduce average across processes if DDP
        if self._is_distributed_initialized():
            t = torch.tensor([avg_loss], device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            avg_loss = t.item()
        return EvalResult(loss=avg_loss, metrics={})

    @torch.no_grad()
    def inference(self, inputs: Dict[str, Any]) -> Any:
        self.model.eval()
        inputs = self._move_to_device(inputs)
        return self.model(**inputs)

    def _forward_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        if self.config.use_amp:
            # CUDA autocast supports fp16/bf16
            if isinstance(self.device, torch.device) and self.device.type == "cuda":
                if self.config.amp_dtype == torch.bfloat16:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out = self.model(**batch)
                else:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
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
        # Handle FSDP consolidated state dict on rank 0
        if FSDP is not None and isinstance(self.model, FSDP):  # type: ignore[arg-type]
            # Only rank 0 will write the full state dict
            state: Dict[str, Any]
            if StateDictType is not None and FullStateDictConfig is not None:
                with FSDP.state_dict_type(
                    self.model,
                    StateDictType.FULL_STATE_DICT,  # type: ignore[attr-defined]
                    FullStateDictConfig(offload_to_cpu=True, rank0_only=True),  # type: ignore[attr-defined]
                ):
                    state = self.model.state_dict()
            else:
                # Fallback if state dict API not available
                state = self.model.state_dict()
        else:
            state = self.model.state_dict()

        payload: Dict[str, Any] = {"model_state": state, "config": asdict(self.config)}
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
        if FSDP is not None and isinstance(self.model, FSDP):  # type: ignore[arg-type]
            if StateDictType is not None and FullStateDictConfig is not None:
                with FSDP.state_dict_type(
                    self.model,
                    StateDictType.FULL_STATE_DICT,  # type: ignore[attr-defined]
                    FullStateDictConfig(offload_to_cpu=True, rank0_only=True),  # type: ignore[attr-defined]
                ):
                    state = self.model.state_dict()
            else:
                state = self.model.state_dict()
            torch.save(state, os.path.join(output_dir, "pytorch_model.bin"))
        else:
            torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    def _log(self, data: Dict[str, Any]) -> None:
        # minimal logging; can be replaced by structured logging later
        msg = " ".join(f"{k}={v}" for k, v in data.items())
        print(f"[trainer] {msg}")

    # --------- helpers: distributed / dataloaders / checkpointing ---------
    def _is_distributed(self) -> bool:
        if self.config.use_ddp:
            return True
        # also allow env-driven use (e.g., torchrun sets WORLD_SIZE)
        return int(os.environ.get("WORLD_SIZE", "1")) > 1

    def _local_rank(self) -> int:
        return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))

    def _is_rank0(self) -> bool:
        if not self._is_distributed_initialized():
            # single process or not yet initialized
            return self._local_rank() == 0
        try:
            return dist.get_rank() == 0
        except Exception:  # pragma: no cover
            return True

    def _is_distributed_initialized(self) -> bool:
        return dist.is_available() and dist.is_initialized()

    def _should_use_ddp(self) -> bool:
        return self._is_distributed()

    def _init_distributed(self) -> None:
        if not dist.is_available():
            warnings.warn("torch.distributed not available; falling back to single process.")
            return
        if not dist.is_initialized():
            backend = self.config.ddp_backend
            init_method = "env://"
            try:
                dist.init_process_group(backend=backend, init_method=init_method)
            except Exception as e:  # pragma: no cover
                warnings.warn(
                    f"Failed to initialize process group: {e}; continuing single process."
                )

        # set device for local rank if CUDA
        if isinstance(self.device, torch.device) and self.device.type == "cuda":
            torch.cuda.set_device(self._local_rank() % max(1, torch.cuda.device_count()))

    def _wrap_model_ddp(self) -> None:
        if self._wrapped_with_ddp:
            return
        if not dist.is_available() or not dist.is_initialized():
            return
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self._local_rank()] if self.device.type == "cuda" else None,
            find_unused_parameters=self.config.ddp_find_unused_parameters,
        )
        self._wrapped_with_ddp = True

    def _shutdown_distributed(self) -> None:
        """Cleanly destroy process group if initialized to avoid NCCL leaks."""
        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            # Best-effort cleanup; ignore errors during interpreter shutdown
            pass

    def _should_use_deepspeed(self) -> bool:
        return bool(self.config.deepspeed)

    def _init_deepspeed(self) -> None:
        try:
            import deepspeed  # type: ignore
        except Exception as e:  # pragma: no cover - optional dep
            warnings.warn(
                f"DeepSpeed requested but not available: {e}. Falling back to DDP/single."
            )
            if self._should_use_ddp():
                self._init_distributed()
                self._wrap_model_ddp()
            return

        # minimal DeepSpeed init; expects optimizer provided
        ds_cfg = self.config.deepspeed_config
        # Allow path (json/yaml) or dict
        init_kwargs: Dict[str, Any] = {
            "model": self.model,
            "optimizer": self.optimizer,
            "model_parameters": self.model.parameters(),
        }
        if isinstance(ds_cfg, dict):
            init_kwargs["config_params"] = ds_cfg
        elif isinstance(ds_cfg, str):
            init_kwargs["config"] = ds_cfg

        self.model, self.optimizer, _, _ = deepspeed.initialize(**init_kwargs)
        self._ds_engine = self.model

    def _build_dataloader(
        self,
        dataset_or_loader: Iterable[Dict[str, Any]] | torch.utils.data.Dataset,
        batch_size: int,
        shuffle: bool,
    ) -> Tuple[DataLoader, Optional[DistributedSampler]]:
        if isinstance(dataset_or_loader, DataLoader):
            return dataset_or_loader, getattr(dataset_or_loader, "sampler", None)

        sampler = None
        if self._is_distributed():
            sampler = DistributedSampler(dataset_or_loader, shuffle=shuffle)  # type: ignore[arg-type]
            shuffle = False  # sampler handles shuffling

        loader = DataLoader(
            dataset_or_loader,  # type: ignore[arg-type]
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers
            if self.config.num_workers > 0
            else False,
            prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else None,
        )
        return loader, sampler

    def _maybe_enable_gradient_checkpointing(self, module: torch.nn.Module) -> None:
        # Try HF-style API first
        if hasattr(module, "gradient_checkpointing_enable"):
            try:
                module.gradient_checkpointing_enable()  # type: ignore[attr-defined]
                return
            except Exception:  # pragma: no cover
                pass
        # Fallback: set attribute for downstream modules to read
        try:
            setattr(module, "gradient_checkpointing", True)
        except Exception:  # pragma: no cover
            pass

    # -------------------- FSDP helpers --------------------
    def _should_use_fsdp(self) -> bool:
        return bool(self.config.fsdp and FSDP is not None)

    def _wrap_model_fsdp(self) -> None:
        if FSDP is None:
            warnings.warn("FSDP requested but not available in this PyTorch build.")
            return
        # Basic full-module wrap. Advanced auto-wrap policies can be added later.
        self.model = FSDP(self.model)  # type: ignore[no-redef]
