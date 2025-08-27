from __future__ import annotations

from dataclasses import dataclass, asdict
import contextlib
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

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

    # evaluation scheduling
    eval_every_steps: Optional[int] = None  # if set, run evaluation every N optimizer steps
    eval_at_epoch_end: bool = True  # keep existing behavior

    # evaluation metrics
    eval_metric: str = "eval_loss"  # e.g., "f1", "accuracy", "dice", "iou", or "eval_loss"
    eval_metric_mode: str = "min"  # "min" for loss, "max" for metrics like f1/accuracy/dice/iou
    eval_compute_metrics: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, float]]] = (
        None
    )
    # If provided, called as eval_compute_metrics(batch, model_output) per batch and aggregated by mean

    # early stopping
    early_stopping_patience: Optional[int] = None  # stop after N eval rounds without improvement
    early_stopping_min_delta: float = 0.0

    # visualization
    visualize_every_steps: Optional[int] = None
    visualize_fn: Optional[Callable[[Dict[str, Any], Dict[str, Any], int, str], None]] = None
    # Called as visualize_fn(batch, model_output, global_step, output_dir)

    # logging backends
    enable_tensorboard: bool = False
    tb_log_dir: Optional[str] = None
    enable_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

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
    save_every_steps: Optional[int] = None
    resume_from: Optional[str] = None  # path to checkpoint .pt

    # export
    export_torchscript: bool = False
    export_onnx: bool = False
    export_input_example: Optional[Dict[str, Any]] = None  # required for export
    onnx_opset: int = 13

    # model versioning
    model_versioning: bool = True

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

        # Optional loggers (initialized lazily in train())
        self._tb_writer = None
        self._wandb = None

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

    # ---------------- Distributed / DeepSpeed / FSDP helpers (moved onto trainer) ----------------
    def _is_distributed(self) -> bool:
        if self.config.use_ddp:
            return True
        return int(os.environ.get("WORLD_SIZE", "1")) > 1

    def _local_rank(self) -> int:
        return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))

    def _is_rank0(self) -> bool:
        if not self._is_distributed_initialized():
            return self._local_rank() == 0
        try:
            return dist.get_rank() == 0
        except Exception:  # pragma: no cover
            return True

    def _is_distributed_initialized(self) -> bool:
        return dist.is_available() and dist.is_initialized()

    def _should_use_ddp(self) -> bool:
        return self._is_distributed()

    def _should_use_fsdp(self) -> bool:
        return bool(self.config.fsdp and FSDP is not None)

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

    def _wrap_model_fsdp(self) -> None:
        if FSDP is None:
            warnings.warn("FSDP requested but not available in this PyTorch build.")
            return
        self.model = FSDP(self.model)  # type: ignore[no-redef]

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

        ds_cfg = self.config.deepspeed_config
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

    def _shutdown_distributed(self) -> None:
        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass

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
            shuffle = False

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

    def _next_version(self, output_dir: str) -> int:
        import json

        os.makedirs(output_dir, exist_ok=True)
        manifest_path = os.path.join(output_dir, "manifest.json")
        data: Dict[str, Any] = {"latest_version": 0}
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {"latest_version": 0}
        v = int(data.get("latest_version", 0)) + 1
        data["latest_version"] = v
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass
        return v

    def _export_artifacts(self, output_dir: str) -> None:
        if not (self.config.export_torchscript or self.config.export_onnx):
            return
        sample = self.config.export_input_example
        if not isinstance(sample, dict):
            warnings.warn(
                "export_* requested but export_input_example is missing; skipping export."
            )
            return
        try:
            self.model.eval()
            with torch.no_grad():
                example_kwargs = self._move_to_device(sample)

                if self.config.export_torchscript:
                    try:
                        scripted = torch.jit.trace(self.model, example_kwargs)
                        ts_path = os.path.join(output_dir, "model.ts")
                        scripted.save(ts_path)
                    except Exception as e:
                        warnings.warn(f"TorchScript export failed: {e}")

                if self.config.export_onnx:
                    try:
                        import torch.onnx  # noqa: F401

                        onnx_path = os.path.join(output_dir, "model.onnx")
                        torch.onnx.export(
                            self.model,
                            f=onnx_path,
                            kwargs=example_kwargs,
                            input_names=list(example_kwargs.keys()),
                            output_names=["output"],
                            opset_version=int(self.config.onnx_opset),
                            dynamic_axes={k: {0: "batch"} for k in example_kwargs.keys()},
                        )
                    except Exception as e:
                        warnings.warn(f"ONNX export failed: {e}")
        except Exception as e:  # pragma: no cover
            warnings.warn(f"Export failed: {e}")

    def _maybe_enable_gradient_checkpointing(self, module: torch.nn.Module) -> None:
        # Try HF-style API first
        try:
            if hasattr(module, "gradient_checkpointing_enable"):
                module.gradient_checkpointing_enable()  # type: ignore[attr-defined]
                return
        except Exception:
            pass
        # Try setting attribute on submodules commonly used
        try:
            for m in module.modules():
                if hasattr(m, "gradient_checkpointing"):
                    setattr(m, "gradient_checkpointing", True)
        except Exception:
            pass

    def train(
        self,
        train_dataset: Iterable[Dict[str, Any]] | torch.utils.data.Dataset,
        eval_dataset: Optional[Iterable[Dict[str, Any]] | torch.utils.data.Dataset] = None,
        output_dir: str = "./fine_tuned_model",
    ) -> None:
        self.prepare_for_training()

        try:
            # Resume from checkpoint if provided
            start_epoch = 0
            global_step = 0
            if self.config.resume_from:
                if self._is_rank0():
                    self._log({"resume_from": self.config.resume_from})
                resume_state = self._load_checkpoint(self.config.resume_from)
                start_epoch = int(resume_state.get("epoch", 0))
                global_step = int(resume_state.get("global_step", 0))

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

            accum_steps = max(1, self.config.gradient_accumulation_steps)

            # Setup logging backends
            if self._is_rank0():
                if self.config.enable_tensorboard:
                    try:
                        from torch.utils.tensorboard import SummaryWriter  # type: ignore

                        tb_dir = self.config.tb_log_dir or os.path.join(output_dir, "tb_logs")
                        os.makedirs(tb_dir, exist_ok=True)
                        self._tb_writer = SummaryWriter(tb_dir)
                    except Exception as e:  # pragma: no cover
                        warnings.warn(f"TensorBoard not available: {e}")
                        self._tb_writer = None
                if self.config.enable_wandb:
                    try:
                        import wandb  # type: ignore

                        self._wandb = wandb
                        wandb.init(
                            project=self.config.wandb_project or "medvllm",
                            name=self.config.wandb_run_name,
                            config=asdict(self.config),
                            reinit=True,
                        )
                    except Exception as e:  # pragma: no cover
                        warnings.warn(f"Weights & Biases not available: {e}")
                        self._wandb = None

            # Default visualization hook if interval is configured but no function provided
            if (
                self.config.visualize_every_steps is not None
                and self.config.visualize_every_steps > 0
                and self.config.visualize_fn is None
            ):
                self.config.visualize_fn = self._default_visualize

            # Early stopping state
            best_metric: Optional[float] = None
            bad_rounds = 0

            try:
                for epoch in range(start_epoch, self.config.num_epochs):
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

                            # Optional step-based checkpointing
                            if (
                                self._is_rank0()
                                and self.config.save_every_steps is not None
                                and self.config.save_every_steps > 0
                                and (global_step % self.config.save_every_steps == 0)
                            ):
                                self._save_checkpoint(
                                    output_dir,
                                    f"checkpoint-step{global_step}",
                                    epoch=epoch,
                                    global_step=global_step,
                                )

                        running_loss += loss.item() * accum_steps
                        if self._is_rank0() and global_step % max(1, self.config.log_every) == 0:
                            log_payload = {
                                "epoch": epoch,
                                "step": global_step,
                                "loss": running_loss / max(1, step),
                                "lr": self.optimizer.param_groups[0]["lr"]
                                if self.optimizer
                                else None,
                            }
                            self._log(log_payload)
                            if self._tb_writer is not None:
                                try:
                                    self._tb_writer.add_scalar(
                                        "train/loss", log_payload["loss"], global_step
                                    )
                                    if log_payload["lr"] is not None:
                                        self._tb_writer.add_scalar(
                                            "train/lr", float(log_payload["lr"]), global_step
                                        )
                                except Exception:
                                    pass
                            if self._wandb is not None:
                                try:
                                    self._wandb.log(
                                        {
                                            "train/loss": log_payload["loss"],
                                            "train/lr": log_payload["lr"],
                                        },
                                        step=global_step,
                                    )
                                except Exception:
                                    pass

                        # Step-based evaluation / visualization
                        if (
                            eval_loader is not None
                            and self.config.eval_every_steps is not None
                            and self.config.eval_every_steps > 0
                            and global_step % self.config.eval_every_steps == 0
                        ):
                            eval_res = self.evaluate(eval_loader)
                            if self._is_rank0():
                                metrics_log = {
                                    "epoch": epoch,
                                    "step": global_step,
                                    "eval_loss": eval_res.loss,
                                    **eval_res.metrics,
                                }
                                self._log(metrics_log)
                                if self._tb_writer is not None:
                                    try:
                                        self._tb_writer.add_scalar(
                                            "eval/loss", eval_res.loss, global_step
                                        )
                                        for k, v in eval_res.metrics.items():
                                            self._tb_writer.add_scalar(f"eval/{k}", v, global_step)
                                    except Exception:
                                        pass
                                if self._wandb is not None:
                                    try:
                                        log_data = {"eval/loss": eval_res.loss}
                                        log_data.update(
                                            {f"eval/{k}": v for k, v in eval_res.metrics.items()}
                                        )
                                        self._wandb.log(log_data, step=global_step)
                                    except Exception:
                                        pass

                            # Visualization hook
                            if (
                                self.config.visualize_every_steps is not None
                                and self.config.visualize_every_steps > 0
                                and global_step % self.config.visualize_every_steps == 0
                                and self.config.visualize_fn is not None
                            ):
                                try:
                                    # Use the last processed batch for quick visualization
                                    with torch.no_grad():
                                        out = self.model(**batch)
                                    self.config.visualize_fn(
                                        batch,
                                        out if isinstance(out, dict) else {"output": out},
                                        global_step,
                                        output_dir,
                                    )
                                except Exception:  # pragma: no cover - best effort
                                    pass

                            # Early stopping check on step-wise eval
                            if self._is_rank0() and self.config.early_stopping_patience:
                                current = self._select_monitored(eval_res)
                                is_better = self._is_better(current, best_metric)
                                if is_better:
                                    best_metric = current
                                    bad_rounds = 0
                                else:
                                    bad_rounds += 1
                                    if bad_rounds >= int(self.config.early_stopping_patience):
                                        if self._is_rank0():
                                            self._log(
                                                {
                                                    "early_stop": True,
                                                    "best": best_metric,
                                                    "at_step": global_step,
                                                }
                                            )
                                        raise _EarlyStopSignal()

                    # end epoch: evaluate + save checkpoint
                    if eval_loader is not None and self.config.eval_at_epoch_end:
                        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                            train_sampler.set_epoch(epoch + 1)
                        if eval_sampler is not None and hasattr(eval_sampler, "set_epoch"):
                            eval_sampler.set_epoch(epoch + 1)
                        eval_res = self.evaluate(eval_loader)
                        if self._is_rank0():
                            self._log(
                                {"epoch": epoch, "eval_loss": eval_res.loss, **eval_res.metrics}
                            )
                            if self._tb_writer is not None:
                                try:
                                    self._tb_writer.add_scalar(
                                        "eval/loss", eval_res.loss, global_step
                                    )
                                    for k, v in eval_res.metrics.items():
                                        self._tb_writer.add_scalar(f"eval/{k}", v, global_step)
                                except Exception:
                                    pass
                            if self._wandb is not None:
                                try:
                                    log_data = {"eval/loss": eval_res.loss}
                                    log_data.update(
                                        {f"eval/{k}": v for k, v in eval_res.metrics.items()}
                                    )
                                    self._wandb.log(log_data, step=global_step)
                                except Exception:
                                    pass

                            # Early stopping check on epoch-end eval
                            if self.config.early_stopping_patience:
                                current = self._select_monitored(eval_res)
                                is_better = self._is_better(current, best_metric)
                                if is_better:
                                    best_metric = current
                                    bad_rounds = 0
                                else:
                                    bad_rounds += 1
                                    if bad_rounds >= int(self.config.early_stopping_patience):
                                        self._log(
                                            {
                                                "early_stop": True,
                                                "best": best_metric,
                                                "at_epoch": epoch,
                                            }
                                        )
                                        raise _EarlyStopSignal()

                if self._is_rank0() and (epoch + 1) % max(1, self.config.save_every_epochs) == 0:
                    self._save_checkpoint(
                        output_dir,
                        f"checkpoint-epoch{epoch+1}",
                        epoch=epoch + 1,
                        global_step=global_step,
                    )

            except _EarlyStopSignal:
                pass

            # final save
            if self._is_rank0():
                self._save_model(output_dir)
                # optional export artifacts
                self._export_artifacts(output_dir)
        finally:
            # Ensure distributed resources are properly released
            self._shutdown_distributed()
            # Close loggers
            try:
                if self._tb_writer is not None:
                    self._tb_writer.close()
            except Exception:
                pass
            try:
                if self._wandb is not None:
                    self._wandb.finish()
            except Exception:
                pass

    def _no_grad(self):
        # Return a context manager for no_grad that is safe under torch mocks
        ng = getattr(torch, "no_grad", None)
        if callable(ng):
            return ng()
        if isinstance(ng, contextlib.AbstractContextManager):  # type: ignore[attr-defined]
            return ng
        return contextlib.nullcontext()

    def evaluate(self, eval_loader: DataLoader) -> EvalResult:
        self.model.eval()
        total_loss = 0.0
        count = 0
        # Metric accumulators
        agg: Dict[str, float] = {}

        with self._no_grad():
            for batch in eval_loader:
                batch = self._move_to_device(batch)
                out = self.model(**batch)
                if isinstance(out, dict) and "loss" in out:
                    loss_t = out["loss"]
                else:
                    # If model returned only loss tensor
                    loss_t = out if torch.is_tensor(out) else torch.as_tensor(0.0)
                total_loss += float(getattr(loss_t, "item", lambda: float(loss_t))())
                count += 1

                # Compute metrics per batch if possible
                metrics_this: Dict[str, float] = {}
                try:
                    if self.config.eval_compute_metrics is not None and isinstance(out, dict):
                        metrics_this = self.config.eval_compute_metrics(batch, out)
                    else:
                        # Heuristics: classification and segmentation
                        labels = None
                        for k in ("labels", "label", "targets", "target", "y"):
                            if k in batch:
                                labels = batch[k]
                                break
                        if labels is not None:
                            if isinstance(out, dict) and ("logits" in out or "preds" in out):
                                preds_t = out.get("preds", None)
                                logits_t = out.get("logits", None)
                                if preds_t is None and logits_t is not None:
                                    # classification-style argmax
                                    if logits_t.dim() >= 2:
                                        preds_t = logits_t.argmax(dim=1)
                                if preds_t is not None:
                                    # If shapes look like images (segmentation)
                                    if preds_t.dim() >= 3 or (labels.dim() >= 3):
                                        dice, iou = self._segmentation_metrics(preds_t, labels)
                                        if dice is not None:
                                            metrics_this["dice"] = dice
                                        if iou is not None:
                                            metrics_this["iou"] = iou
                                        hd = self._try_hausdorff(preds_t, labels)
                                        if hd is not None:
                                            metrics_this["hausdorff"] = hd
                                    else:
                                        # classification metrics using helper if available
                                        try:
                                            from medvllm.utils.metrics import (
                                                compute_classification_metrics,
                                            )

                                            m = compute_classification_metrics(
                                                y_true=labels.detach().view(-1).cpu().tolist(),
                                                y_pred=preds_t.detach().view(-1).cpu().tolist(),
                                                average="macro",
                                            )
                                            metrics_this.update(m)
                                        except Exception:
                                            pass
                except Exception:
                    # Metrics are best-effort; ignore per-batch errors
                    pass

                for k, v in metrics_this.items():
                    agg[k] = agg.get(k, 0.0) + float(v)

        avg_loss = total_loss / max(1, count)
        metrics_avg = {k: v / max(1, count) for k, v in agg.items()}

        # all-reduce average across processes if DDP
        if self._is_distributed_initialized():
            t = torch.tensor([avg_loss], device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            avg_loss = t.item()
            # reduce metrics as well
            for k in list(metrics_avg.keys()):
                tt = torch.tensor([metrics_avg[k]], device=self.device)
                dist.all_reduce(tt, op=dist.ReduceOp.AVG)
                metrics_avg[k] = tt.item()
        return EvalResult(loss=avg_loss, metrics=metrics_avg)

    def _default_visualize(
        self, batch: Dict[str, Any], output: Dict[str, Any], global_step: int, output_dir: str
    ) -> None:
        """Best-effort default visualization without extra heavy deps.

        Saves a lightweight summary per step under ``{output_dir}/visualizations/`` with:
        - shapes and dtypes of common tensors
        - optional single-channel PNG if PIL is available and a 2D/3D mask is detected
        """
        try:
            import os

            os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
            path = os.path.join(output_dir, "visualizations", f"step{global_step}.pt")

            def _summarize(x: Any) -> Any:
                import torch

                if torch.is_tensor(x):
                    return {
                        "shape": tuple(x.shape),
                        "dtype": str(x.dtype),
                        "min": float(x.min().item()),
                        "max": float(x.max().item()),
                    }
                if isinstance(x, (list, tuple)):
                    return [_summarize(v) for v in x[:4]]
                if isinstance(x, dict):
                    return {k: _summarize(v) for k, v in list(x.items())[:8]}
                return str(type(x))

            summary = {
                "inputs": _summarize(batch),
                "outputs": _summarize(output),
            }

            # Always save a compact summary tensor file
            import torch as _t

            _t.save(summary, path)

            # Optionally save a quick mask preview if possible
            try:
                from PIL import Image  # type: ignore
                import numpy as np  # type: ignore

                # Heuristic: find a mask-like tensor
                mask = None
                for key in ("preds", "logits"):
                    if key in output and _t.is_tensor(output[key]):
                        val = output[key]
                        if val.dim() >= 3:
                            if key == "logits" and val.size(1) > 1:
                                val = val.argmax(dim=1)
                            else:
                                val = val.squeeze(1)
                            mask = val[0].detach().cpu()
                            break
                if mask is not None:
                    arr = (mask.float() - mask.min()) / (max(1e-8, (mask.max() - mask.min())))
                    arr = (arr * 255.0).clamp(0, 255).byte().numpy()
                    if arr.ndim == 3:
                        # take a middle slice for 3D volumes
                        arr = arr[arr.shape[0] // 2]
                    img = Image.fromarray(arr)
                    img.save(
                        os.path.join(output_dir, "visualizations", f"mask_step{global_step}.png")
                    )
            except Exception:
                pass
        except Exception:
            # Never break training due to visualization errors
            pass

    def inference(self, inputs: Dict[str, Any]) -> Any:
        self.model.eval()
        inputs = self._move_to_device(inputs)
        with self._no_grad():
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

    def _save_checkpoint(
        self,
        output_dir: str,
        name: str,
        *,
        epoch: Optional[int] = None,
        global_step: Optional[int] = None,
    ) -> None:
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
        payload["trainer_state"] = {
            "epoch": epoch,
            "global_step": global_step,
        }

        # simple versioning: include incrementing version in manifest and payload
        if self.config.model_versioning:
            version = self._next_version(output_dir)
            payload["version"] = version
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
            # versioned save
            if self.config.model_versioning:
                v = self._next_version(output_dir)
                ver_path = os.path.join(output_dir, f"pytorch_model.v{v}.bin")
                torch.save(state, ver_path)
                # also keep/update latest
                torch.save(state, os.path.join(output_dir, "pytorch_model.bin"))
            else:
                torch.save(state, os.path.join(output_dir, "pytorch_model.bin"))
        else:
            state = self.model.state_dict()
            if self.config.model_versioning:
                v = self._next_version(output_dir)
                ver_path = os.path.join(output_dir, f"pytorch_model.v{v}.bin")
                torch.save(state, ver_path)
                torch.save(state, os.path.join(output_dir, "pytorch_model.bin"))
            else:
                torch.save(state, os.path.join(output_dir, "pytorch_model.bin"))

    def _log(self, data: Dict[str, Any]) -> None:
        # minimal logging; can be replaced by structured logging later
        msg = " ".join(f"{k}={v}" for k, v in data.items())
        print(f"[trainer] {msg}")

    def _select_monitored(self, eval_res: EvalResult | float) -> float:
        if isinstance(eval_res, (int, float)):
            return float(eval_res)
        if self.config.eval_metric == "eval_loss" or not eval_res.metrics:
            return float(eval_res.loss)
        val = eval_res.metrics.get(self.config.eval_metric)
        if val is None:
            # fallback to loss
            return float(eval_res.loss)
        return float(val)

    def _is_better(self, current: float, best: Optional[float]) -> bool:
        if best is None:
            return True
        mode = (self.config.eval_metric_mode or "min").lower()
        delta = float(self.config.early_stopping_min_delta)
        if mode == "max":
            return current > (best + delta)
        return current < (best - delta)

    def _segmentation_metrics(
        self, preds: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[Optional[float], Optional[float]]:
        """Compute Dice and IoU for binary or multi-class masks.

        Heuristic binarization/argmax where appropriate. Returns (dice, iou) averaged over batch.
        """
        try:
            # Align shapes: if logits [B,C,...], take argmax; if [B,...] already class ids
            if preds.dim() > labels.dim():
                # likely logits with channel dimension
                if preds.size(1) > 1:
                    preds = preds.argmax(dim=1)
                else:
                    preds = (preds.squeeze(1) > 0).to(labels.dtype)
            if labels.dim() > preds.dim():
                labels = labels.squeeze(1)

            preds_flat = preds.reshape(preds.size(0), -1)
            labels_flat = labels.reshape(labels.size(0), -1)

            # Binary if only two classes present across batch
            num_classes = int(
                torch.max(torch.stack([preds_flat.max(), labels_flat.max()])).item() + 1
            )
            if num_classes <= 2:
                # Treat as binary foreground=1
                p = (preds_flat > 0).float()
                l = (labels_flat > 0).float()
                intersection = (p * l).sum(dim=1)
                union = p.sum(dim=1) + l.sum(dim=1)
                dice = ((2.0 * intersection + 1e-8) / (union + 1e-8)).mean().item()
                denom = (p + l - p * l).sum(dim=1)
                iou = ((intersection + 1e-8) / (denom + 1e-8)).mean().item()
                return float(dice), float(iou)
            else:
                # Multi-class macro average
                dice_scores = []
                iou_scores = []
                for c in range(num_classes):
                    p = (preds_flat == c).float()
                    l = (labels_flat == c).float()
                    intersection = (p * l).sum(dim=1)
                    union = p.sum(dim=1) + l.sum(dim=1)
                    dice_c = ((2.0 * intersection + 1e-8) / (union + 1e-8)).mean()
                    denom = (p + l - p * l).sum(dim=1)
                    iou_c = ((intersection + 1e-8) / (denom + 1e-8)).mean()
                    dice_scores.append(dice_c)
                    iou_scores.append(iou_c)
                dice = torch.stack(dice_scores).mean().item()
                iou = torch.stack(iou_scores).mean().item()
                return float(dice), float(iou)
        except Exception:
            return None, None

    def _try_hausdorff(self, preds: torch.Tensor, labels: torch.Tensor) -> Optional[float]:
        """Best-effort Hausdorff distance approximation.

        Uses scipy if available; otherwise approximates via point set distances on foreground.
        Returns mean symmetric Hausdorff over batch, or None on failure.
        """
        try:
            # Prepare binary masks
            if preds.dim() > labels.dim():
                if preds.size(1) > 1:
                    preds = preds.argmax(dim=1)
                else:
                    preds = (preds.squeeze(1) > 0).to(labels.dtype)
            if labels.dim() > preds.dim():
                labels = labels.squeeze(1)
            p = (preds > 0).bool()
            l = (labels > 0).bool()

            try:
                from scipy.spatial.distance import directed_hausdorff  # type: ignore

                def hd_np(a: torch.Tensor, b: torch.Tensor) -> float:
                    a_idx = a.nonzero(as_tuple=False).cpu().numpy()
                    b_idx = b.nonzero(as_tuple=False).cpu().numpy()
                    if a_idx.size == 0 or b_idx.size == 0:
                        return float("inf")
                    h1 = directed_hausdorff(a_idx, b_idx)[0]
                    h2 = directed_hausdorff(b_idx, a_idx)[0]
                    return float(max(h1, h2))

                vals = [hd_np(p[i], l[i]) for i in range(p.size(0))]
                # guard large infs
                finite_vals = [v for v in vals if v != float("inf")]
                return float(sum(finite_vals) / max(1, len(finite_vals))) if finite_vals else None
            except Exception:
                # Fallback approximation using torch.cdist on sampled points
                vals: list[float] = []
                for i in range(p.size(0)):
                    pa = p[i].nonzero(as_tuple=False).float()
                    lb = l[i].nonzero(as_tuple=False).float()
                    if pa.numel() == 0 or lb.numel() == 0:
                        continue
                    # sample to limit cost
                    max_pts = 2048
                    if pa.size(0) > max_pts:
                        idx = torch.randperm(pa.size(0))[:max_pts]
                        pa = pa[idx]
                    if lb.size(0) > max_pts:
                        idx = torch.randperm(lb.size(0))[:max_pts]
                        lb = lb[idx]
                    dists = torch.cdist(pa, lb, p=2)
                    # directed hd
                    h1 = dists.min(dim=1).values.max().item()
                    h2 = dists.min(dim=0).values.max().item()
                    vals.append(max(h1, h2))
                return float(sum(vals) / max(1, len(vals))) if vals else None
        except Exception:
            return None


class _EarlyStopSignal(Exception):
    pass
