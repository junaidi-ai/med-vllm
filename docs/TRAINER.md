# MedicalModelTrainer

This page documents the lightweight fine-tuning trainer in `medvllm/training/trainer.py`.

- Class: `MedicalModelTrainer`
- Config: `TrainerConfig`

## Highlights
- Optimizers: AdamW, Adam, SGD.
- Schedulers: linear warmup+decay, step, cosine, one-cycle.
- Gradient clipping: norm or value.
- AMP: autocast (CUDA/XLA) with GradScaler for fp16 on CUDA.
- Distributed: DDP, optional FSDP and DeepSpeed best-effort.
- Checkpointing: save/load model, optimizer, scheduler, scaler, and trainer state.
- Export: TorchScript and ONNX with `export_input_example`.
- Logging: TensorBoard and Weights & Biases (lazy init, rank-0 only).
- QoL: gradient accumulation, gradient checkpointing, channels_last, dropout override.

## Quick Start
```python
from medvllm.training.trainer import MedicalModelTrainer, TrainerConfig

model = ...  # a torch.nn.Module that returns {"loss": loss} or a loss tensor
train_ds = ...  # PyTorch Dataset or iterable of dict batches

config = TrainerConfig(
    num_epochs=1,
    batch_size=8,
    learning_rate=5e-5,
    optimizer="adamw",
    scheduler="linear_warmup",
    warmup_steps=10,
    use_amp=False,  # set True on CUDA for mixed precision
    save_every_epochs=1,
    enable_tensorboard=False,
)

trainer = MedicalModelTrainer(model, config)
trainer.train(train_ds, eval_dataset=None, output_dir="./fine_tuned_model")
```

## Common Options
- `optimizer`: one of `adamw`, `adam`, `sgd`.
- `scheduler`: `linear_warmup`, `step`, `cosine`, `one_cycle`, or `none`.
- `gradient_accumulation_steps`: accumulate before stepping.
- `grad_clip_mode`: `norm` | `value` | `none`; `grad_clip_value` when `value`.
- `use_amp`: enable autocast; `amp_dtype`: `torch.float16` or `torch.bfloat16`.
- `device`: `auto` by default (prefers CUDA when available), or `cpu`/`cuda`/`xla`.
- `save_every_epochs`: checkpoint cadence. `save_optimizer`, `save_scheduler` control payload.
- `resume_from`: path to a `*.pt` checkpoint previously saved by the trainer.
- `export_torchscript` / `export_onnx`: export at the end of training. Provide `export_input_example`.
- `enable_tensorboard`: write scalars to a `tb_logs/` subdir. `enable_wandb` for W&B.
- `gradient_checkpointing`: attempts to enable via HF-style API or attribute toggle.
- `channels_last`: set model memory format to `torch.channels_last` on CUDA.

## Checkpointing
- Saves to `{output_dir}/checkpoint-epoch{N}.pt` or `-step{S}.pt` depending on call site.
- Payload includes: `model_state`, `optimizer_state` (optional), `scheduler_state` (optional), `scaler_state` (AMP), and `trainer_state` with `epoch`/`global_step`.
- Use `TrainerConfig.resume_from` to resume:
```python
config = TrainerConfig(resume_from="./fine_tuned_model/checkpoint-epoch1.pt")
trainer = MedicalModelTrainer(model, config)
trainer.train(train_ds, eval_dataset)
```

## Export
Provide an input example dict matching your model signature:
```python
config = TrainerConfig(
    export_torchscript=True,
    export_onnx=False,
    export_input_example={"input_ids": torch.zeros(1, 8, dtype=torch.long)},
)
```
Exports are written under `output_dir` as `model.pt` (TorchScript) and/or `model.onnx`.

## Logging
- `enable_tensorboard=True` writes scalars under `{output_dir}/tb_logs/`.
- `enable_wandb=True` initializes a W&B run on rank-0 with `wandb_project`/`wandb_run_name`.

## Distributed and FSDP/DeepSpeed
- DDP: enable by launching with `torchrun` (WORLD_SIZE>1). Trainer auto-wraps the model.
- FSDP: set `fsdp=True` if PyTorch build supports it. Saving handles full state dict on rank 0.
- DeepSpeed: set `deepspeed=True` with `deepspeed_config` dict or path. Falls back gracefully if unavailable.

## Tips
- Use tiny synthetic datasets for quick smoke tests.
- AMP with fp16 requires CUDA; bf16 is supported on CUDA and XLA.
- `channels_last` benefits conv-heavy models on CUDA.

## Testing Notes
The CI/test environment can be constrained (e.g., partial PyTorch shims without CUDA, optimizers, or autograd profiler). Tests for `MedicalModelTrainer` are hardened to avoid flakiness:

- __Profiler guard__: In some builds, `torch.autograd.profiler.record_function` can crash. Tests may monkeypatch it to a no-op context manager.
- __DataLoader replacement__: To avoid multiprocessing/pin memory and profiler interactions, some tests replace `medvllm.training.trainer.DataLoader` with a minimal in-test loader that batches dicts of tensors.
- __Device movement bypass__: Tests can monkeypatch `trainer._move_to_device` to identity to avoid missing APIs like `torch.is_tensor` in minimal shims.
- __Optimizer/scheduler monkeypatch__: Tests monkeypatch `_build_param_groups`, `_optimizer_factory`, and `_scheduler_factory` to simple/dummy implementations to avoid reliance on `torch.optim` and scheduler APIs.
- __Gradient checkpointing__: Tests call `_maybe_enable_gradient_checkpointing()` directly and assert the best-effort flag on submodules. If not available, tests skip rather than fail.
- __Export tests__: TorchScript/ONNX exports are guarded on module availability and focus on "no-crash" semantics rather than artifact existence under constrained environments.

When running locally with a full PyTorch installation, you can remove or ignore these guards to exercise the complete training loop. Consider running the suite with CUDA available to validate AMP, `channels_last`, and export paths end-to-end.
