import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

try:
    from tests.medical.memory_profiler import MemoryProfiler  # type: ignore
except Exception:

    class MemoryProfiler:  # type: ignore
        def __init__(self, device: str = "cpu"):
            self.device = device
            self.results = {}

        def profile(self):
            from contextlib import contextmanager

            @contextmanager
            def _cm():
                yield

            return _cm()


# Simple synthetic dataset for fast CI runs
class TinyTextDataset(Dataset):
    def __init__(
        self, size: int, seq_len: int, vocab_size: int, num_classes: int, device: torch.device
    ):
        self.size = int(size)
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.num_classes = int(num_classes)
        self.device = device

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        y = torch.randint(0, self.num_classes, (1,), dtype=torch.long).squeeze(0)
        return x, y


# Tiny model to keep CI fast
class TinyModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (B, T)
        emb = self.embedding(x)
        out, _ = self.encoder(emb)
        pooled = out.mean(dim=1)
        logits = self.classifier(pooled)
        return logits


# JSONL text dataset for real adapter
class JsonlTextDataset(Dataset):
    def __init__(self, path: Path, num_classes: int, limit: Optional[int] = None):
        self.samples: List[Tuple[str, int]] = []
        cnt = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("text")
                except Exception:
                    text = None
                if isinstance(text, str) and text:
                    self.samples.append((text, cnt % max(1, num_classes)))
                    cnt += 1
                    if limit is not None and cnt >= limit:
                        break
        if not self.samples:
            # fallback single sample
            self.samples = [("This is a biomedical note.", 0)]
        self.num_classes = num_classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class AdapterWithHead(nn.Module):
    def __init__(self, adapter: nn.Module, hidden_dim: int, num_classes: int):
        super().__init__()
        self.adapter = adapter
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, **kwargs):
        outputs = self.adapter(**kwargs)
        # try to access last_hidden_state like HF outputs
        x = getattr(outputs, "last_hidden_state", None)
        if x is None:
            # if adapter returns plain tensor, assume (B, T, H)
            x = outputs
        pooled = x.mean(dim=1)
        return self.classifier(pooled)


class TrainerModelWrapper(nn.Module):
    """Wraps adapter+head to return dict(loss=..., logits=...) when labels provided.

    This makes it compatible with MedicalModelTrainer which expects model(**batch)
    to return either a dict with 'loss' or a loss tensor directly.
    """

    def __init__(self, core: AdapterWithHead):
        super().__init__()
        self.core = core
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, **kwargs):
        labels = kwargs.pop("labels", None)
        logits = self.core(**kwargs)
        if labels is not None:
            loss = self.criterion(logits, labels)
            return {"loss": loss, "logits": logits}
        return logits


def train_one_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device
) -> Tuple[float, int]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    steps = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        steps += 1
    return total_loss / max(1, steps), steps


def parse_args():
    p = argparse.ArgumentParser(description="Training performance benchmark (tiny synthetic run)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-length", type=int, default=64)
    p.add_argument("--dataset-size", type=int, default=256)
    p.add_argument("--vocab-size", type=int, default=1024)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-classes", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--output",
        type=str,
        default=f"benchmarks/results/train_{datetime.now().strftime('%Y%m%d%H%M%S')}.json",
    )
    # Real adapter options
    p.add_argument(
        "--use-real-adapter", action="store_true", help="Use a small real adapter training step"
    )
    p.add_argument(
        "--adapter", type=str, default="biobert", help="Adapter type: biobert or clinicalbert"
    )
    p.add_argument(
        "--dataset-file", type=str, default="benchmarks/datasets/mimic_notes_sample.jsonl"
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu"
    )
    Path(Path(args.output).parent).mkdir(parents=True, exist_ok=True)

    # Prepare data/model/optimizer
    use_real = bool(args.use_real_adapter)
    real_info: Dict[str, Any] = {}
    used_trainer = False
    if use_real:
        try:
            from medvllm.models.adapters import BioBERTAdapter, ClinicalBERTAdapter
            from medvllm.training.trainer import MedicalModelTrainer, TrainerConfig

            adapter_type = (args.adapter or "biobert").lower()
            if adapter_type == "biobert":
                adapter_cls, model_id = BioBERTAdapter, "monologg/biobert_v1.1_pubmed"
            else:
                adapter_cls, model_id = ClinicalBERTAdapter, "emilyalsentzer/Bio_ClinicalBERT"
            adapter = adapter_cls.from_pretrained(model_id).to(device)
            if device.type == "cuda":
                adapter = adapter.half()

            # Infer hidden size from adapter config if possible
            # Try adapter.config.hidden_size, then adapter.model.config.hidden_size; fallback to args.hidden_dim
            hidden_dim = args.hidden_dim
            try:
                hidden_dim = int(
                    getattr(
                        getattr(adapter, "config", None),
                        "hidden_size",
                        getattr(
                            getattr(getattr(adapter, "model", None), "config", None),
                            "hidden_size",
                            args.hidden_dim,
                        ),
                    )
                )
            except Exception:
                pass
            head_model = AdapterWithHead(
                adapter, hidden_dim=hidden_dim, num_classes=args.num_classes
            ).to(device)

            # Build dataset/loader from JSONL texts
            dspath = Path(args.dataset_file)
            text_ds = JsonlTextDataset(
                dspath, num_classes=args.num_classes, limit=args.dataset_size
            )

            def collate(batch: List[Tuple[str, int]]):
                texts = [t for t, _ in batch]
                labels = torch.tensor([y for _, y in batch], dtype=torch.long)
                enc = adapter.preprocess_biomedical_text(texts)
                # Ensure a plain dict so MedicalModelTrainer._move_to_device can move tensors
                try:
                    inputs: Dict[str, Any] = {k: v for k, v in enc.items()}
                except Exception:
                    inputs = dict(enc)
                inputs["labels"] = labels
                return inputs

            loader = DataLoader(
                text_ds,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=collate,
            )

            # Always use MedicalModelTrainer for real adapter path
            trainer_model = TrainerModelWrapper(head_model).to(device)
            trainer_cfg = TrainerConfig(
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=3e-4,
                device=str(device),
                optimizer="adamw",
                scheduler="none",
                log_every=10,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
            )
            trainer = MedicalModelTrainer(trainer_model, trainer_cfg)
            model = trainer  # sentinel for downstream
            optimizer = None  # type: ignore
            used_trainer = True
            real_info = {
                "model_type": f"{adapter_type}_adapter",
                "model_id": model_id,
                "dataset": str(dspath),
            }
        except Exception as e:
            print(f"Falling back to TinyModel: real adapter path failed: {e}")
            use_real = False

    if not use_real:
        dataset = TinyTextDataset(
            args.dataset_size, args.seq_length, args.vocab_size, args.num_classes, device
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        model = TinyModel(args.vocab_size, args.hidden_dim, args.num_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    total_tokens = (
        args.epochs
        * math.ceil(args.dataset_size / args.batch_size)
        * args.batch_size
        * args.seq_length
    )

    mem_prof = MemoryProfiler(
        device=device.type if isinstance(device, torch.device) else str(device)
    )

    start = time.time()
    with mem_prof.profile():
        total_steps = 0
        epoch_losses = []
        if use_real and used_trainer:
            # Use the trainer's train loop; we won't have per-step loss, so record epochs only
            model.train(
                train_dataset=loader, eval_dataset=None, output_dir="benchmarks/results/tmp_train"
            )
            # Loss is logged internally; set placeholder
            epoch_losses.append(0.0)
            total_steps = len(loader) * args.epochs
        else:
            for _ in range(args.epochs):
                avg_loss, steps = train_one_epoch(model, loader, optimizer, device)
                epoch_losses.append(avg_loss)
                total_steps += steps
    total_time = time.time() - start

    tokens_per_sec = total_tokens / max(1e-6, total_time)
    avg_step_time = total_time / max(1, total_steps)

    result = {
        "timestamp": datetime.now().isoformat(),
        "benchmark_type": "training",
        "model_type": real_info.get("model_type", "tiny_trainer"),
        "model_id": real_info.get("model_id", "TinyModel-v1"),
        "dataset": real_info.get("dataset", "synthetic"),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seq_length": args.seq_length,
        "dataset_size": args.dataset_size,
        "avg_epoch_loss": sum(epoch_losses) / max(1, len(epoch_losses)),
        "avg_step_time_sec": avg_step_time,
        "total_time_sec": total_time,
        "tokens_per_second": tokens_per_sec,
        "device": str(device),
        "memory_usage_mb": getattr(mem_prof, "results", {}),
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved training benchmark to {args.output}")


if __name__ == "__main__":
    main()
