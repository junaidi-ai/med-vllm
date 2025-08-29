"""Simple medical model optimizer utilities.

Provides:
- MedicalModelOptimizer: quantization (dynamic/bitsandbytes), memory tweaks,
  micro-benchmarking, and export helpers.

Note: bitsandbytes paths require a CUDA environment and the bitsandbytes package.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - optional at runtime
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore


@dataclass
class OptimizerConfig:
    model_name_or_path: str
    quantization_bits: Optional[int] = None  # 4 or 8
    quantization_method: Optional[str] = None  # dynamic|cpu|torch|bnb-8bit|bnb-nf4
    device_map: Optional[str] = None  # e.g., "auto"


class MedicalModelOptimizer:
    def __init__(self, model: Any, config: OptimizerConfig) -> None:
        self.model = model
        self.config = config

    # Quantization API
    def quantize(self, bits: int = 8, method: str = "dynamic") -> Any:
        method_l = (method or "").lower()
        if bits not in (4, 8):
            raise ValueError(f"Unsupported quantization bits: {bits}")

        # bitsandbytes path: load via HF flags
        if "bnb" in method_l:
            if AutoModelForCausalLM is None:
                raise RuntimeError("transformers not available for bitsandbytes loading")
            load_kwargs: Dict[str, Any] = {
                "trust_remote_code": True,
            }
            if bits == 8:
                load_kwargs["load_in_8bit"] = True
            else:
                load_kwargs["load_in_4bit"] = True
                if "nf4" in method_l:
                    load_kwargs["bnb_4bit_quant_type"] = "nf4"
            if self.config.device_map:
                load_kwargs["device_map"] = self.config.device_map
            else:
                load_kwargs["device_map"] = "auto"

            # If model is a name/path, re-load; if it's a module with .name_or_path try that
            name_or_path = (
                getattr(self.model, "name_or_path", None) or self.config.model_name_or_path
            )
            self.model = AutoModelForCausalLM.from_pretrained(name_or_path, **load_kwargs)
            return self.model

        # Default: dynamic int8 on CPU
        from medvllm.optim.quantization import QuantizationConfig, quantize_model

        self.model = self.model.to("cpu")
        qcfg = QuantizationConfig(dtype=torch.qint8, inplace=False)
        self.model = quantize_model(self.model, qcfg)
        return self.model

    # Memory tweaks geared for inference
    def optimize_memory(self) -> None:
        try:
            self.model.eval()
        except Exception:
            pass
        torch.set_grad_enabled(False)
        # Mixed-precision and backends toggles (conservative defaults)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Optional: enable TF32 for Ampere+
            try:
                torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
                torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
            except Exception:
                pass

    # Micro benchmark
    def benchmark(
        self,
        input_texts: Iterable[str],
        batch_sizes: Iterable[int] = (1, 2, 4),
        iterations: int = 10,
        use_tqdm: bool = False,
    ) -> Dict[str, Any]:
        if AutoTokenizer is None:
            raise RuntimeError("transformers is required for benchmarking tokenization")

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path, use_fast=True)
        device = (
            next(self.model.parameters()).device if hasattr(self.model, "parameters") else "cpu"
        )

        def encode_batch(texts: List[str]):
            return tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

        results: Dict[str, Any] = {}
        for bs in batch_sizes:
            # Prepare a batch (repeat or trim)
            texts = list(input_texts)
            if not texts:
                texts = ["Patient presents with chest pain."]
            batch = (texts * ((bs + len(texts) - 1) // len(texts)))[:bs]

            # Warmup
            with torch.inference_mode():
                for _ in range(2):
                    enc = encode_batch(batch)
                    enc = {k: v.to(device) for k, v in enc.items()}
                    _ = self.model(**enc)

            # Measure
            start = time.perf_counter()
            tok_count = 0
            with torch.inference_mode():
                for _ in range(iterations):
                    enc = encode_batch(batch)
                    input_lengths = int(enc["input_ids"].numel())
                    tok_count += input_lengths
                    enc = {k: v.to(device) for k, v in enc.items()}
                    _ = self.model(**enc)
            elapsed = time.perf_counter() - start

            mem_info: Dict[str, Any] = {}
            if torch.cuda.is_available():
                mem_info = {
                    "cuda_allocated_mb": int(torch.cuda.memory_allocated() / (1024 * 1024)),
                    "cuda_reserved_mb": int(torch.cuda.memory_reserved() / (1024 * 1024)),
                }

            results[str(bs)] = {
                "elapsed_s": elapsed,
                "iterations": iterations,
                "tokens_processed": tok_count,
                "throughput_tok_per_s": tok_count / elapsed if elapsed > 0 else None,
                **mem_info,
            }

        return results

    def export_optimized_model(self, output_path: str, fmt: str = "torchscript") -> str:
        os.makedirs(output_path, exist_ok=True)
        export_info = {"format": fmt}
        path = output_path

        if fmt == "onnx":
            # Minimal safe ONNX export attempt
            try:
                dummy = torch.randint(0, 100, (1, 8))
                torch.onnx.export(
                    self.model,
                    (dummy,),
                    os.path.join(output_path, "model.onnx"),
                    input_names=["input_ids"],
                    output_names=["logits"],
                    opset_version=17,
                    dynamic_axes={"input_ids": {0: "batch"}},
                )
                path = os.path.join(output_path, "model.onnx")
            except Exception as e:
                export_info["error"] = f"ONNX export failed: {e}"
        else:
            # TorchScript trace as a simple fallback
            try:
                self.model.eval()
                example = torch.randint(0, 100, (1, 8))
                traced = torch.jit.trace(self.model, (example,))
                ts_path = os.path.join(output_path, "model.ts")
                traced.save(ts_path)
                path = ts_path
            except Exception as e:
                export_info["error"] = f"TorchScript export failed: {e}"

        with open(os.path.join(output_path, "export_meta.json"), "w") as f:
            json.dump(export_info, f, indent=2)
        return path
