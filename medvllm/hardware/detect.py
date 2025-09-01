"""Hardware detection utilities for Med vLLM.

Avoid hard dependencies. Use best-effort import checks and torch backends.
"""

from __future__ import annotations

import platform
import os
from typing import Any, Dict, List

import torch


def _gpu_devices() -> List[Dict[str, Any]]:
    devices: List[Dict[str, Any]] = []
    if not torch.cuda.is_available():
        return devices
    try:
        count = torch.cuda.device_count()
    except Exception:
        count = 0
    for i in range(count):
        try:
            cap = torch.cuda.get_device_capability(i)
            props = torch.cuda.get_device_properties(i)
            devices.append(
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "capability": f"{cap[0]}.{cap[1]}",
                    "total_mem_gb": getattr(props, "total_memory", 0) / (1024**3),
                }
            )
        except Exception:
            continue
    return devices


def _module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def get_hardware_profile() -> Dict[str, Any]:
    """Return a structured description of available hardware/backends."""
    prof: Dict[str, Any] = {
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
        },
        "devices": {
            "cpu": {
                "available": True,
                "threads": os.cpu_count(),
                "mkldnn": getattr(torch.backends, "mkldnn", None) is not None
                and torch.backends.mkldnn.is_available(),
            },
            "cuda": {
                "available": torch.cuda.is_available(),
                "version": torch.version.cuda if torch.cuda.is_available() else None,
                "cudnn": torch.backends.cudnn.is_available()
                if hasattr(torch.backends, "cudnn")
                else False,
                "devices": _gpu_devices(),
                "hip": getattr(torch.version, "hip", None),  # non-None suggests ROCm build
            },
            "mps": {
                "available": getattr(torch.backends, "mps", None) is not None
                and torch.backends.mps.is_available(),
            },
            "xpu": {
                # Intel GPU (XPU) via IPEX/oneAPI if present
                "available": _module_available("intel_extension_for_pytorch"),
            },
            "tpu": {
                "available": _module_available("torch_xla")
                or bool(os.environ.get("COLAB_TPU_ADDR")),
            },
        },
        "accelerators": {
            "tensorrt": _module_available("tensorrt"),
            "openvino": _module_available("openvino.runtime") or _module_available("openvino"),
            "onnxruntime": _module_available("onnxruntime") or _module_available("onnxruntime_gpu"),
            "coreml": _module_available("coremltools"),
            "directml": _module_available("torch_directml"),
        },
    }
    return prof


def list_available_backends() -> List[Dict[str, Any]]:
    """Summarize runnable backends with device string and precision support.

    Returns a list of entries like:
      {"name": "cpu", "device": "cpu", "precisions": {"fp32": True, "fp16": False, "int8": maybe}}
    """
    prof = get_hardware_profile()
    b: List[Dict[str, Any]] = []

    # CPU
    cpu_prec = {"fp32": True, "fp16": False, "int8": False}
    try:
        # int8 available if quantized engines exist
        engines = (
            getattr(torch.backends.quantized, "supported_engines", [])
            if hasattr(torch.backends, "quantized")
            else []
        )
        cpu_prec["int8"] = bool(engines)
    except Exception:
        pass
    b.append({"name": "cpu", "device": "cpu", "precisions": cpu_prec})

    # CUDA / ROCm
    if prof["devices"]["cuda"]["available"]:
        b.append(
            {
                "name": "cuda",
                "device": "cuda",
                "precisions": {"fp32": True, "fp16": True, "int8": False},
            }
        )

    # MPS
    if prof["devices"]["mps"]["available"]:
        b.append(
            {
                "name": "mps",
                "device": "mps",
                "precisions": {"fp32": True, "fp16": True, "int8": False},
            }
        )

    # XPU (Intel GPU)
    if prof["devices"]["xpu"]["available"]:
        b.append(
            {
                "name": "xpu",
                "device": "xpu",
                "precisions": {"fp32": True, "fp16": True, "int8": False},
            }
        )

    # Optional accelerators summary entries (not used for torch execution here)
    if prof["accelerators"]["tensorrt"]:
        b.append(
            {
                "name": "tensorrt",
                "device": "cuda",
                "precisions": {"fp32": True, "fp16": True, "int8": True},
            }
        )
    if prof["accelerators"]["openvino"]:
        b.append(
            {
                "name": "openvino",
                "device": "cpu",
                "precisions": {"fp32": True, "fp16": True, "int8": True},
            }
        )

    return b
