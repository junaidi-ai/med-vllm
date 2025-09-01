"""Compatibility shims for optional hardware accelerators.

These helpers avoid hard-importing optional packages. They expose lightweight
capability checks and placeholders for adapter hooks in the future.
"""

from __future__ import annotations

from typing import Any, Dict

from .detect import _module_available


def is_tensorrt_available() -> bool:
    return _module_available("tensorrt")


def is_openvino_available() -> bool:
    return _module_available("openvino.runtime") or _module_available("openvino")


def is_onnxruntime_available() -> bool:
    return _module_available("onnxruntime") or _module_available("onnxruntime_gpu")


def is_coreml_available() -> bool:
    return _module_available("coremltools")


def is_directml_available() -> bool:
    return _module_available("torch_directml")


def describe_optional_backends() -> Dict[str, Any]:
    return {
        "tensorrt": is_tensorrt_available(),
        "openvino": is_openvino_available(),
        "onnxruntime": is_onnxruntime_available(),
        "coreml": is_coreml_available(),
        "directml": is_directml_available(),
    }
