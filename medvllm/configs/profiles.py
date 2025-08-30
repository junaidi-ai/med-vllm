from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from medvllm.config import Config
from medvllm.cli.utils import debug


def _repo_root() -> Path:
    # medvllm/configs/profiles.py -> medvllm -> repo_root
    return Path(__file__).resolve().parents[2]


def _default_profiles_dir() -> Path:
    return _repo_root() / "configs" / "deployment"


def _resolve_profile_path(profile: str | os.PathLike[str]) -> Path:
    p = Path(profile)
    if p.suffix.lower() == ".json" and p.exists():
        return p
    # Treat as named profile under configs/deployment/
    candidate = _default_profiles_dir() / f"{p.stem}.json"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Deployment profile not found: {profile}")


def load_profile(profile: str | os.PathLike[str] | None = None) -> Dict[str, Any]:
    """Load a deployment profile JSON by name (cpu, gpu_8bit) or path.

    Resolution order:
    1) Explicit 'profile' argument when provided
    2) Environment MEDVLLM_DEPLOYMENT_PROFILE when set

    Returns the parsed JSON dict. Raises FileNotFoundError/ValueError on error.
    """
    selected = profile or os.environ.get("MEDVLLM_DEPLOYMENT_PROFILE")
    if not selected:
        return {}
    path = _resolve_profile_path(selected)
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception as e:
            raise ValueError(f"Invalid deployment profile JSON: {path}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"Profile must be a JSON object: {path}")
    _validate_profile_shape(data, source=str(path))
    return data


def profile_engine_kwargs(profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract engine kwargs from a loaded profile, filtered to Config fields.

    Unknown keys are ignored. This allows forward-compat fields like 'device'.
    """
    if not profile_data:
        return {}
    engine = profile_data.get("engine") or {}
    if not isinstance(engine, dict):
        return {}
    cfg_fields = {f.name for f in Config.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return {k: v for k, v in engine.items() if k in cfg_fields}


def resolve_profile_engine_kwargs(
    profile: str | os.PathLike[str] | None,
    *,
    optimize_for: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience: load profile (or env) and return filtered engine kwargs.

    If no explicit profile is provided and the env var is unset, attempts to
    auto-select a suitable profile based on detected hardware.
    """
    data = load_profile(profile)
    if not data:
        # Try auto-selection when nothing specified
        autopath = select_profile_by_hardware(optimize_for=optimize_for)
        if autopath is not None:
            try:
                data = load_profile(autopath)
                debug(
                    f"Auto-selected deployment profile: {autopath.stem}"
                    + (f" (optimize_for={optimize_for})" if optimize_for else "")
                )
            except Exception:
                data = {}
    return profile_engine_kwargs(data)


def select_profile_by_hardware(*, optimize_for: Optional[str] = None) -> Optional[Path]:
    """Best-effort selection of a deployment profile based on hardware.

    Heuristics:
    - If CUDA not available: prefer edge CPU INT8 profile if present, else generic cpu.json
    - If multiple GPUs (>=4): prefer cloud TP4 FP16
    - If single GPU and bf16 likely available: prefer on-prem BF16 + Flash
    - Otherwise: fall back to gpu_8bit or gpu_4bit based on memory

    Returns a Path to the chosen profile JSON, or None if no suitable file exists.
    """
    profiles_dir = _default_profiles_dir()

    # Helper to safely resolve by name
    def _p(name: str) -> Optional[Path]:
        f = profiles_dir / f"{name}.json"
        return f if f.exists() else None

    # Try importing torch lazily; fall back to CPU
    try:
        import importlib

        torch = importlib.import_module("torch")  # type: ignore
        cuda = getattr(torch, "cuda", None)
    except Exception:
        cuda = None

    if not cuda or not getattr(cuda, "is_available", lambda: False)():
        if optimize_for == "memory":
            return _p("edge_cpu_int8") or _p("cpu")
        # latency/throughput fall back to generic cpu if present; else edge cpu
        return _p("cpu") or _p("edge_cpu_int8")

    # CUDA available
    try:
        device_count = int(cuda.device_count())  # type: ignore[attr-defined]
    except Exception:
        device_count = 1

    if device_count >= 4:
        # Throughput prefers multi-GPU TP
        if optimize_for in (None, "throughput"):
            cand = _p("cloud_gpu_tp4_fp16")
            if cand:
                return cand

    # Inspect first device properties for memory and arch (bf16 hint)
    total_mem_gb = None
    sm_major = None
    try:
        props = cuda.get_device_properties(0)  # type: ignore[attr-defined]
        total_mem_gb = getattr(props, "total_memory", None)
        if isinstance(total_mem_gb, int):
            total_mem_gb = total_mem_gb / (1024**3)
        sm_major = getattr(props, "major", None)
    except Exception:
        pass

    # Rough heuristic: Ampere (SM 8.x) or newer likely supports bf16
    if sm_major is not None and sm_major >= 8:
        # Latency prefers bf16/flash on single GPU
        if optimize_for in (None, "latency", "throughput"):
            cand = _p("onprem_gpu_bf16_flash")
            if cand:
                return cand

    # Memory-based fallback for quantized GPU profiles
    if optimize_for == "memory":
        return _p("gpu_4bit") or _p("gpu_8bit") or _p("edge_cpu_int8")
    if isinstance(total_mem_gb, (int, float)) and total_mem_gb < 10:
        return _p("gpu_4bit") or _p("gpu_8bit")
    # Default preference order when unbiased
    return _p("gpu_8bit") or _p("onprem_gpu_bf16_flash") or _p("cloud_gpu_tp4_fp16")


def _validate_profile_shape(data: Dict[str, Any], source: str | None = None) -> None:
    """Lightweight validation for deployment profile structure.

    - Top-level: optional "profile" (str), optional "description" (str), optional "category"
      (one of: "edge", "onprem", "cloud"), required "engine" (dict)
    - "engine" keys must be a subset of `Config` fields (unknown are ignored downstream)
    - Basic type checks only; detailed semantic validation occurs in `Config`
    """
    if not isinstance(data, dict):
        raise ValueError(_fmt(source, "Profile root must be an object"))

    engine = data.get("engine")
    if engine is None or not isinstance(engine, dict):
        raise ValueError(_fmt(source, "Profile must contain an 'engine' object"))

    if "profile" in data and not isinstance(data["profile"], str):
        raise ValueError(_fmt(source, "'profile' must be a string when present"))
    if "description" in data and not isinstance(data["description"], str):
        raise ValueError(_fmt(source, "'description' must be a string when present"))
    if "category" in data:
        cat = data["category"]
        if cat not in ("edge", "onprem", "cloud"):
            raise ValueError(_fmt(source, "'category' must be one of: edge, onprem, cloud"))

    # Filter check: engine keys should map to Config fields (unknowns permitted, ignored later)
    cfg_fields = {f.name for f in Config.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    for k in engine.keys():
        # Allow unknown keys to support forward-compat (e.g., 'device')
        # No action needed; just ensure keys are strings
        if not isinstance(k, str):
            raise ValueError(_fmt(source, "Engine keys must be strings"))


def _fmt(source: str | None, message: str) -> str:
    return f"{message} ({source})" if source else message
