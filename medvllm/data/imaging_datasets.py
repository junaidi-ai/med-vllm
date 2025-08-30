"""Simple imaging dataset adapters for medical data (DICOM, NIfTI, PNG/JPG).

This module provides a lightweight PyTorch-compatible dataset with:
- File discovery by pattern and format
- Optional 2D/3D loading (3D for NIfTI; DICOM assumed per-slice 2D for simplicity)
- Basic normalization (min-max or z-score)
- Minimal augmentation (random flips/rot90)
- Metadata extraction
- On-disk caching of preprocessed tensors

Dependencies: pydicom (for DICOM), nibabel (for NIfTI) are optional.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import os

import numpy as np
import torch

try:
    import pydicom  # type: ignore
except Exception:  # pragma: no cover - optional dep
    pydicom = None  # type: ignore

try:
    import nibabel as nib  # type: ignore
except Exception:  # pragma: no cover - optional dep
    nib = None  # type: ignore

from medvllm.data.config import MedicalDatasetConfig
from medvllm.data.tiling import tile_2d, tile_3d


class ImagingDataset:
    """Generic imaging dataset with simple preprocessing and caching.

    Note: We intentionally do not subclass torch.utils.data.Dataset to avoid
    import-time failures in environments that mock or stub torch.utils.
    The class is still fully compatible with PyTorch DataLoader since it
    implements __len__ and __getitem__.
    """

    def __init__(self, config: MedicalDatasetConfig) -> None:
        self.config = config
        if not self.config.data_dir:
            raise ValueError("ImagingDataset requires config.data_dir")
        self.root = Path(self.config.data_dir)
        if not self.root.exists():
            raise FileNotFoundError(f"data_dir not found: {self.root}")

        # Infer pattern by format
        self.image_format = (self.config.image_format or "").lower()
        pattern = self.config.pattern
        if not pattern:
            if self.image_format in {"dicom", "dcm"}:
                pattern = "**/*.dcm"
            elif self.image_format in {"nifti", "nii", "nii.gz"}:
                pattern = "**/*.nii*"
            else:
                pattern = "**/*.*"  # fallback
        self.files = sorted([p for p in self.root.glob(pattern) if p.is_file()])
        if not self.files:
            raise ValueError(f"No files found under {self.root} with pattern {pattern}")

        # Optional annotations mapping
        self.annotations: Dict[str, Any] = {}
        if self.config.annotation_path:
            ap = Path(self.config.annotation_path)
            if ap.exists():
                with ap.open("r", encoding="utf-8") as f:
                    self.annotations = json.load(f)

        # Prepare cache directory
        self.cache_dir = Path(self.config.cache_dir) if self.config.cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Defaults
        self.is_3d = (
            bool(self.config.is_3d)
            if self.config.is_3d is not None
            else (self.image_format in {"nifti", "nii", "nii.gz"})
        )
        self.normalization = (self.config.normalization or "none").lower()
        self.augment = bool(self.config.augment)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        cache_key = self._cache_key(path)
        tensor = None
        meta: Dict[str, Any]
        # Try cache first
        if self.cache_dir is not None:
            cached = self.cache_dir / f"{cache_key}.pt"
            meta_p = self.cache_dir / f"{cache_key}.meta.json"
            if cached.exists() and meta_p.exists():
                tensor = torch.load(cached)
                with meta_p.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
        # Load image (if not cached)
        if tensor is None:
            # Load image
            if self.image_format in {"dicom", "dcm"}:
                image, meta = self._load_dicom(path)
            elif self.image_format in {"nifti", "nii", "nii.gz"}:
                image, meta = self._load_nifti(path)
            else:
                image, meta = self._load_generic(path)

            # Normalize
            image = self._normalize(image)

            # Augment (very light)
            if self.augment:
                image = self._augment(image)

            # Optional Triton/torch op (prototype)
            if bool(self.config.enable_triton_ops):
                op = (self.config.triton_op or "").lower()
                win = int(getattr(self.config, "triton_window", 3) or 3)
                if op in {"median2d", "boxblur2d"} and win >= 1:
                    try:
                        image = self._apply_2d_op(image, op, win)
                    except Exception:
                        # Best-effort: ignore failures
                        pass

            # Convert to tensor with channel-first
            tensor = self._to_tensor(image)

            # Cache
            if self.cache_dir is not None:
                cached = self.cache_dir / f"{cache_key}.pt"
                meta_p = self.cache_dir / f"{cache_key}.meta.json"
                try:
                    torch.save(tensor, cached)
                    with meta_p.open("w", encoding="utf-8") as f:
                        json.dump(meta, f)
                except Exception:
                    pass  # cache best-effort

        item = self._build_item(tensor, meta, path)
        # Optional tiling (adds keys only when enabled)
        try:
            if self.is_3d and bool(getattr(self.config, "enable_tiling_3d", False)):
                tsz3 = getattr(self.config, "tile_size_3d", None) or (32, 128, 128)
                tstr3 = getattr(self.config, "tile_stride_3d", None) or tsz3
                tiles, locs = tile_3d(tensor, tsz3, tstr3)
                item["tiles"] = tiles
                item["tile_locs"] = locs
            elif (not self.is_3d) and bool(getattr(self.config, "enable_tiling_2d", False)):
                tsz2 = getattr(self.config, "tile_size_2d", None) or (256, 256)
                tstr2 = getattr(self.config, "tile_stride_2d", None) or tsz2
                tiles, locs = tile_2d(tensor, tsz2, tstr2)
                item["tiles"] = tiles
                item["tile_locs"] = locs
        except Exception:
            # Best-effort: if tiling fails, just return the base item
            pass

        return item

    def _build_item(self, tensor: torch.Tensor, meta: Dict[str, Any], path: Path) -> Dict[str, Any]:
        # Attach label if available via annotations
        label = None
        key = path.name
        if key in self.annotations:
            label = self.annotations[key]
        elif str(path) in self.annotations:
            label = self.annotations[str(path)]

        item: Dict[str, Any] = {
            "image": tensor,
            "meta": meta,
            "path": str(path),
        }
        if label is not None:
            # If label is scalar, convert to tensor
            if isinstance(label, (int, float)):
                item["label"] = torch.as_tensor(label)
            else:
                item["label"] = label
        return item

    def _cache_key(self, path: Path) -> str:
        h = hashlib.sha1()
        cfg = {
            "norm": self.normalization,
            "is_3d": self.is_3d,
            "aug": self.augment,
            "fmt": self.image_format,
        }
        h.update(str(path).encode("utf-8"))
        h.update(json.dumps(cfg, sort_keys=True).encode("utf-8"))
        return h.hexdigest()

    def _load_dicom(self, path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        if pydicom is None:
            raise ImportError("pydicom is required for DICOM support. Install 'pydicom'.")
        ds = pydicom.dcmread(str(path))
        pixels = ds.pixel_array.astype(np.float32)
        # Rescale if available
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        pixels = pixels * slope + intercept
        meta = {
            "PatientID": getattr(ds, "PatientID", None),
            "StudyInstanceUID": getattr(ds, "StudyInstanceUID", None),
            "SeriesInstanceUID": getattr(ds, "SeriesInstanceUID", None),
            "Modality": getattr(ds, "Modality", None),
            "Rows": int(getattr(ds, "Rows", pixels.shape[-2] if pixels.ndim >= 2 else 0)),
            "Columns": int(getattr(ds, "Columns", pixels.shape[-1] if pixels.ndim >= 2 else 0)),
        }
        return pixels, meta

    def _load_nifti(self, path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        if nib is None:
            raise ImportError("nibabel is required for NIfTI support. Install 'nibabel'.")
        img = nib.load(str(path))
        data = img.get_fdata(dtype=np.float32)
        hdr = img.header
        meta = {
            "dim": tuple(int(x) for x in hdr.get_data_shape()),
            "zooms": tuple(float(x) for x in hdr.get_zooms()),
            "datatype": int(hdr.get_data_dtype().num),
        }
        return data, meta

    def _load_generic(self, path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Fallback: try numpy load for .npy or imageio for images if present
        ext = path.suffix.lower()
        if ext == ".npy":
            arr = np.load(str(path))
            return arr.astype(np.float32), {"shape": arr.shape}
        try:  # pragma: no cover - optional dependency path
            import imageio.v3 as iio  # type: ignore

            arr = iio.imread(str(path))
            return arr.astype(np.float32), {"shape": arr.shape}
        except Exception as e:  # pragma: no cover - best-effort
            raise ValueError(f"Unsupported file type for {path}: {e}")

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        mode = self.normalization
        if mode == "minmax":
            min_v = np.nanmin(arr)
            max_v = np.nanmax(arr)
            if max_v > min_v:
                arr = (arr - min_v) / (max_v - min_v)
            else:
                arr = np.zeros_like(arr, dtype=np.float32)
        elif mode == "zscore":
            mean = np.nanmean(arr)
            std = float(np.nanstd(arr))
            arr = (arr - mean) / (std + 1e-6)
        return arr.astype(np.float32)

    def _augment(self, arr: np.ndarray) -> np.ndarray:
        # Very lightweight, deterministic-ish augmentations
        # random flips
        if np.random.rand() < 0.5:
            arr = np.flip(arr, axis=-1)
        if np.random.rand() < 0.5:
            arr = np.flip(arr, axis=-2)
        # occasional 90-degree rotation on last two dims
        if np.random.rand() < 0.25:
            arr = np.rot90(arr, k=1, axes=(-2, -1))
        return arr.copy()

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        # Ensure channel-first tensor shape
        if self.is_3d:
            # Expect (D, H, W) -> (1, D, H, W)
            if arr.ndim == 3:
                arr = arr[None, ...]
            elif arr.ndim == 4 and arr.shape[0] in (1, 3):
                pass  # already channel-first likely
            else:
                # Best-effort: collapse extra dims
                arr = np.squeeze(arr)
                if arr.ndim == 3:
                    arr = arr[None, ...]
        else:
            # Expect (H, W) or (H, W, C) -> (C, H, W)
            if arr.ndim == 2:
                arr = arr[None, ...]
            elif arr.ndim == 3:
                arr = np.moveaxis(arr, -1, 0)
            else:
                arr = np.squeeze(arr)
                if arr.ndim == 2:
                    arr = arr[None, ...]
        return torch.from_numpy(arr.astype(np.float32))

    # --- Experimental: 2D ops using Triton if available, else torch ---
    def _apply_2d_op(self, arr: np.ndarray, op: str, window: int) -> np.ndarray:
        # Work on last two dims as HxW; support 2D or 3D/4D by applying per-slice
        arr_np = np.asarray(arr, dtype=np.float32)
        if arr_np.ndim == 2:
            return self._apply_2d_op_single(arr_np, op, window)
        elif arr_np.ndim == 3:
            # (C,H,W) or (D,H,W) -> map over first dim
            out = [self._apply_2d_op_single(arr_np[i], op, window) for i in range(arr_np.shape[0])]
            return np.stack(out, axis=0)
        elif arr_np.ndim == 4:
            # (N,C,H,W) -> map over N,C
            out = []
            for n in range(arr_np.shape[0]):
                out.append(
                    [
                        self._apply_2d_op_single(arr_np[n, c], op, window)
                        for c in range(arr_np.shape[1])
                    ]
                )
            return np.stack([np.stack(x, axis=0) for x in out], axis=0)
        else:
            return arr_np

    def _apply_2d_op_single(self, img: np.ndarray, op: str, window: int) -> np.ndarray:
        k = max(1, int(window))
        k = k if k % 2 == 1 else k + 1  # force odd
        pad = k // 2
        # Try Triton path (optional), otherwise torch
        use_triton = False
        try:  # detect triton import presence
            import triton  # type: ignore

            _ = triton.runtime
            use_triton = True
        except Exception:
            use_triton = False

        if op == "boxblur2d":
            # Torch avgpool as fallback, with replication padding
            t = torch.from_numpy(img[None, None, ...].astype(np.float32))
            t = torch.nn.functional.pad(t, (pad, pad, pad, pad), mode="replicate")
            out = torch.nn.functional.avg_pool2d(t, kernel_size=k, stride=1)
            return out[0, 0].numpy()
        elif op == "median2d":
            # Torch unfold-based median filter
            t = torch.from_numpy(img[None, None, ...].astype(np.float32))
            t = torch.nn.functional.pad(t, (pad, pad, pad, pad), mode="replicate")
            patches = torch.nn.functional.unfold(t, kernel_size=k, stride=1)  # (N*K*K, L)
            patches = patches.transpose(1, 2)  # (L, K*K)
            med = patches.median(dim=-1).values  # (L,)
            H, W = img.shape
            out = med.view(1, 1, H, W)
            return out[0, 0].numpy()
        else:
            return img.astype(np.float32)


def get_imaging_dataset(config: MedicalDatasetConfig | Dict[str, Any]) -> ImagingDataset:
    """Factory to build an ImagingDataset from config or dict."""
    if isinstance(config, dict):
        cfg = MedicalDatasetConfig(**config)
    else:
        cfg = config
    return ImagingDataset(cfg)
