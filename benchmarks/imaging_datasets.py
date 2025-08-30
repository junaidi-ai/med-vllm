from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, expected_sha256: str | None = None) -> Path:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if expected_sha256 and _sha256(dest) == expected_sha256:
            return dest
    urllib.request.urlretrieve(url, dest)  # nosec - controlled URL via CI env
    if expected_sha256:
        got = _sha256(dest)
        if got != expected_sha256:
            raise RuntimeError(f"Checksum mismatch for {dest}: {got} != {expected_sha256}")
    return dest


def load_seg2d_small(
    root: str | Path = "benchmarks/datasets_cache",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load a tiny 2D segmentation dataset (images, masks) as tensors.
    - If MEDVLLM_SEG2D_URL is set, downloads a .npz with arrays 'imgs' [N,H,W] and 'masks' [N,H,W].
    - Else, generates deterministic synthetic shapes (N=6) for CI.

    Returns:
        imgs: Float tensor [N,1,H,W]
        masks: Float tensor [N,1,H,W] (0/1)
    """
    root = Path(root)
    url = os.getenv("MEDVLLM_SEG2D_URL")
    checksum = os.getenv("MEDVLLM_SEG2D_SHA256")

    if url:
        cache = root / "seg2d_small.npz"
        _download(url, cache, expected_sha256=checksum)
        data = np.load(cache)
        imgs = data["imgs"].astype(np.float32)
        masks = data["masks"].astype(np.float32)
    else:
        # Deterministic synthetic dataset
        rng = np.random.default_rng(0)
        N, H, W = 6, 128, 128
        imgs = np.zeros((N, H, W), dtype=np.float32)
        masks = np.zeros((N, H, W), dtype=np.float32)
        for i in range(N):
            # random circle
            cy, cx = rng.integers(32, 96), rng.integers(32, 96)
            r = rng.integers(10, 20)
            yy, xx = np.ogrid[:H, :W]
            circ = (yy - cy) ** 2 + (xx - cx) ** 2 <= (r**2)
            masks[i][circ] = 1.0
            # image as blurred mask + noise
            imgs[i] = masks[i] + 0.1 * rng.standard_normal((H, W)).astype(np.float32)
        # simple normalization
        imgs = (imgs - imgs.mean()) / (imgs.std() + 1e-6)
    imgs_t = torch.from_numpy(imgs).unsqueeze(1)  # [N,1,H,W]
    masks_t = torch.from_numpy(masks).unsqueeze(1)  # [N,1,H,W]
    return imgs_t, masks_t
