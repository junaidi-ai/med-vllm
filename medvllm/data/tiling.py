"""Tiling and sliding-window utilities for large 2D/3D medical images.

Functions operate on numpy arrays or torch tensors. When tensors are passed,
outputs are tensors on the same device; when numpy arrays are passed, outputs
are numpy arrays.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


def _is_tensor(x: ArrayLike) -> bool:
    return isinstance(x, torch.Tensor)


def _slice_nd(arr: ArrayLike, slices: Sequence[slice]) -> ArrayLike:
    if _is_tensor(arr):
        return arr[tuple(slices)]
    return arr[tuple(slices)]


def _stack(items: List[ArrayLike]) -> ArrayLike:
    if len(items) == 0:
        return items
    if _is_tensor(items[0]):
        return torch.stack(items, dim=0)
    return np.stack(items, axis=0)


def sliding_window_indices_2d(
    h: int, w: int, tile: Tuple[int, int], stride: Tuple[int, int]
) -> List[Tuple[slice, slice]]:
    th, tw = tile
    sh, sw = stride
    hs = list(range(0, max(1, h - th + 1), max(1, sh))) or [0]
    ws = list(range(0, max(1, w - tw + 1), max(1, sw))) or [0]
    # Ensure coverage of edges
    if hs[-1] != max(0, h - th):
        hs.append(max(0, h - th))
    if ws[-1] != max(0, w - tw):
        ws.append(max(0, w - tw))
    return [(slice(i, i + th), slice(j, j + tw)) for i in hs for j in ws]


def sliding_window_indices_3d(
    d: int, h: int, w: int, tile: Tuple[int, int, int], stride: Tuple[int, int, int]
) -> List[Tuple[slice, slice, slice]]:
    td, th, tw = tile
    sd, sh, sw = stride
    ds = list(range(0, max(1, d - td + 1), max(1, sd))) or [0]
    hs = list(range(0, max(1, h - th + 1), max(1, sh))) or [0]
    ws = list(range(0, max(1, w - tw + 1), max(1, sw))) or [0]
    if ds[-1] != max(0, d - td):
        ds.append(max(0, d - td))
    if hs[-1] != max(0, h - th):
        hs.append(max(0, h - th))
    if ws[-1] != max(0, w - tw):
        ws.append(max(0, w - tw))
    return [
        (slice(k, k + td), slice(i, i + th), slice(j, j + tw)) for k in ds for i in hs for j in ws
    ]


def tile_2d(
    x: ArrayLike, tile: Tuple[int, int], stride: Tuple[int, int]
) -> Tuple[ArrayLike, List[Tuple[int, int]]]:
    """Tile a 2D array or a CHW/CxHxW tensor along the last two dims (H, W).

    Returns (tiles, locations) where locations are (h0, w0) offsets.
    """
    arr = x
    if arr.ndim == 2:
        H, W = arr.shape
        idx = sliding_window_indices_2d(H, W, tile, stride)
        tiles = [_slice_nd(arr, (sH, sW)) for (sH, sW) in idx]
        locs = [(sH.start, sW.start) for (sH, sW) in idx]
        return _stack(tiles), locs
    elif arr.ndim >= 3:
        # Assume ... x H x W
        H, W = arr.shape[-2], arr.shape[-1]
        idx = sliding_window_indices_2d(H, W, tile, stride)
        tiles = []
        locs = []
        for sH, sW in idx:
            slices = [slice(None)] * (arr.ndim - 2) + [sH, sW]
            tiles.append(_slice_nd(arr, slices))
            locs.append((sH.start, sW.start))
        return _stack(tiles), locs
    else:
        raise ValueError("Input must be 2D or higher with spatial dims on the last axes")


def tile_3d(
    x: ArrayLike, tile: Tuple[int, int, int], stride: Tuple[int, int, int]
) -> Tuple[ArrayLike, List[Tuple[int, int, int]]]:
    """Tile a 3D volume or a CxDxHxW tensor along last three dims (D, H, W).

    Returns (tiles, locations) where locations are (d0, h0, w0).
    """
    arr = x
    if arr.ndim == 3:
        D, H, W = arr.shape
        idx = sliding_window_indices_3d(D, H, W, tile, stride)
        tiles = [_slice_nd(arr, (sD, sH, sW)) for (sD, sH, sW) in idx]
        locs = [(sD.start, sH.start, sW.start) for (sD, sH, sW) in idx]
        return _stack(tiles), locs
    elif arr.ndim >= 4:
        # Assume ... x D x H x W
        D, H, W = arr.shape[-3], arr.shape[-2], arr.shape[-1]
        idx = sliding_window_indices_3d(D, H, W, tile, stride)
        tiles = []
        locs = []
        for sD, sH, sW in idx:
            slices = [slice(None)] * (arr.ndim - 3) + [sD, sH, sW]
            tiles.append(_slice_nd(arr, slices))
            locs.append((sD.start, sH.start, sW.start))
        return _stack(tiles), locs
    else:
        raise ValueError("Input must be 3D or higher with spatial dims on the last axes")
