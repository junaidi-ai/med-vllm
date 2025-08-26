"""Example: Using get_dataset() for DICOM/NIfTI with a simple DataLoader.

This demo tries to construct imaging datasets if the corresponding directories
exist. It prints the first batch shapes and a sample of metadata.

Usage:
  python examples/imaging_dataset_example.py \
      --dicom_dir /path/to/dicom_slices \
      --nifti_dir /path/to/nifti_volumes \
      --cache_dir ./.cache_imaging

All arguments are optional. If a directory is missing, that part is skipped.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch

from medvllm.data import get_dataset, MedicalDatasetConfig


def build_and_show(config_dict: Dict[str, Any], title: str) -> None:
    try:
        ds = get_dataset(config_dict)
    except Exception as e:
        print(f"[SKIP] {title}: {e}")
        return
    if len(ds) == 0:
        print(f"[SKIP] {title}: empty dataset")
        return

    print(f"[OK] {title}: {len(ds)} samples")
    dl = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(dl))
    images = batch["image"]  # [B, C, H, W] or [B, C, D, H, W]
    print(f"  batch image tensor shape: {tuple(images.shape)}")
    print(
        f"  sample meta: {batch['meta'][0] if isinstance(batch['meta'], list) else batch['meta']}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom_dir", type=str, default=None)
    ap.add_argument("--nifti_dir", type=str, default=None)
    ap.add_argument("--cache_dir", type=str, default=None)
    args = ap.parse_args()

    cache_dir = args.cache_dir

    if args.dicom_dir and Path(args.dicom_dir).exists():
        dicom_cfg = dict(
            name="demo_dicom",
            data_dir=args.dicom_dir,
            image_format="dicom",
            normalization="zscore",
            augment=False,
            cache_dir=cache_dir,
        )
        build_and_show(dicom_cfg, "DICOM Dataset")
    else:
        print("[INFO] --dicom_dir not provided or not found; skipping DICOM demo")

    if args.nifti_dir and Path(args.nifti_dir).exists():
        nifti_cfg = dict(
            name="demo_nifti",
            data_dir=args.nifti_dir,
            image_format="nifti",
            is_3d=True,
            normalization="minmax",
            augment=False,
            cache_dir=cache_dir,
        )
        build_and_show(nifti_cfg, "NIfTI Dataset")
    else:
        print("[INFO] --nifti_dir not provided or not found; skipping NIfTI demo")


if __name__ == "__main__":
    main()
