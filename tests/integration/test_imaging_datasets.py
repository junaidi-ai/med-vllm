import os
from pathlib import Path

import numpy as np
import pytest

from medvllm.data import get_dataset


@pytest.mark.integration
def test_generic_npy_loading(tmp_path: Path):
    # Create a simple 2D array and save as .npy
    arr = (np.random.rand(16, 12) * 255).astype(np.float32)
    data_dir = tmp_path / "generic"
    data_dir.mkdir()
    np.save(data_dir / "sample.npy", arr)

    cfg = dict(
        name="generic_npy",
        data_dir=str(data_dir),
        pattern="**/*.npy",
        normalization="minmax",
        augment=False,
        cache_dir=str(tmp_path / "cache"),
        # image_format left None to trigger generic loader path
    )

    ds = get_dataset(cfg)
    assert len(ds) == 1
    item = ds[0]
    img = item["image"]
    assert img.ndim == 3 and img.shape[0] == 1  # (C, H, W)
    # After minmax, values should be in [0, 1]
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0
    # Ensure caching writes files
    key_files = list((tmp_path / "cache").glob("*.pt"))
    assert key_files, "expected cached tensor file to exist"


@pytest.mark.integration
@pytest.mark.skipif(
    pytest.importorskip("nibabel", reason="nibabel not installed") is None, reason="nibabel missing"
)
def test_nifti_loading_if_available(tmp_path: Path):
    import nibabel as nib  # type: ignore

    data_dir = tmp_path / "nifti"
    data_dir.mkdir()
    vol = np.random.rand(8, 10, 6).astype(np.float32)
    img = nib.Nifti1Image(vol, affine=np.eye(4))
    nib.save(img, str(data_dir / "sample.nii.gz"))

    cfg = dict(
        name="nifti_demo",
        data_dir=str(data_dir),
        image_format="nifti",
        is_3d=True,
        normalization="zscore",
        augment=False,
        cache_dir=str(tmp_path / "cache_nii"),
    )

    ds = get_dataset(cfg)
    assert len(ds) == 1
    item = ds[0]
    img_t = item["image"]
    # Expect (C, D, H, W)
    assert img_t.ndim == 4 and img_t.shape[0] == 1
    assert "dim" in item["meta"]


@pytest.mark.integration
@pytest.mark.skipif(
    pytest.importorskip("pydicom", reason="pydicom not installed") is None, reason="pydicom missing"
)
def test_dicom_loading_if_available(tmp_path: Path):
    import pydicom  # type: ignore
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import ImplicitVRLittleEndian

    data_dir = tmp_path / "dicom"
    data_dir.mkdir()

    # Create a minimal DICOM file with pixel data
    rows, cols = 16, 12
    pixels = (np.random.rand(rows, cols) * 1024).astype(np.uint16)

    meta = Dataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
    meta.MediaStorageSOPInstanceUID = "1.2.826.0.1.3680043.2.1125.1"
    meta.TransferSyntaxUID = ImplicitVRLittleEndian

    ds = FileDataset(str(data_dir / "sample.dcm"), {}, file_meta=meta, preamble=b"\x00" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = pixels.tobytes()
    ds.PatientID = "TESTPATIENT"
    ds.save_as(str(data_dir / "sample.dcm"))

    cfg = dict(
        name="dicom_demo",
        data_dir=str(data_dir),
        image_format="dicom",
        normalization="minmax",
        augment=False,
        cache_dir=str(tmp_path / "cache_dcm"),
    )

    ds_obj = get_dataset(cfg)
    assert len(ds_obj) == 1
    item = ds_obj[0]
    img_t = item["image"]
    # Expect (C, H, W)
    assert img_t.ndim == 3 and img_t.shape[0] == 1
    assert item["meta"].get("PatientID") == "TESTPATIENT"
