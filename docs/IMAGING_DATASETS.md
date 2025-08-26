# Imaging Datasets (DICOM & NIfTI)

Med vLLM now includes lightweight imaging dataset adapters for common medical formats.

- Supported formats: DICOM (`pydicom`), NIfTI (`nibabel`), PNG/JPG (`imageio`) and `.npy` arrays
- Preprocessing: normalization (`none|minmax|zscore`), minimal augmentation (random flips, 90° rotate)
- 2D/3D handling: channel-first tensors `(C, H, W)` or `(C, D, H, W)`
- Metadata: DICOM tags (PatientID, UIDs, Modality), NIfTI header (shape, zooms, dtype)
- Caching: preprocessed tensor + metadata caching in `cache_dir`

See the example: `examples/imaging_dataset_example.py`

## Installation (optional dependencies)

```
pip install pydicom nibabel imageio
```

Alternatively, add them via `requirements/requirements-medical.txt`.

## Quickstart

Use the unified factory `get_dataset()` to create either a text dataset (Hugging Face Datasets) or an imaging dataset (files on disk), based on the fields present in `MedicalDatasetConfig`.

### DICOM

```python
from medvllm.data import get_dataset

cfg = dict(
    name="demo_dicom",
    data_dir="/path/to/dicom_slices",
    image_format="dicom",
    normalization="zscore",
    augment=False,
    cache_dir="./.cache_imaging",
)

ds = get_dataset(cfg)
```

### NIfTI (3D)

```python
from medvllm.data import get_dataset

cfg = dict(
    name="demo_nifti",
    data_dir="/path/to/nifti_volumes",
    image_format="nifti",
    is_3d=True,
    normalization="minmax",
    augment=False,
    cache_dir="./.cache_imaging",
)

ds = get_dataset(cfg)
```

### DataLoader

```python
import torch

loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
first_batch = next(iter(loader))
print(first_batch["image"].shape)  # (B, C, H, W) or (B, C, D, H, W)
print(first_batch["meta"][0])     # sample metadata
```

## Config reference

Imaging-specific fields in `medvllm/data/config.py` (`MedicalDatasetConfig`):

- `data_dir`: root directory containing your files
- `image_format`: `dicom` | `nifti` | `png` | `jpg`
- `pattern`: optional glob to match files (defaults per format)
- `annotation_path`: optional JSON mapping from filename/path to label
- `is_3d`: whether to treat data as 3D volumes (defaults to True for NIfTI)
- `normalization`: `none` | `minmax` | `zscore`
- `augment`: enable minimal random flips/rot90
- `cache_dir`: where to store processed tensor + metadata cache

Example JSON config (loadable via `MedicalDatasetConfig.from_json_file()`):

```json
{
  "name": "demo_nifti",
  "data_dir": "/data/nifti",
  "image_format": "nifti",
  "is_3d": true,
  "normalization": "zscore",
  "augment": false,
  "cache_dir": "./.cache_imaging"
}
```

## Notes

- DICOM is treated as per-slice 2D for simplicity. NIfTI is ideal for 3D volumes.
- Caching keys include key preprocessing settings; delete `cache_dir` to rebuild.
- If a dependency is missing, relevant loaders will raise a clear ImportError; tests skip accordingly.
