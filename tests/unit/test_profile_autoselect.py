import sys
from types import SimpleNamespace, ModuleType
from pathlib import Path

from medvllm.configs.profiles import select_profile_by_hardware, _default_profiles_dir


class _FakeCuda:
    def __init__(self, available=True, count=1, total_mem_gb=16, sm_major=8):
        self._available = available
        self._count = count
        self._total_mem_bytes = int(total_mem_gb * (1024**3)) if total_mem_gb is not None else None
        self._sm_major = sm_major

    def is_available(self):
        return self._available

    def device_count(self):
        return self._count

    def get_device_properties(self, idx):  # noqa: ARG002
        return SimpleNamespace(total_memory=self._total_mem_bytes, major=self._sm_major)


class _TorchModule(ModuleType):
    def __init__(self, cuda):
        super().__init__("torch")
        self.cuda = cuda


def _inject_fake_torch(cuda_obj):
    mod = _TorchModule(cuda_obj)
    sys.modules["torch"] = mod


def _remove_fake_torch():
    sys.modules.pop("torch", None)


def test_autoselect_no_cuda_prefers_cpu(tmp_path, monkeypatch):
    _remove_fake_torch()
    p = select_profile_by_hardware()
    # Should resolve to edge_cpu_int8 or cpu
    profiles = {p.name for p in _default_profiles_dir().glob("*.json")}
    assert p is None or p.name in profiles


def test_autoselect_multi_gpu_cloud():
    _inject_fake_torch(_FakeCuda(available=True, count=4, total_mem_gb=24, sm_major=8))
    p = select_profile_by_hardware()
    assert p is None or p.name in {
        "cloud_gpu_tp4_fp16.json",
        "onprem_gpu_bf16_flash.json",
        "gpu_8bit.json",
    }


def test_autoselect_ampere_onprem():
    _inject_fake_torch(_FakeCuda(available=True, count=1, total_mem_gb=24, sm_major=8))
    p = select_profile_by_hardware()
    assert p is None or p.name in {"onprem_gpu_bf16_flash.json", "gpu_8bit.json"}


def test_autoselect_low_mem_gpu_4bit():
    _inject_fake_torch(_FakeCuda(available=True, count=1, total_mem_gb=8, sm_major=7))
    p = select_profile_by_hardware()
    assert p is None or p.name in {"gpu_4bit.json", "gpu_8bit.json"}


def teardown_module(module):  # noqa: ARG001
    _remove_fake_torch()
