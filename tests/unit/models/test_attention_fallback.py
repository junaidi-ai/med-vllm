import warnings
import types
import torch
import pytest

from medvllm.layers.attention import Attention
import medvllm.layers.attention as attn_mod


class DummyTensor:
    def __init__(self, shape=(1, 8, 8)):
        self._shape = shape

    @property
    def shape(self):
        return self._shape

    def size(self, dim=None):
        if dim is None:
            return self._shape
        return self._shape[dim]

    @property
    def is_cuda(self):  # used in Attention.forward
        return False

    def view(self, *args, **kwargs):
        return self

    def transpose(self, *args, **kwargs):
        return self

    def contiguous(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def __mul__(self, other):
        return self

    def __add__(self, other):
        return self


def _mk_qkv():
    return DummyTensor(), DummyTensor(), DummyTensor()


def test_fallback_flash_unavailable_warns_and_uses_sdpa(monkeypatch):
    # Ensure FLASH is unavailable
    monkeypatch.setattr(attn_mod, "FLASH_ATTN_AVAILABLE", False, raising=False)
    F = torch.nn.functional
    # Inject a minimal SDPA if not present
    restored = False
    if not hasattr(F, "scaled_dot_product_attention"):

        def _mock_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True, scale=None):  # noqa: ARG002
            if scale is None:
                scale = q.size(-1) ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            if is_causal:
                Lq, Lk = q.size(-2), k.size(-2)
                mask = torch.triu(torch.ones(Lq, Lk, device=q.device, dtype=torch.bool), diagonal=1)
                attn = attn.masked_fill(mask, float('-inf'))
            p = torch.softmax(attn, dim=-1)
            return torch.matmul(p, v)

        monkeypatch.setattr(F, "scaled_dot_product_attention", _mock_sdpa, raising=False)
        restored = True

    # Make math ops no-ops on dummy tensors
    monkeypatch.setattr(torch, "matmul", lambda a, b: a, raising=False)
    monkeypatch.setattr(
        attn_mod.F, "scaled_dot_product_attention", lambda *a, **k: a[0], raising=False
    )
    monkeypatch.setattr(torch, "softmax", lambda x, dim=-1: x, raising=False)

    att = Attention(
        num_heads=2,
        head_dim=4,
        scale=1.0,
        num_kv_heads=2,
        use_medical_attention=False,
        attention_impl="flash",
    )
    setattr(att, "training", False)
    q, k, v = _mk_qkv()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = att.forward(q, k, v)
    assert out is not None
    # Expect a warning about flash unavailable
    assert any(
        "flash" in str(x.message).lower() and "unavailable" in str(x.message).lower() for x in w
    )


def test_fallback_sdpa_unavailable_warns_and_uses_manual(monkeypatch):
    # Remove SDPA attribute from F
    F = torch.nn.functional
    had_attr = hasattr(F, "scaled_dot_product_attention")
    if had_attr:
        sdp = getattr(F, "scaled_dot_product_attention")
        monkeypatch.delattr(F, "scaled_dot_product_attention", raising=False)
    monkeypatch.setattr(attn_mod, "FLASH_ATTN_AVAILABLE", False, raising=False)
    # Make math ops no-ops on dummy tensors
    monkeypatch.setattr(torch, "matmul", lambda a, b: a, raising=False)
    monkeypatch.setattr(torch, "softmax", lambda x, dim=-1: x, raising=False)

    try:
        att = Attention(
            num_heads=2,
            head_dim=4,
            scale=1.0,
            num_kv_heads=2,
            use_medical_attention=False,
            attention_impl="sdpa",
        )
        setattr(att, "training", False)
        q, k, v = _mk_qkv()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = att.forward(q, k, v)
        assert out is not None
        assert any(
            "sdpa" in str(x.message).lower() and "unavailable" in str(x.message).lower() for x in w
        )
    finally:
        # Restore SDPA if it originally existed
        if had_attr:
            setattr(F, "scaled_dot_product_attention", sdp)
