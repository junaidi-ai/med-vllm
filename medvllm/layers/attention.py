import warnings

# Make PyTorch and Triton imports optional
HAS_TORCH = False
HAS_TRITON = False
FLASH_ATTN_AVAILABLE = False

# Try to import PyTorch
try:
    import torch
    from torch import nn

    HAS_TORCH = True
except ImportError:
    warnings.warn(
        "PyTorch is not installed. Some functionality will be limited. "
        "Please install PyTorch to enable all features."
    )

# Try to import Triton if PyTorch is available
if HAS_TORCH:
    try:
        import triton
        import triton.language as tl

        HAS_TRITON = True
    except ImportError:
        warnings.warn(
            "Triton is not installed. Some optimizations will be disabled. "
            "Install with: pip install triton"
        )

    # Try to import flash_attn if PyTorch is available
    try:
        from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

        FLASH_ATTN_AVAILABLE = True
    except ImportError:
        warnings.warn(
            "Flash Attention is not installed. Falling back to a simpler attention implementation. "
            "For better performance, install flash_attn with: pip install flash-attn --no-build-isolation"
        )

# Import context only if PyTorch is available
if HAS_TORCH:
    from medvllm.utils.context import get_context
else:
    # Create a dummy get_context if PyTorch is not available
    def get_context():
        """Dummy context function when PyTorch is not available."""

        class DummyContext:
            def __init__(self):
                self.slot_mapping = None
                self.block_tables = None
                self.is_prefill = False
                self.max_seqlen_q = 0
                self.cu_seqlens_q = None

        return DummyContext()


# Only define Triton kernel if Triton is available
if HAS_TORCH and HAS_TRITON:

    @triton.jit
    def store_kvcache_kernel(
        key_ptr,
        key_stride,
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
    ):
        idx = tl.program_id(0)
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        slot = tl.load(slot_mapping_ptr + idx)
        cache_offsets = slot * D + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)


# Only define store_kvcache if PyTorch is available
if HAS_TORCH:

    def store_kvcache(
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        N, num_heads, head_dim = key.shape
        D = num_heads * head_dim
        assert key.stride(-1) == 1 and value.stride(-1) == 1
        assert key.stride(1) == head_dim and value.stride(1) == head_dim
        assert k_cache.stride(1) == D and v_cache.stride(1) == D
        assert slot_mapping.numel() == N
        store_kvcache_kernel[(N,)](
            key,
            key.stride(0),
            value,
            value.stride(0),
            k_cache,
            v_cache,
            slot_mapping,
            D,
        )


# Only define Attention class if PyTorch is available
if HAS_TORCH:

    class Attention(nn.Module):
        def __init__(
            self,
            num_heads,
            head_dim,
            scale,
            num_kv_heads,
        ):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = head_dim
            self.scale = scale
            self.num_kv_heads = num_kv_heads
            self.k_cache = self.v_cache = torch.tensor([])
            self.context = get_context()

        def _reshape_output(self, output):
            return output.reshape(-1, self.num_heads * self.head_dim)

        def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
            # Reshape tensors for attention computation
            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)

            if FLASH_ATTN_AVAILABLE:
                # Use flash attention implementation if available
                store_kvcache(
                    k, v, self.k_cache, self.v_cache, self.context.slot_mapping
                )

                if self.context.is_prefill:
                    # Prefill stage
                    if self.context.block_tables is not None:  # prefix cache
                        k, v = self.k_cache, self.v_cache
                    output = flash_attn_varlen_func(
                        q,
                        k,
                        v,
                        max_seqlen_q=self.context.max_seqlen_q,
                        cu_seqlens_q=self.context.cu_seqlens_q,
                        max_seqlen_k=self.context.max_seqlen_k,
                        cu_seqlens_k=self.context.cu_seqlens_k,
                        softmax_scale=self.scale,
                        causal=True,
                        block_table=self.context.block_tables,
                    )
                else:  # decode
                    # Decode stage
                    output = flash_attn_with_kvcache(
                        q.unsqueeze(1),
                        self.k_cache,
                        self.v_cache,
                        cache_seqlens=self.context.context_lens,
                        block_table=self.context.block_tables,
                        softmax_scale=self.scale,
                        causal=True,
                    )
            else:
                # Fallback implementation when flash_attn is not available
                # This is a simplified attention implementation for testing purposes
                # It's not optimized for performance

                # Compute attention scores
                attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

                # Apply causal mask if needed
                if self.context.is_prefill and self.context.max_seqlen_q > 1:
                    mask = torch.triu(
                        torch.ones(
                            self.context.max_seqlen_q,
                            self.context.max_seqlen_k,
                            device=q.device,
                        ),
                        diagonal=1,
                    ).bool()
                    attn_scores = attn_scores.masked_fill(mask, float("-inf"))

                # Compute attention weights and output
                attn_weights = torch.softmax(attn_scores, dim=-1)
                output = torch.matmul(attn_weights, v)

            # Reshape output to match expected shape
            return self._reshape_output(output)
