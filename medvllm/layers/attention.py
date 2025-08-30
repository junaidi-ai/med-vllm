import warnings

# Make PyTorch and Triton imports optional
HAS_TORCH = False
HAS_TRITON = False
FLASH_ATTN_AVAILABLE = False

# Try to import PyTorch
try:
    import torch
    from torch import nn
    import torch.nn.functional as F

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
        """Multi-head attention with medical domain optimizations.

        Features:
        - Efficient KV caching for clinical text patterns
        - Support for varying sequence lengths in medical notes
        - Optimized attention computation for medical entities
        - Integration with medical model adapters
        """

        def __init__(
            self,
            num_heads: int,
            head_dim: int,
            scale: float,
            num_kv_heads: int,
            max_sequence_length: int = 4096,
            use_medical_attention: bool = True,
            **kwargs,
        ):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = head_dim
            self.scale = scale
            self.num_kv_heads = num_kv_heads
            self.max_sequence_length = max_sequence_length
            self.use_medical_attention = use_medical_attention
            # Allow caller to hint the implementation: None | 'flash' | 'sdpa' | 'manual'
            self.attention_impl = kwargs.get("attention_impl", None)

            # Initialize KV cache
            self.k_cache = None
            self.v_cache = None
            self.cache_enabled = False
            self.context = get_context()

            # Medical attention specific parameters
            self.medical_attention_mask = None
            self.attention_window = kwargs.get("attention_window", 512)

            # Initialize parameters for medical attention
            if self.use_medical_attention:
                self._init_medical_attention()

        def _init_medical_attention(self):
            """Initialize medical attention specific components."""
            # Initialize medical attention mask if needed
            if self.medical_attention_mask is None:
                self.medical_attention_mask = torch.ones(
                    (1, 1, self.max_sequence_length, self.max_sequence_length),
                    dtype=torch.bool,
                    device=self.context.device if hasattr(self.context, "device") else "cuda",
                )

                # Create sliding window attention mask
                if self.attention_window < self.max_sequence_length:
                    self.medical_attention_mask = torch.triu(
                        torch.ones_like(self.medical_attention_mask),
                        diagonal=-self.attention_window,
                    ) & torch.tril(
                        torch.ones_like(self.medical_attention_mask),
                        diagonal=self.attention_window,
                    )

        def _reshape_output(self, output):
            return output.reshape(-1, self.num_heads * self.head_dim)

        def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
            # Reshape tensors for attention computation
            batch_size, seq_len, _ = q.shape
            q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)

            # KV cache handling
            if self.cache_enabled and self.k_cache is not None and self.v_cache is not None:
                # Update cache with new keys and values
                if self.k_cache.size(2) < seq_len:
                    # Resize cache if needed for medical texts with long sequences
                    new_cache_size = max(seq_len, self.k_cache.size(2) * 2)
                    new_k_cache = torch.zeros_like(
                        self.k_cache,
                        device=self.k_cache.device,
                        dtype=self.k_cache.dtype,
                    )
                    new_v_cache = torch.zeros_like(
                        self.v_cache,
                        device=self.v_cache.device,
                        dtype=self.v_cache.dtype,
                    )
                    new_k_cache[:, :, : self.k_cache.size(2), :] = self.k_cache
                    new_v_cache[:, :, : self.v_cache.size(2), :] = self.v_cache
                    self.k_cache = new_k_cache
                    self.v_cache = new_v_cache

                # Update cache with new keys and values
                self.k_cache[:, :, :seq_len, :] = k
                self.v_cache[:, :, :seq_len, :] = v
                k, v = self.k_cache[:, :, :seq_len, :], self.v_cache[:, :, :seq_len, :]

            # Determine availability
            sdpa_available = hasattr(F, "scaled_dot_product_attention")
            using_cuda = q.is_cuda and (torch.cuda.is_available())
            flash_available_rt = FLASH_ATTN_AVAILABLE and using_cuda

            # Decide implementation with validation and warnings
            requested = (self.attention_impl or "auto").lower()
            chosen = None  # type: Optional[str]

            if requested == "flash":
                if flash_available_rt:
                    chosen = "flash"
                else:
                    warnings.warn(
                        "Attention requested 'flash' but Flash Attention is unavailable (missing package or CUDA/device). "
                        "Falling back to SDPA/manual."
                    )
                    chosen = "sdpa" if sdpa_available else "manual"
            elif requested == "sdpa":
                if sdpa_available:
                    chosen = "sdpa"
                else:
                    warnings.warn(
                        "Attention requested 'sdpa' but PyTorch SDPA is unavailable on this runtime. Falling back to manual/flash."
                    )
                    chosen = "flash" if flash_available_rt else "manual"
            elif requested == "manual":
                chosen = "manual"
            else:  # auto
                if flash_available_rt:
                    chosen = "flash"
                elif sdpa_available:
                    chosen = "sdpa"
                else:
                    chosen = "manual"

            # Execute chosen implementation
            output = None
            if not self.use_medical_attention and chosen == "sdpa":
                try:
                    output = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=None,
                        dropout_p=0.0,
                        is_causal=True,
                        scale=self.scale,
                    )
                except Exception:
                    output = None
                    warnings.warn("SDPA execution failed; falling back to manual.")
            elif not self.use_medical_attention and chosen == "flash" and flash_available_rt:
                if self.context.is_prefill:
                    output = flash_attn_varlen_func(
                        q,
                        k,
                        v,
                        cu_seqlens_q=self.context.cu_seqlens,
                        cu_seqlens_k=self.context.cu_seqlens,
                        max_seqlen_q=self.context.max_seqlen_q,
                        max_seqlen_k=self.context.max_seqlen_k,
                        causal=True,
                    )
                else:
                    output = flash_attn_with_kvcache(
                        q,
                        k,
                        v,
                        k_cache=self.k_cache,
                        v_cache=self.v_cache,
                        cache_seqlens=self.context.cache_seqlens,
                        softmax_scale=self.scale,
                        causal=True,
                    )
            if 'output' not in locals() or output is None:
                # Use manual attention with medical optimizations
                # Calculate attention scores with medical scaling
                attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

                # Apply medical attention mask if enabled
                if self.use_medical_attention and self.medical_attention_mask is not None:
                    mask = self.medical_attention_mask[:, :, : q.size(2), : k.size(2)].to(q.device)
                    attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
                # Apply regular attention mask if provided
                elif hasattr(self.context, "attention_mask"):
                    attn_scores = attn_scores + self.context.attention_mask.to(q.device)

                # Compute attention weights with numerical stability
                attn_weights = torch.softmax(attn_scores, dim=-1)

                # Optional: Apply medical attention dropout if needed
                if self.training and hasattr(self, "attention_dropout"):
                    attn_weights = self.attention_dropout(attn_weights)

                output = torch.matmul(attn_weights, v)

            # Reshape output to [batch_size, seq_len, hidden_size]
            output = (
                output.transpose(1, 2)
                .contiguous()
                .view(batch_size, -1, self.num_heads * self.head_dim)
            )
            return output
