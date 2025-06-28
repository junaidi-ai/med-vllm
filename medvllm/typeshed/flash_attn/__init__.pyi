from typing import Any, Optional, Tuple

def flash_attn_func(
    q: Any,
    k: Any,
    v: Any,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    return_attn_probs: bool = False,
) -> Tuple[Any, ...]:
    ...

def flash_attn_varlen_func(
    q: Any,
    k: Any,
    v: Any,
    cu_seqlens_q: Any,
    cu_seqlens_k: Any,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    return_attn_probs: bool = False,
) -> Tuple[Any, ...]:
    ...

def flash_attn_with_kvcache(
    q: Any,
    k_cache: Any,
    v_cache: Any,
    k: Any,
    v: Any,
    cache_seqlens: Any,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
) -> Any:
    ...
