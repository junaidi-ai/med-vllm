"""Mock implementation of flash_attn for testing purposes."""

from typing import Any


class MockFlashAttn:
    @staticmethod
    def flash_attn_varlen_func(*args: Any, **kwargs: Any) -> None:
        """Mock implementation of flash_attn_varlen_func."""
        return None

    @staticmethod
    def flash_attn_with_kvcache(*args: Any, **kwargs: Any) -> None:
        """Mock implementation of flash_attn_with_kvcache."""
        return None


# Create mock functions
flash_attn_varlen_func = MockFlashAttn.flash_attn_varlen_func
flash_attn_with_kvcache = MockFlashAttn.flash_attn_with_kvcache
