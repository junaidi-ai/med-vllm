"""Mock implementation of flash_attn for testing purposes."""


class MockFlashAttn:
    @staticmethod
    def flash_attn_varlen_func(*args, **kwargs):
        return None

    @staticmethod
    def flash_attn_with_kvcache(*args, **kwargs):
        return None


# Create mock functions
flash_attn_varlen_func = MockFlashAttn.flash_attn_varlen_func
flash_attn_with_kvcache = MockFlashAttn.flash_attn_with_kvcache
