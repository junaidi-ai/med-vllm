"""Configuration parameters for text generation sampling.

This module defines the SamplingParams class which is used to configure various
parameters for controlling the text generation process in the med-vllm system.
"""

from dataclasses import dataclass


@dataclass
class SamplingParams:
    """Parameters for controlling the text generation sampling process.

    Attributes:
        temperature: Controls randomness in the generation. Lower values make the
            model more deterministic, while higher values increase randomness.
            Must be > 0. Defaults to 1.0.
        max_tokens: Maximum number of tokens to generate. Defaults to 64.
        ignore_eos: If True, the model will continue generating tokens after
            encountering an EOS (end-of-sequence) token. Defaults to False.
    """

    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
