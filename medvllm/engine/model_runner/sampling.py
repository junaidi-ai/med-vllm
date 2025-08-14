from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Optional,
    Union,
)

import torch
from torch import Tensor

from .types import *

if TYPE_CHECKING:
    from .base import ModelRunner
    from .types import _Tensor
else:
    # Create a dummy _Tensor type for runtime that's compatible with torch.Tensor
    _Tensor = Tensor


class SamplingManager:
    """Manages token sampling and logits processing."""

    def __init__(self, runner: "ModelRunner") -> None:
        """Initialize the sampling manager.

        Args:
            runner: The parent ModelRunner instance.
        """
        self.runner = runner
        self.logits_processor: Optional[LogitsProcessorT] = None

    def set_logits_processor(self, processor: LogitsProcessorT) -> None:
        """Set the logits processor function.

        Args:
            processor: A callable that processes logits before sampling.
        """
        if not callable(processor):
            raise TypeError("Logits processor must be callable")
        self.logits_processor = processor

    def process_logits(
        self,
        logits: Union[torch.Tensor, "_Tensor"],
        input_ids: Optional[Union[torch.Tensor, "_Tensor"]] = None,
    ) -> torch.Tensor:
        # Import _Tensor here to avoid circular imports
        from .types import _Tensor

        # Import the actual _Tensor type for runtime checks
        if not TYPE_CHECKING:
            _Tensor = torch.Tensor  # type: ignore[misc, assignment]
        """Process logits before sampling.

        Args:
            logits: The raw logits from the model.
            input_ids: The input token IDs (optional).

        Returns:
            Processed logits ready for sampling.
        """
        # Convert input to torch.Tensor if needed
        if not isinstance(logits, (torch.Tensor, _Tensor)):
            logits_tensor = torch.tensor(logits, device=self.runner.device)
        else:
            logits_tensor = logits

        if self.logits_processor is None:
            return logits_tensor  # type: ignore[return-value]

        # Prepare input_ids for the processor
        input_ids_tensor: Optional[Union[torch.Tensor, _Tensor]] = None
        if input_ids is not None:
            if not isinstance(input_ids, (torch.Tensor, _Tensor)):
                input_ids_tensor = torch.tensor(input_ids, device=self.runner.device)
            else:
                input_ids_tensor = input_ids

        if self.logits_processor is None:
            return logits_tensor  # type: ignore[return-value]

        # Convert tensors to the expected _Tensor type if needed
        def to_tensort(t: Union[torch.Tensor, _Tensor]) -> _Tensor:
            if isinstance(t, _Tensor) or not hasattr(_Tensor, "_as_tensor"):
                return t  # type: ignore[return-value]
            return _Tensor(t)  # type: ignore[call-arg]

        # Process the logits with the processor
        if input_ids_tensor is not None:
            result = self.logits_processor(to_tensort(logits_tensor), to_tensort(input_ids_tensor))
        else:
            # For the None case, we need to handle it specially since we can't convert None to _Tensor
            result = self.logits_processor(to_tensort(logits_tensor), None)  # type: ignore[call-arg]

        # Ensure the result is a tensor of the correct type
        if not isinstance(result, torch.Tensor):
            return torch.tensor(result, device=logits_tensor.device, dtype=logits_tensor.dtype)
        return result

    def sample(
        self,
        logits: Union[torch.Tensor, "TensorT"],
        seqs: List[Any],  # Using Any to avoid circular imports with SequenceT
    ) -> List[int]:
        """Sample tokens from logits.

        Args:
            logits: The logits tensor of shape (batch_size, seq_len, vocab_size).
            seqs: List of sequences being processed.

        Returns:
            A list of sampled token IDs, one for each sequence in the batch.
        """
        # Ensure logits is a tensor on the correct device
        if not isinstance(logits, torch.Tensor):
            logits_tensor = torch.tensor(logits, device=self.runner.device)
        else:
            logits_tensor = logits

        # Get the last token logits for each sequence
        last_logits = logits_tensor[:, -1, :]

        # Process logits if a processor is set
        processed_logits = self.process_logits(last_logits)

        # Get temperatures from sequences with proper fallback
        temperatures: List[float] = []
        for seq in seqs:
            if hasattr(seq, "sampling_params") and hasattr(seq.sampling_params, "temperature"):
                temp = seq.sampling_params.temperature
                temperatures.append(float(temp) if temp is not None else 1.0)
            else:
                temperatures.append(1.0)  # Default temperature

        # Sample tokens using the runner's sampler
        sampled = self.runner.sampler(processed_logits, temperatures)

        # Convert to list of integers
        token_ids: List[int] = []
        if isinstance(sampled, torch.Tensor):
            token_ids = sampled.tolist()
        elif isinstance(sampled, (list, tuple)):
            token_ids = [int(t) for t in sampled]
        else:
            token_ids = [int(sampled)]

        # Validate the output
        if not all(isinstance(x, int) for x in token_ids):
            raise TypeError(f"Expected sampler to return integers, got {token_ids}")

        if len(token_ids) != len(seqs):
            raise ValueError(f"Expected {len(seqs)} samples, got {len(token_ids)}")

        return token_ids

    def prepare_temperature(self, temperature: Optional[float] = None) -> Optional[torch.Tensor]:
        """Prepare temperature tensor for sampling.

        Args:
            temperature: Temperature value for sampling. If None, no temperature scaling is applied.

        Returns:
            A tensor containing the temperature value on the correct device, or None.
        """
        if temperature is not None:
            return torch.tensor(temperature, device=self.runner.device, dtype=torch.float32)
        return None

    def prepare_temperatures(
        self,
        seqs: List[SequenceT],
    ) -> Optional[List[float]]:
        """Prepare temperatures for sampling.

        Args:
            seqs: List of sequences to sample from.

        Returns:
            List of temperatures or None if not applicable.
        """
        if not seqs:
            return None

        temperatures = []
        for seq in seqs:
            if hasattr(seq, "sampling_params") and hasattr(seq.sampling_params, "temperature"):
                temp = seq.sampling_params.temperature
                if temp is not None and temp > 0:
                    temperatures.append(temp)
                else:
                    temperatures.append(1.0)  # Default temperature
            else:
                temperatures.append(1.0)  # Default temperature

        return temperatures if any(t != 1.0 for t in temperatures) else None
