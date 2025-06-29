from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
from torch import Tensor

from .types import *

if TYPE_CHECKING:
    from .base import ModelRunner

class SamplingManager:
    """Manages token sampling and logits processing."""
    
    def __init__(self, runner: 'ModelRunner') -> None:
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
        logits: TensorT,
        input_ids: Optional[TensorT] = None,
    ) -> TensorT:
        """Process logits before sampling.
        
        Args:
            logits: The raw logits from the model.
            input_ids: The input token IDs (optional).
            
        Returns:
            Processed logits ready for sampling.
        """
        if self.logits_processor is not None:
            return self.logits_processor(logits, input_ids)
        return logits
    
    def sample(
        self,
        logits: 'TensorT',
        seqs: List[SequenceT],
    ) -> List[int]:
        """Sample tokens from logits.
        
        Args:
            logits: The logits tensor of shape (batch_size, seq_len, vocab_size).
            seqs: List of sequences being processed.
            
        Returns:
            A list of sampled token IDs, one for each sequence in the batch.
        """
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"Expected logits to be a torch.Tensor, got {type(logits)}")
            
        # Get the last token logits for each sequence
        last_logits = logits[:, -1, :]
        
        # Process logits if a processor is set
        processed_logits = self.process_logits(last_logits)
        
        # Get temperatures from sequences
        temperatures = [s.sampling_params.temperature for s in seqs]
        
        # Sample tokens using the runner's sampler
        sampled = self.runner.sampler(processed_logits, temperatures)
        
        # Convert to list of integers
        if isinstance(sampled, torch.Tensor):
            token_ids = sampled.tolist()
        elif isinstance(sampled, (list, tuple)):
            token_ids = [int(t) for t in sampled]
        else:
            token_ids = [int(sampled)]
        
        # Validate the output
        if not all(isinstance(x, int) for x in token_ids):
            raise TypeError(
                f"Expected sampler to return integers, got {token_ids}"
            )
        
        if len(token_ids) != len(seqs):
            raise ValueError(
                f"Expected {len(seqs)} samples, got {len(token_ids)}"
            )
        
        return token_ids
    
    def prepare_temperature(self, temperature: Optional[float] = None) -> Optional['TensorT']:
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
            if hasattr(seq, 'sampling_params') and hasattr(seq.sampling_params, 'temperature'):
                temp = seq.sampling_params.temperature
                if temp is not None and temp > 0:
                    temperatures.append(temp)
                else:
                    temperatures.append(1.0)  # Default temperature
            else:
                temperatures.append(1.0)  # Default temperature
        
        return temperatures if any(t != 1.0 for t in temperatures) else None
