from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from medvllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )
        # Store weight_loader as a bound method to avoid mypy issues
        self._weight_loader = self.weight_loader.__get__(self)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        param_data = param.data
        shard_size = int(param_data.size(0))  # Convert to int explicitly
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(
            dim=0, start=int(start_idx), length=shard_size  # Convert to int explicitly
        )
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ) -> None:
        super().__init__(num_embeddings, embedding_dim)
        # Initialize bias with proper type annotation
        self.bias: Optional[nn.Parameter] = None
        if bias:
            self.bias = nn.Parameter(torch.empty(self.num_embeddings_per_partition))
            self._bias_loader = self.weight_loader.__get__(self)
        else:
            self.register_parameter("bias", None)
            self._bias_loader = None

    def forward(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()

        # Handle the case where bias might be None
        bias = self.bias if self.bias is not None else None
        current_logits = F.linear(x, self.weight, bias)

        if self.tp_size <= 1:
            return current_logits

        # For tensor parallelism, handle the gathering of results
        result: Optional[torch.Tensor] = None

        if self.tp_rank == 0:
            # Create a list to hold the gathered tensors
            all_logits: List[torch.Tensor] = [
                torch.empty_like(current_logits) for _ in range(self.tp_size)
            ]

            # Gather the results
            dist.gather(current_logits, all_logits, 0)

            # Concatenate the results
            result = torch.cat(all_logits, dim=-1)
        else:
            # Non-root ranks just need to send their data
            dist.gather(current_logits, None, 0)

        return result
