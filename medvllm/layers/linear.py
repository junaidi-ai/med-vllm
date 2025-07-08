from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

T = TypeVar("T", bound="LinearBase")


def divide(numerator: int, denominator: int) -> int:
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ) -> None:
        super().__init__(input_size, output_size)
        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size))
        # Store weight_loader as a bound method to avoid mypy issues
        self._weight_loader = self.weight_loader.__get__(self)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self._bias_loader = self.weight_loader.__get__(self)
        else:
            self.register_parameter("bias", None)
            self._bias_loader = None

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ) -> None:
        super().__init__(input_size, output_size, 0)
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)

        self.weight = nn.Parameter(
            torch.empty(self.output_size_per_partition, self.input_size)
        )
        # Store weight_loader as a bound method to avoid mypy issues
        self._weight_loader = self.weight_loader.__get__(self)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            self._bias_loader = self.weight_loader.__get__(self)
        else:
            self.register_parameter("bias", None)
            self._bias_loader = None

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        param_data = param.data
        if self.tp_dim is None:
            param_data.copy_(loaded_weight)
            return

        tp_dim = int(cast(int, self.tp_dim))  # Ensure tp_dim is an int
        shard_size = int(param_data.size(tp_dim))  # Convert to int explicitly
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(
            dim=tp_dim,
            start=int(start_idx),  # Convert to int explicitly
            length=shard_size,
        )
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ) -> None:
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias=bias)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | None = None,
    ) -> None:
        param_data = param.data
        if loaded_shard_id is None or self.tp_dim is None:
            # Handle the case when called from parent class or no tensor parallelism
            param_data.copy_(loaded_weight)
            return

        tp_dim = int(cast(int, self.tp_dim))  # Ensure tp_dim is an int
        shard_offset = int(
            sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        )  # Convert to int
        shard_size = int(
            self.output_sizes[loaded_shard_id] // self.tp_size
        )  # Convert to int
        param_data = param_data.narrow(
            dim=tp_dim, start=shard_offset, length=shard_size
        )
        loaded_weight = loaded_weight.chunk(chunks=self.tp_size, dim=tp_dim)[
            self.tp_rank
        ]
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ) -> None:
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads
        tp_size = dist.get_world_size()
        self.num_heads = divide(self.total_num_heads, tp_size)
        self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
        input_size = hidden_size
        output_size = (
            self.total_num_heads + 2 * self.total_num_kv_heads
        ) * self.head_size
        super().__init__(input_size, output_size, bias)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ) -> None:
        param_data = param.data
        if loaded_shard_id is None:
            # Handle the case when called from parent class
            param_data.copy_(loaded_weight)
            return

        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = (
                self.num_heads * self.head_size + self.num_kv_heads * self.head_size
            )

        if self.tp_dim is not None:
            tp_dim = int(cast(int, self.tp_dim))  # Ensure tp_dim is an int
            param_data = param_data.narrow(tp_dim, shard_offset, shard_size)
            loaded_weight = loaded_weight.chunk(self.tp_size, dim=tp_dim)[self.tp_rank]
            assert param_data.size() == loaded_weight.size()
            param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ) -> None:
        super().__init__(input_size, output_size, 1)
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size

        self.weight = nn.Parameter(
            torch.empty(self.output_size, self.input_size_per_partition)
        )
        # Store weight_loader as a bound method to avoid mypy issues
        self._weight_loader = self.weight_loader.__get__(self)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self._bias_loader = self.weight_loader.__get__(self)
        else:
            self.register_parameter("bias", None)
            self._bias_loader = None

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        param_data = param.data
        if self.tp_dim is None:
            param_data.copy_(loaded_weight)
            return

        tp_dim = int(cast(int, self.tp_dim))  # Ensure tp_dim is an int
        shard_size = int(param_data.size(tp_dim))  # Convert to int explicitly
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(
            dim=tp_dim,
            start=int(start_idx),  # Convert to int explicitly
            length=shard_size,
        )
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
