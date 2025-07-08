from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.cuda import Stream

from .types import *

if TYPE_CHECKING:
    from .base import ModelRunner


class CUDAGraphManager:
    """Manages CUDA graphs for model execution."""

    def __init__(self, runner: "ModelRunner") -> None:
        """Initialize the CUDA graph manager.

        Args:
            runner: The parent ModelRunner instance.
        """
        self.runner = runner
        self.graphs: CUDAGraphsT = {}
        self.graph_pool: Any = None
        self.graph_bs: List[int] = []
        self.graph_vars: GraphVarsT = {}

        # Initialize CUDA graph pool if available
        self._init_cuda_graph_pool()

    def _init_cuda_graph_pool(self) -> None:
        """Initialize the CUDA graph pool if CUDA is available."""
        if not torch.cuda.is_available() or self.runner.enforce_eager:
            return

        try:
            # Initialize CUDA graph pool
            if hasattr(torch.cuda, "graph_pool_handle"):
                self.graph_pool = torch.cuda.graph_pool_handle()
        except Exception as e:
            print(f"Warning: Failed to initialize CUDA graph pool: {e}")

    def capture_cudagraph(
        self,
        input_ids: TensorT,
        positions: TensorT,
        is_prefill: bool = False,
    ) -> None:
        """Capture a CUDA graph for the given input shape.

        Args:
            input_ids: Input token IDs.
            positions: Position IDs.
            is_prefill: Whether this is the prefill phase.
        """
        if not torch.cuda.is_available() or self.runner.enforce_eager:
            return

        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)

        # Skip if we've already captured a graph for this batch size
        if batch_size in self.graphs:
            return

        # Warmup
        self._warmup_cuda_graph(batch_size, seq_len, is_prefill)

        # Create and capture the graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.graph_pool):
            # Ensure tensors are on the correct device and type
            input_ids_t = input_ids.to(device=self.runner.device)
            positions_t = positions.to(device=self.runner.device)

            # Convert to _Tensor if needed (for type checking)
            if hasattr(input_ids_t, "_as_tensor") and callable(input_ids_t._as_tensor):
                input_ids_t = input_ids_t._as_tensor()
            if hasattr(positions_t, "_as_tensor") and callable(positions_t._as_tensor):
                positions_t = positions_t._as_tensor()

            # Run the model with type ignore since we've handled the conversion
            self.runner._run_model_impl(input_ids_t, positions_t, is_prefill)  # type: ignore[arg-type]

        # Store the graph and its metadata
        self.graphs[batch_size] = graph  # type: ignore[assignment]
        self.graph_bs.append(batch_size)
        self.graph_bs.sort()

    def _warmup_cuda_graph(
        self,
        batch_size: int,
        seq_len: int,
        is_prefill: bool,
        num_warmup: int = 3,
    ) -> None:
        """Warm up CUDA before capturing a graph.

        Args:
            batch_size: Batch size for warmup.
            seq_len: Sequence length for warmup.
            is_prefill: Whether this is the prefill phase.
            num_warmup: Number of warmup iterations.
        """
        with torch.no_grad():
            for _ in range(num_warmup):
                # Create dummy inputs
                input_ids = torch.randint(
                    0, 1000, (batch_size, seq_len), device=self.runner.device
                )
                positions = (
                    torch.arange(0, seq_len, device=self.runner.device)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )

                # Ensure tensors are on the correct device and type
                input_ids_t = input_ids.to(device=self.runner.device)
                positions_t = positions.to(device=self.runner.device)

                # Convert to _Tensor if needed (for type checking)
                if hasattr(input_ids_t, "_as_tensor") and callable(
                    input_ids_t._as_tensor
                ):
                    input_ids_t = input_ids_t._as_tensor()
                if hasattr(positions_t, "_as_tensor") and callable(
                    positions_t._as_tensor
                ):
                    positions_t = positions_t._as_tensor()

                # Run the model with type ignore since we've handled the conversion
                self.runner._run_model_impl(input_ids_t, positions_t, is_prefill)  # type: ignore[arg-type]

            # Synchronize to ensure all operations are complete
            torch.cuda.synchronize()

    def clear_graphs(self) -> None:
        """Clear all captured CUDA graphs."""
        self.graphs.clear()
        self.graph_bs.clear()
        self.graph_vars.clear()

    def get_graph_for_batch_size(self, batch_size: int) -> Optional[CUDAGraphT]:
        """Get the smallest graph that can handle the given batch size.

        Args:
            batch_size: The desired batch size.

        Returns:
            A CUDA graph that can handle the batch size, or None if none exists.
        """
        if not self.graph_bs or batch_size > self.graph_bs[-1]:
            return None

        # Find the smallest batch size >= requested batch size
        for bs in self.graph_bs:
            if bs >= batch_size:
                return self.graphs[bs]
        return None
