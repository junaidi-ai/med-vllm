from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event as MP_Event
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import torch
from torch import Tensor
from torch.cuda.graphs import CUDAGraph
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from typing_extensions import TypeAlias, TypeGuard

# Import local modules
from medvllm.config import Config
from medvllm.engine.sequence import Sequence
from medvllm.layers.sampler import Sampler
from medvllm.models.qwen3 import Qwen3ForCausalLM

# Type variables
T = TypeVar("T")

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    # Define type aliases with proper imports
    class _Tensor(Tensor):
        """Dummy class for Tensor type hints."""

        pass

    class _CUDAGraph(CUDAGraph):
        """Dummy class for CUDAGraph type hints."""

        pass

    class _SharedMemory(SharedMemory):
        """Dummy class for SharedMemory type hints."""

        pass

    class _MP_Event(MP_Event):
        """Dummy class for MP_Event type hints."""

        pass

    class _PretrainedConfig(Any):
        """Dummy class for PretrainedConfig type hints."""

        pass

    # Type aliases with proper forward references
    TensorT: TypeAlias = _Tensor
    CUDAGraphT: TypeAlias = _CUDAGraph
    SharedMemoryT: TypeAlias = _SharedMemory
    MP_EventT: TypeAlias = _MP_Event
    ConfigT: TypeAlias = Config
    SequenceT: TypeAlias = Sequence
    SamplerT: TypeAlias = Sampler
    Qwen3ForCausalLMT: TypeAlias = Qwen3ForCausalLM
    PretrainedConfigT: TypeAlias = _PretrainedConfig

    # Type aliases for model components
    OptimizerT: TypeAlias = Optimizer
    LRSchedulerT: TypeAlias = LRScheduler
    DataLoaderT: TypeAlias = DataLoader[Any]
    TensorDictT: TypeAlias = Dict[str, "TensorT"]
    BatchT: TypeAlias = Dict[str, Union["TensorT", List[Any]]]
    LogitsProcessorT: TypeAlias = Callable[["TensorT", "TensorT"], "TensorT"]
    PastKeyValuesT: TypeAlias = Optional[Tuple[Tuple["TensorT", ...], ...]]
    CUDAGraphsT: TypeAlias = Dict[int, "CUDAGraphT"]
    GraphVarsT: TypeAlias = Dict[str, "TensorT"]

    # Type hints for instance variables with forward reference
    from medvllm.engine.model_runner import ModelRunner as _ModelRunner

    _ModelRunnerT: TypeAlias = _ModelRunner
else:
    # Runtime type aliases (simplified for performance)
    TensorT: TypeAlias = Any
    CUDAGraphT: TypeAlias = Any
    SharedMemoryT: TypeAlias = Any
    MP_EventT: TypeAlias = Any
    ConfigT: TypeAlias = Any
    SequenceT: TypeAlias = Any
    SamplerT: TypeAlias = Any
    Qwen3ForCausalLMT: TypeAlias = Any
    PretrainedConfigT: TypeAlias = Any
    OptimizerT: TypeAlias = Any
    LRSchedulerT: TypeAlias = Any
    DataLoaderT: TypeAlias = Any
    TensorDictT: TypeAlias = Dict[str, Any]
    BatchT: TypeAlias = Dict[str, Any]
    LogitsProcessorT: TypeAlias = Callable[..., Any]
    PastKeyValuesT: TypeAlias = Any
    CUDAGraphsT: TypeAlias = Dict[int, Any]
    GraphVarsT: TypeAlias = Dict[str, Any]
    _ModelRunnerT: TypeAlias = Any
