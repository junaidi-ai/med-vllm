"""Adapter Manager for Medical Language Models.

This module provides utilities for automatically detecting model types and
creating appropriate adapters for medical language models.
"""

import logging
from typing import Any, Dict, Optional, Union

import torch
from torch import nn
from transformers import AutoConfig, PreTrainedModel

from .adapter import create_medical_adapter
from .adapters.medical_adapter_base import MedicalModelAdapterBase

logger = logging.getLogger(__name__)


class AdapterManager:
    """Manager for creating and configuring medical model adapters."""

    # Model type detection patterns
    MODEL_TYPE_PATTERNS = {
        "biobert": ["biobert", "bio-bert", "bio_bert"],
        "clinicalbert": ["clinicalbert", "clinical-bert", "clinical_bert", "clinbert"],
        "pubmedbert": ["pubmedbert", "pubmed-bert", "pubmed_bert"],
        "bluebert": ["bluebert", "blue-bert", "blue_bert"],
    }

    @classmethod
    def detect_model_type(
        cls, model_name_or_path: str, config: Optional[AutoConfig] = None
    ) -> str:
        """Detect the medical model type from model name or configuration.

        Args:
            model_name_or_path: Path or name of the model
            config: Optional model configuration

        Returns:
            Detected model type (defaults to 'biobert' if unknown)
        """
        model_name_lower = model_name_or_path.lower()

        # Check model name against known patterns
        for model_type, patterns in cls.MODEL_TYPE_PATTERNS.items():
            if any(pattern in model_name_lower for pattern in patterns):
                logger.info(
                    f"Detected model type '{model_type}' from model name: {model_name_or_path}"
                )
                return model_type

        # Check configuration if available
        if config is not None:
            # Check model type in config
            if hasattr(config, "model_type"):
                config_type = config.model_type.lower()
                for model_type, patterns in cls.MODEL_TYPE_PATTERNS.items():
                    if any(pattern in config_type for pattern in patterns):
                        logger.info(
                            f"Detected model type '{model_type}' from config: {config_type}"
                        )
                        return model_type

            # Check architecture name
            if hasattr(config, "architectures") and config.architectures:
                arch_name = config.architectures[0].lower()
                for model_type, patterns in cls.MODEL_TYPE_PATTERNS.items():
                    if any(pattern in arch_name for pattern in patterns):
                        logger.info(
                            f"Detected model type '{model_type}' from architecture: {arch_name}"
                        )
                        return model_type

        # Default to biobert for medical models
        logger.warning(
            f"Could not detect specific model type for {model_name_or_path}, defaulting to 'biobert'"
        )
        return "biobert"

    @classmethod
    def create_adapter(
        cls,
        model: Union[nn.Module, PreTrainedModel],
        model_name_or_path: str,
        adapter_type: Optional[str] = None,
        adapter_config: Optional[Dict[str, Any]] = None,
        hf_config: Optional[AutoConfig] = None,
    ) -> MedicalModelAdapterBase:
        """Create an appropriate medical model adapter.

        Args:
            model: The PyTorch model to adapt
            model_name_or_path: Path or name of the model
            adapter_type: Explicit adapter type (auto-detect if None)
            adapter_config: Configuration for the adapter
            hf_config: Hugging Face model configuration

        Returns:
            Configured medical model adapter
        """
        # Use provided adapter type or auto-detect
        if adapter_type is None:
            adapter_type = cls.detect_model_type(model_name_or_path, hf_config)

        # Use default config if none provided
        if adapter_config is None:
            adapter_config = cls.get_default_adapter_config(adapter_type)

        # Add model-specific configuration
        adapter_config = cls._enhance_adapter_config(adapter_config, model, hf_config)

        logger.info(f"Creating {adapter_type} adapter for model: {model_name_or_path}")

        try:
            adapter = create_medical_adapter(adapter_type, model, adapter_config)
            logger.info(f"Successfully created {adapter_type} adapter")
            return adapter
        except Exception as e:
            logger.error(f"Failed to create adapter: {e}")
            # Fallback to biobert adapter
            logger.info("Falling back to BioBERT adapter")
            return create_medical_adapter("biobert", model, adapter_config)

    @classmethod
    def get_default_adapter_config(cls, adapter_type: str) -> Dict[str, Any]:
        """Get default configuration for a specific adapter type.

        Args:
            adapter_type: Type of the adapter

        Returns:
            Default configuration dictionary
        """
        base_config = {
            "use_kv_cache": True,
            "use_cuda_graphs": False,
            "max_batch_size": 32,
            "max_seq_length": 512,
            # Tensor parallelism settings
            "tensor_parallel_size": 1,
            "rank": 0,
            "world_size": 1,
            # CUDA optimization settings
            "memory_efficient": True,
            "enable_mixed_precision": False,
        }

        # Adapter-specific configurations
        adapter_specific = {
            "biobert": {
                "vocab_size": 30522,
                "hidden_size": 768,
                "num_attention_heads": 12,
                "num_hidden_layers": 12,
            },
            "clinicalbert": {
                "vocab_size": 30522,
                "hidden_size": 768,
                "num_attention_heads": 12,
                "num_hidden_layers": 12,
            },
            "pubmedbert": {
                "vocab_size": 30522,
                "hidden_size": 768,
                "num_attention_heads": 12,
                "num_hidden_layers": 12,
            },
            "bluebert": {
                "vocab_size": 30522,
                "hidden_size": 768,
                "num_attention_heads": 12,
                "num_hidden_layers": 12,
            },
        }

        config = base_config.copy()
        if adapter_type in adapter_specific:
            config.update(adapter_specific[adapter_type])

        return config

    @classmethod
    def _enhance_adapter_config(
        cls,
        config: Dict[str, Any],
        model: Union[nn.Module, PreTrainedModel],
        hf_config: Optional[AutoConfig],
    ) -> Dict[str, Any]:
        """Enhance adapter configuration with model-specific parameters.

        Args:
            config: Base adapter configuration
            model: The PyTorch model
            hf_config: Hugging Face model configuration

        Returns:
            Enhanced configuration dictionary
        """
        enhanced_config = config.copy()

        # Extract parameters from HF config if available
        if hf_config is not None:
            if hasattr(hf_config, "vocab_size"):
                enhanced_config["vocab_size"] = hf_config.vocab_size
            if hasattr(hf_config, "hidden_size"):
                enhanced_config["hidden_size"] = hf_config.hidden_size
            if hasattr(hf_config, "num_attention_heads"):
                enhanced_config["num_attention_heads"] = hf_config.num_attention_heads
            if hasattr(hf_config, "num_hidden_layers"):
                enhanced_config["num_hidden_layers"] = hf_config.num_hidden_layers
            if hasattr(hf_config, "max_position_embeddings"):
                enhanced_config["max_seq_length"] = min(
                    enhanced_config.get("max_seq_length", 512),
                    hf_config.max_position_embeddings,
                )

        # Add device information
        if hasattr(model, "device"):
            enhanced_config["device"] = str(model.device)
        elif next(model.parameters(), None) is not None:
            enhanced_config["device"] = str(next(model.parameters()).device)
        else:
            enhanced_config["device"] = "cpu"

        return enhanced_config


__all__ = ["AdapterManager"]
