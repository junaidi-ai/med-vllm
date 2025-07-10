"""
Logging utilities for the medvllm package.
"""

import logging
import logging.config
import os
from typing import Any, AnyStr, Dict, List, Mapping, Optional, Union, cast


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: The name of the logger. If None, returns the root logger.

    Returns:
        A configured logger instance.
    """
    # Configure logging if not already configured
    if not logging.root.handlers:
        configure_logging()

    return logging.getLogger(name)


def configure_logging(
    level: int = logging.INFO, log_file: Optional[str] = None
) -> None:
    """
    Configure logging for the application.

    Args:
        level: The logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional path to a log file. If None, logs will only
                 go to stderr.
    """
    # Initialize the logging configuration
    log_config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": ("%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "level": level,
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stderr",
            },
        },
        "loggers": {
            "": {  # root logger
                "handlers": ["console"],
                "level": level,
                "propagate": True,
            },
            "medvllm": {
                "handlers": ["console"],
                "level": level,
                "propagate": False,
            },
        },
    }

    # Add file handler if log_file is provided
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # Add file handler configuration
        log_config["handlers"]["file"] = {
            "level": level,
            "class": "logging.FileHandler",
            "filename": log_file,
            "formatter": "standard",
            "mode": "a",  # append to the file if it exists
        }

        # Safely get and update handlers for root logger
        root_logger = log_config["loggers"][""]
        if "handlers" not in root_logger or not isinstance(
            root_logger["handlers"], list
        ):
            root_logger["handlers"] = ["console"]
        root_handlers = cast(List[str], root_logger["handlers"])
        if "file" not in root_handlers:
            root_handlers.append("file")

        # Safely get and update handlers for medvllm logger
        medvllm_logger = log_config["loggers"]["medvllm"]
        if "handlers" not in medvllm_logger or not isinstance(
            medvllm_logger["handlers"], list
        ):
            medvllm_logger["handlers"] = ["console"]
        medvllm_handlers = cast(List[str], medvllm_logger["handlers"])
        if "file" not in medvllm_handlers:
            medvllm_handlers.append("file")

    # Ensure the 'class' key is set for each handler
    if "handlers" in log_config:
        for handler_config in log_config["handlers"].values():
            if "class_" in handler_config:
                handler_config["class"] = handler_config.pop("class_")

    # Apply the configuration
    logging.config.dictConfig(log_config)
