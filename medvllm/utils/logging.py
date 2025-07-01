"""
Logging utilities for the medvllm package.
"""

import logging
import logging.config
import os
from typing import Optional


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
    log_config = {
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

        log_config["handlers"]["file"] = {
            "level": level,
            "class": "logging.FileHandler",
            "filename": log_file,
            "formatter": "standard",
            "mode": "a",  # append to the file if it exists
        }

        # Add file handler to root logger and medvllm logger
        log_config["loggers"][""]["handlers"].append("file")
        log_config["loggers"]["medvllm"]["handlers"].append("file")

    # Apply the configuration
    logging.config.dictConfig(log_config)
