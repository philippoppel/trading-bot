"""
Logging configuration using loguru.
"""

import sys
from pathlib import Path

from loguru import logger

from src.config.settings import get_settings


def setup_logger(name: str = "trading_bot") -> None:
    """
    Configure the loguru logger with settings from config.

    Args:
        name: Logger name for the log file
    """
    settings = get_settings()
    log_config = settings.logging

    # Remove default handler
    logger.remove()

    # Add console handler
    logger.add(
        sys.stderr,
        format=log_config.format,
        level=log_config.level,
        colorize=True
    )

    # Add file handler
    log_dir = settings.get_log_dir()
    log_file = log_dir / f"{name}.log"

    logger.add(
        log_file,
        format=log_config.format,
        level=log_config.level,
        rotation=log_config.rotation,
        retention=log_config.retention,
        compression="zip"
    )

    logger.info(f"Logger initialized. Log file: {log_file}")


def get_logger():
    """
    Get the configured logger instance.

    Returns:
        loguru logger instance
    """
    return logger
