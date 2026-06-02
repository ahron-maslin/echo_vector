"""Logging configuration for EchoVector."""

import logging
import sys


def setup_logger(name: str = "echovector", level: int = logging.INFO) -> logging.Logger:
    """Set up and return a logger with the specified name and level.

    Args:
        name: Name of the logger.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logger()
