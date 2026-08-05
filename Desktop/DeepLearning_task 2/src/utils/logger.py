"""
logger.py

This module provides a centralized logging utility for the entire project.
All project modules should obtain their logger from this file instead of
creating individual loggers.
"""
import logging
from pathlib import Path
def get_logger(logger_name: str) -> logging.Logger:
    """
    Create and return a configured logger.

    Parameters
    ----------
    logger_name : str
        Name of the logger (usually __name__).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    # Create logs directory if it doesn't exist
    log_directory = Path("outputs/logs")
    log_directory.mkdir(parents=True, exist_ok=True)
    log_file = log_directory / "training.log"
    logger = logging.getLogger(logger_name)
    # Prevent duplicate handlers
    if logger.hasHandlers():
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    # Console Output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File Output
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger