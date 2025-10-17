"""
Logging utilities for A2A Pipeline
"""

import sys
import logging
from pathlib import Path
from typing import Optional
from loguru import logger
from configs.config import LOGS_DIR

# Remove default loguru handler
logger.remove()

def setup_logger(
    name: str = "a2a_pipeline",
    level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True
) -> None:
    """
    Set up logging for the A2A Pipeline

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
    """
    # Console logging
    if log_to_console:
        logger.add(
            sys.stdout,
            level=level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True
        )

    # File logging
    if log_to_file:
        log_file = LOGS_DIR / f"{name}.log"
        logger.add(
            log_file,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="7 days",
            compression="zip"
        )

        # Separate error log
        error_log_file = LOGS_DIR / f"{name}_errors.log"
        logger.add(
            error_log_file,
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="30 days"
        )

def get_logger(name: Optional[str] = None):
    """Get a logger instance"""
    if name:
        return logger.bind(name=name)
    return logger

def log_model_info(model_name: str, config: dict):
    """Log model configuration information"""
    logger.info(f"Loading model: {model_name}")
    logger.info(f"Model config: {config}")

def log_search_results(query: str, num_results: int, sources: list):
    """Log search results"""
    logger.info(f"Search query: '{query}' returned {num_results} results")
    logger.debug(f"Sources: {sources}")

def log_evaluation_results(dataset: str, metrics: dict):
    """Log evaluation results"""
    logger.info(f"Evaluation results for {dataset}:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value}")

def log_error_with_context(error: Exception, context: str):
    """Log error with additional context"""
    logger.error(f"Error in {context}: {type(error).__name__}: {str(error)}")
    logger.exception("Full traceback:")

# Performance logging
class PerformanceLogger:
    """Context manager for logging performance metrics"""

    def __init__(self, operation: str):
        self.operation = operation
        self.start_time = None

    def __enter__(self):
        import time
        self.start_time = time.time()
        logger.debug(f"Starting {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.time() - self.start_time
        if exc_type is None:
            logger.info(f"Completed {self.operation} in {duration:.2f}s")
        else:
            logger.error(f"Failed {self.operation} after {duration:.2f}s: {exc_val}")

# Initialize default logger
setup_logger()
main_logger = get_logger("a2a_pipeline")