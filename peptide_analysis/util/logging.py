# util/logging.py

"""
Logging Utilities for Peptide Analysis

This module provides utility functions for setting up and configuring logging
across different analysis scripts in the peptide_analysis package.

Functions:
- setup_logging: Configure logging with specified parameters.
- get_logger: Retrieve a logger with a given name.
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

def setup_logging(output_dir, log_prefix='analysis'):
    """
    Set up logging to file and console.

    Args:
        output_dir (str): Directory where the log file will be saved.
        log_prefix (str): Prefix for the log file name.
    """
    logger = logging.getLogger()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

     # Check if handlers are already present
    if not getattr(logger, 'handler_set', None):
        # Create handlers# File handler for log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(output_dir, f'{log_prefix}_{timestamp}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Console handler for terminal output
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # Prevent adding handlers multiple times
        setattr(logger, 'handler_set', True)

    return logger

def get_logger(name):
    """
    Get a logger with the given name.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger.
    """
    return logging.getLogger(name)
