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

def setup_logging(log_dir, log_filename=None, console_level=logging.INFO, file_level=logging.DEBUG):
    """
    Set up logging configuration.

    Args:
        log_dir (str): Directory where the log file will be saved.
        log_filename (str, optional): Name of the log file. If None, a timestamped filename is created.
        console_level (int): Logging level for the console handler.
        file_level (int): Logging level for the file handler.

    Returns:
        logging.Logger: Configured logger object.
    """
    # Ensure the log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a timestamped log filename if not provided
    if log_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"analysis_{timestamp}.log"

    log_path = os.path.join(log_dir, log_filename)

    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set the root logger level to DEBUG

    # Remove any existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_path)

    c_handler.setLevel(console_level)
    f_handler.setLevel(file_level)

    # Create formatters and add them to handlers
    c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    # Log the start of a new analysis session
    logger.info(f"Logging initialized. Log file: {log_path}")

    return logger

def get_logger(name):
    """
    Retrieve a logger with the given name.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Logger object with the specified name.
    """
    return logging.getLogger(name)
