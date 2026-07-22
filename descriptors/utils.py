"""
descriptors/utils.py

Shared utilities for universe loading, logging, output directory management,
and warning suppressions across shape descriptor modules.
"""

import os
import logging
import warnings
from datetime import datetime
import MDAnalysis as mda

def suppress_warnings():
    """Suppress common non-critical warnings from MDAnalysis and Biopython."""
    warnings.filterwarnings("ignore", message=".*BiopythonDeprecationWarning.*")
    warnings.filterwarnings("ignore", message=".*Bio.Application.*")
    warnings.filterwarnings("ignore", category=UserWarning)

def ensure_output_directory(output_dir):
    """Create directory if it does not exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def load_universe(topology, trajectory):
    """Load MDAnalysis Universe with validation."""
    if not os.path.exists(topology):
        raise FileNotFoundError(f"Topology file not found: {topology}")
    if not os.path.exists(trajectory):
        raise FileNotFoundError(f"Trajectory file not found: {trajectory}")
    return mda.Universe(topology, trajectory)

def setup_logger(module_name, output_dir):
    """Configure module logger with both file and console handlers."""
    ensure_output_directory(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"{module_name}_{timestamp}.log")

    logger = logging.getLogger(f"descriptors.{module_name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger
