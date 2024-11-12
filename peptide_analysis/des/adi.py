#!/usr/bin/env python3

"""
adi.py

Aggregate Detection Index (ADI) Analysis

This script analyzes peptide simulations to detect and characterize aggregates.
It uses persistent contact analysis to identify stable aggregates over time.

Key Features:
- Detects aggregates based on a distance cutoff.
- Tracks persistent aggregates over the simulation time.
- Analyzes cluster size distribution over time.
- Saves results to CSV files and generates plots.

Usage:
    python adi.py -t topology.gro -x trajectory.xtc [options]

"""

import numpy as np
import argparse
import logging
import os
import sys
from collections import defaultdict
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
from Bio import BiopythonDeprecationWarning
warnings.simplefilter('ignore', BiopythonDeprecationWarning)

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import utility functions from the util package
from util.io import (
    ensure_output_directory,
    load_and_crop_trajectory,
    parse_arguments,
)
from util.clustering import (
    identify_aggregates_with_cutoff,
    calculate_adaptive_cutoff,
)
from util.data import (
    save_cluster_size_distribution,
    save_persistent_aggregates,
)
from util.visualization import (
    plot_cluster_size_distribution,
    plot_persistent_aggregates,
)
from util.logging import (
    setup_logging,
    get_logger,
)

# Constants
DEFAULT_CUTOFF = 6.0  # Distance cutoff for clustering in Angstroms
DEFAULT_MIN_PERSISTENCE = 5  # Minimum number of frames a contact must persist
DEFAULT_RDF_RANGE = (4.0, 15.0)  # Range for RDF calculation in Angstroms
DEFAULT_NBINS = 50  # Number of bins for RDF calculation

def add_arguments(parser):
    """
    Add ADI-specific arguments to the parser.

    Args:
        parser (argparse.ArgumentParser): Argument parser object.
    """
    parser.add_argument('-p', '--persistence', type=int, default=DEFAULT_MIN_PERSISTENCE,
                        help=f'Minimum persistence (in frames) for aggregates (default: {DEFAULT_MIN_PERSISTENCE} frames)')
    parser.add_argument('--rdf_range', type=float, nargs=2, default=DEFAULT_RDF_RANGE,
                        help=f'Range for RDF calculation in Angstroms (default: {DEFAULT_RDF_RANGE})')
    parser.add_argument('--nbins', type=int, default=DEFAULT_NBINS,
                        help=f'Number of bins for RDF calculation (default: {DEFAULT_NBINS})')

def run(args):
    """
    Main function to run the ADI analysis.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    import os

    # Set the descriptor name
    descriptor_name = 'adi'

    # Create the descriptor-specific output directory
    output_dir = os.path.join(args.output, descriptor_name)
    ensure_output_directory(output_dir)

    # Initialize logging with a dynamic log filename
    logger = setup_logging(output_dir, log_prefix='adi_analysis')
    logger.info("Starting ADI analysis...")
    module_logger = get_logger('adi_analysis')

    # Load and crop the trajectory
    module_logger.info("Loading and processing trajectory...")
    u, temp_files = load_and_crop_trajectory(
        args.topology,
        args.trajectory,
        args.first,
        args.last,
        args.skip,
        args.selection
    )
    n_frames = len(u.trajectory)
    module_logger.info(f"Total frames to analyze: {n_frames}")

    try:
        # Calculate adaptive cutoff once
        module_logger.info("Calculating adaptive cutoff distance based on RDF...")
        adaptive_cutoff = calculate_adaptive_cutoff(
            universe=u,
            selection_string=args.selection,
            rdf_range=args.rdf_range,
            nbins=args.nbins,
            output_dir=output_dir
        )
        module_logger.info(f"Adaptive cutoff distance determined: {adaptive_cutoff:.2f} Å")

        # Initialize variables for analysis
        persistent_aggregates = defaultdict(list)  # {aggregate_id: [frame_numbers]}
        cluster_size_distribution = []  # List to store cluster sizes per frame

        # Analyze each frame
        module_logger.info("Starting analysis loop over trajectory frames...")
        for ts in tqdm(u.trajectory, desc="Processing frames"):
            frame_number = ts.frame  # Get current frame number
            # Select peptides
            peptides = u.select_atoms(args.selection)
            positions = peptides.positions

            # Identify aggregates using the adaptive cutoff
            aggregates = identify_aggregates_with_cutoff(
                positions=positions,
                distance_cutoff=adaptive_cutoff
            )
            module_logger.debug(f"Frame {frame_number}: Found {len(aggregates)} aggregates")

            # Store cluster sizes for this frame
            cluster_sizes = [len(aggregate) for aggregate in aggregates]
            cluster_size_distribution.append({'frame': frame_number, 'cluster_sizes': cluster_sizes})

            # Update persistent aggregates
            for aggregate in aggregates:
                # Convert indices to a tuple to use as a key
                aggregate_key = tuple(sorted(aggregate))
                persistent_aggregates[aggregate_key].append(frame_number)

        # Filter persistent aggregates based on persistence threshold
        min_persistence = args.persistence
        filtered_aggregates = []
        for aggregate_key, frames in persistent_aggregates.items():
            if len(frames) >= min_persistence:
                filtered_aggregates.append((aggregate_key, frames))
        module_logger.info(f"Found {len(filtered_aggregates)} persistent aggregates with minimum persistence of {min_persistence} frames")

        # Save and plot results
        save_cluster_size_distribution(cluster_size_distribution, output_dir)
        save_persistent_aggregates(filtered_aggregates, output_dir)
        plot_cluster_size_distribution(cluster_size_distribution, output_dir)
        plot_persistent_aggregates(filtered_aggregates, output_dir)
    finally:
        # Clean up temporary files if any
        if temp_files:
            module_logger.info("Cleaning up temporary files...")
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                    module_logger.debug(f"Deleted temporary file: {temp_file}")
                except OSError as e:
                    module_logger.warning(f"Error deleting temporary file {temp_file}: {e}")

if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_arguments('Aggregate Detection Index (ADI) Analysis', add_arguments)
    # Run the analysis
    run(args)
