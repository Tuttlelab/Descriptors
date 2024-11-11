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
from collections import defaultdict

# Import utility functions from the util package
from util.io import (
    ensure_output_directory,
    load_and_crop_trajectory,
    parse_arguments,
)
from util.clustering import (
    identify_aggregates,
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

def add_arguments(parser):
    """
    Add ADI-specific arguments to the parser.

    Args:
        parser (argparse.ArgumentParser): Argument parser object.
    """
    parser.add_argument('-c', '--cutoff', type=float, default=6.0, help='Distance cutoff for clustering in Å (default: 6.0 Å)')
    parser.add_argument('-p', '--persistence', type=int, default=5, help='Minimum persistence (in frames) for aggregates (default: 5 frames)')
    # Add any other ADI-specific arguments as needed

def run(args):
    """
    Main function to run the ADI analysis.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # Ensure the output directory exists
    ensure_output_directory(args.output)

    # Initialize logging
    logger = setup_logging(args.output)
    module_logger = get_logger(__name__)

    # Load and crop the trajectory
    module_logger.info("Loading and processing trajectory...")
    u = load_and_crop_trajectory(
        args.topology,
        args.trajectory,
        args.first,
        args.last,
        args.skip,
        args.selection
    )
    n_frames = len(u.trajectory)
    module_logger.info(f"Total frames to analyze: {n_frames}")

    # Initialize variables for analysis
    persistent_aggregates = defaultdict(list)  # {aggregate_id: [frame_numbers]}
    cluster_size_distribution = []  # List to store cluster sizes per frame
    aggregate_id_counter = 0

    # Analyze each frame
    module_logger.info("Analyzing frames for aggregates...")
    for frame_number, ts in enumerate(u.trajectory):
        module_logger.debug(f"Processing frame {frame_number + 1}/{n_frames}...")
        peptides = u.select_atoms(args.selection)
        positions = peptides.positions

        # Identify aggregates
        aggregates = identify_aggregates(positions, args.cutoff)
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
    save_cluster_size_distribution(cluster_size_distribution, args.output)
    save_persistent_aggregates(filtered_aggregates, args.output)
    plot_cluster_size_distribution(cluster_size_distribution, args.output)
    plot_persistent_aggregates(filtered_aggregates, args.output)

    module_logger.info("ADI analysis completed successfully.")

if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_arguments('Aggregate Detection Index (ADI) Analysis', add_arguments)
    # Run the analysis
    run(args)
