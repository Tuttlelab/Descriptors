#!/usr/bin/env python3

"""
tfi.py

Tube Formation Index (TFI) Analysis

This script calculates the Tube Formation Index (TFI) for peptide simulations.
It includes features such as:

- Identifying tubular structures based on geometric criteria.
- Analyzing the size and persistence of tubes over the simulation time.
- Calculating shape descriptors to characterize tube morphology.
- Saving analysis results to CSV files and generating plots.

Usage:
    python tfi.py -t topology.gro -x trajectory.xtc [options]

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
from util.geometry import (
    perform_cylindrical_analysis,
    compute_sphericity,
    compute_hollowness_ratio,
    compute_shape_anisotropy,
)
from util.clustering import (
    identify_aggregates,
)
from util.data import (
    save_lifetimes,
    save_frame_results,
    analyze_lifetimes,
)
from util.visualization import (
    plot_tube_lifetimes,
    plot_number_of_structures_per_frame,
)
from util.logging import (
    setup_logging,
    get_logger,
)

def add_arguments(parser):
    """
    Add TFI-specific arguments to the parser.

    Args:
        parser (argparse.ArgumentParser): Argument parser object.
    """
    parser.add_argument('--min_tube_size', type=int, default=5000,
                        help='Minimum number of atoms to consider a tube (default: 5000)')
    parser.add_argument('--distance_cutoff', type=float, default=8.0,
                        help='Distance cutoff for clustering in Å (default: 8.0 Å)')
    # Add any other TFI-specific arguments as needed

def run(args):
    """
    Main function to run the TFI analysis.

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
    tube_records = defaultdict(list)  # {tube_id: [frame_numbers]}
    frame_results = []

    # Analyze each frame
    module_logger.info("Analyzing frames for tube formation...")
    tube_id_counter = 0
    for frame_number, ts in enumerate(u.trajectory):
        module_logger.debug(f"Processing frame {frame_number + 1}/{n_frames}...")
        peptides = u.select_atoms(args.selection)
        positions = peptides.positions

        # Identify aggregates based on distance cutoff
        aggregates = identify_aggregates(positions, args.distance_cutoff)
        module_logger.debug(f"Frame {frame_number}: Found {len(aggregates)} aggregates")

        for aggregate_indices in aggregates:
            aggregate_atoms = peptides[aggregate_indices]
            results = analyze_aggregate(aggregate_atoms, frame_number, args)
            if results.get('is_tube'):
                tube_id = f"tube_{tube_id_counter}"
                tube_records[tube_id].append(frame_number)
                tube_id_counter += 1
                module_logger.debug(f"Frame {frame_number}: Aggregate classified as tube (ID: {tube_id})")
            else:
                module_logger.debug(f"Frame {frame_number}: Aggregate not classified as tube.")
            frame_results.append(results)

    # Analyze tube lifetimes
    tube_lifetimes = analyze_lifetimes(tube_records)
    module_logger.info(f"Total tubes detected: {len(tube_lifetimes)}")

    # Save and plot results
    headers = ['frame', 'aggregate_size', 'sphericity', 'hollowness_ratio',
               'radial_std', 'angular_uniformity', 'is_tube']
    save_lifetimes(tube_lifetimes, args.output, 'tube')
    save_frame_results(frame_results, args.output, 'tfi', headers)
    plot_tube_lifetimes(tube_lifetimes, args.output)
    plot_number_of_structures_per_frame(frame_results, 'is_tube', args.output)

    module_logger.info("TFI analysis completed successfully.")

    # Clean up temporary files if any
    temp_files = ['temp_topology.gro', 'temp_trajectory.xtc']
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            module_logger.debug(f"Removed temporary file: {temp_file}")

def analyze_aggregate(aggregate_atoms, frame_number, args):
    """
    Analyze a single aggregate for tube properties.

    Args:
        aggregate_atoms (MDAnalysis.AtomGroup): AtomGroup of the aggregate.
        frame_number (int): Frame number.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        dict: Dictionary containing analysis results for the aggregate.
    """
    results = {}
    positions = aggregate_atoms.positions
    if len(positions) < args.min_tube_size:
        results['is_tube'] = False
        logging.debug(f"Frame {frame_number}: Aggregate too small to be a tube (size={len(positions)}).")
        return results

    # Compute sphericity
    sphericity = compute_sphericity(positions)

    # Compute hollowness ratio
    hollowness_ratio = compute_hollowness_ratio(positions)

    # Perform cylindrical analysis
    radial_std, angular_uniformity, _, _, _, principal_axis = perform_cylindrical_analysis(positions)

    # Classification criteria
    is_tube = (
        sphericity < 0.5 and
        hollowness_ratio > 0.2 and
        radial_std < 15.0 and
        angular_uniformity > 0.5
    )

    # Debug print: final classification
    logging.debug(f"Frame {frame_number}: Is tube={is_tube} (sphericity={sphericity:.3f}, hollowness_ratio={hollowness_ratio:.3f}, radial_std={radial_std:.3f}, angular_uniformity={angular_uniformity:.3f})")

    # Store results with explicit type casting
    results['frame'] = frame_number
    results['aggregate_size'] = len(positions)
    results['sphericity'] = sphericity
    results['hollowness_ratio'] = hollowness_ratio
    results['radial_std'] = radial_std
    results['angular_uniformity'] = angular_uniformity
    results['is_tube'] = is_tube  # Store as boolean

    return results

if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_arguments('Tube Formation Index (TFI) Analysis', add_arguments)
    # Run the analysis
    run(args)
