#!/usr/bin/env python3

"""
vfi.py

Vesicle Formation Index (VFI) Analysis

This script calculates the Vesicle Formation Index (VFI) for peptide simulations.
It includes features such as:

- Identifying vesicular structures based on geometric criteria.
- Analyzing the size and persistence of vesicles over the simulation time.
- Calculating shape descriptors to characterize vesicle morphology.
- Saving analysis results to CSV files and generating plots.

Usage:
    python vfi.py -t topology.gro -x trajectory.xtc [options]

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
    plot_vesicle_lifetimes,
    plot_number_of_structures_per_frame,
)
from util.logging import (
    setup_logging,
    get_logger,
)

def add_arguments(parser):
    """
    Add VFI-specific arguments to the parser.

    Args:
        parser (argparse.ArgumentParser): Argument parser object.
    """
    parser.add_argument('--min_vesicle_size', type=int, default=5000,
                        help='Minimum number of atoms to consider a vesicle (default: 5000)')
    parser.add_argument('--distance_cutoff', type=float, default=8.0,
                        help='Distance cutoff for clustering in Å (default: 8.0 Å)')
    # Add any other VFI-specific arguments as needed

def run(args):
    """
    Main function to run the VFI analysis.

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
    vesicle_records = defaultdict(list)  # {vesicle_id: [frame_numbers]}
    frame_results = []

    # Analyze each frame
    module_logger.info("Analyzing frames for vesicle formation...")
    vesicle_id_counter = 0
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
            if results.get('is_vesicle'):
                vesicle_id = f"vesicle_{vesicle_id_counter}"
                vesicle_records[vesicle_id].append(frame_number)
                vesicle_id_counter += 1
                module_logger.debug(f"Frame {frame_number}: Aggregate classified as vesicle (ID: {vesicle_id})")
            else:
                module_logger.debug(f"Frame {frame_number}: Aggregate not classified as vesicle.")
            frame_results.append(results)

    # Analyze vesicle lifetimes
    vesicle_lifetimes = analyze_lifetimes(vesicle_records)
    module_logger.info(f"Total vesicles detected: {len(vesicle_lifetimes)}")

    # Save and plot results
    headers = ['frame', 'aggregate_size', 'sphericity', 'hollowness_ratio',
               'asphericity', 'acylindricity', 'is_vesicle']
    save_lifetimes(vesicle_lifetimes, args.output, 'vesicle')
    save_frame_results(frame_results, args.output, 'vfi', headers)
    plot_vesicle_lifetimes(vesicle_lifetimes, args.output)
    plot_number_of_structures_per_frame(frame_results, 'is_vesicle', args.output)

    module_logger.info("VFI analysis completed successfully.")

    # Clean up temporary files if any
    temp_files = ['temp_topology.gro', 'temp_trajectory.xtc']
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            module_logger.debug(f"Removed temporary file: {temp_file}")

def analyze_aggregate(aggregate_atoms, frame_number, args):
    """
    Analyze a single aggregate for vesicle properties.

    Args:
        aggregate_atoms (MDAnalysis.AtomGroup): AtomGroup of the aggregate.
        frame_number (int): Frame number.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        dict: Dictionary containing analysis results for the aggregate.
    """
    results = {}
    positions = aggregate_atoms.positions
    if len(positions) < args.min_vesicle_size:
        results['is_vesicle'] = False
        logging.debug(f"Frame {frame_number}: Aggregate too small to be a vesicle (size={len(positions)}).")
        return results

    # Compute sphericity
    sphericity = compute_sphericity(positions)

    # Compute hollowness ratio
    hollowness_ratio = compute_hollowness_ratio(positions)

    # Compute shape anisotropy
    asphericity, acylindricity = compute_shape_anisotropy(positions)

    # Classification criteria
    is_vesicle = (
        sphericity > 0.7 and
        hollowness_ratio > 0.3 and
        asphericity < 0.2
    )

    # Debug print: final classification
    logging.debug(f"Frame {frame_number}: Is vesicle={is_vesicle} (sphericity={sphericity:.3f}, hollowness_ratio={hollowness_ratio:.3f}, asphericity={asphericity:.3f})")

    # Store results with explicit type casting
    results['frame'] = frame_number
    results['aggregate_size'] = len(positions)
    results['sphericity'] = sphericity
    results['hollowness_ratio'] = hollowness_ratio
    results['asphericity'] = asphericity
    results['acylindricity'] = acylindricity
    results['is_vesicle'] = is_vesicle  # Store as boolean

    return results

if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_arguments('Vesicle Formation Index (VFI) Analysis', add_arguments)
    # Run the analysis
    run(args)
