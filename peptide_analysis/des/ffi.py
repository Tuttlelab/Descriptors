#!/usr/bin/env python3

"""
ffi.py

Fiber Formation Index (FFI) Analysis

This script calculates the Fiber Formation Index (FFI) for peptide simulations.
It incorporates advanced features such as:

- Multidimensional shape analysis using moments of inertia.
- Detailed alignment analysis using orientation distribution.
- Cross-sectional profiling to assess fiber uniformity.
- Temporal tracking of fiber growth over the simulation time.
- Integration with the Fibrillar Order Parameter (FOP) for internal ordering assessment.

Usage:
    python ffi.py -t topology.gro -x trajectory.xtc [options]

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
    center_and_wrap_trajectory,
    parse_arguments,
)
from util.geometry import (
    compute_moments_of_inertia,
    get_peptide_orientations,
    analyze_orientation_distribution,
    compute_fop,
    cross_sectional_profiling,
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
    plot_fiber_lifetimes,
    plot_number_of_structures_per_frame,
)
from util.logging import (
    setup_logging,
    get_logger,
)

def add_arguments(parser):
    """
    Add FFI-specific arguments to the parser.

    Args:
        parser (argparse.ArgumentParser): Argument parser object.
    """
    parser.add_argument('--min_fiber_size', type=int, default=1000,
                        help='Minimum number of beads to consider a fiber (default: 1000)')
    parser.add_argument('--distance_cutoff', type=float, default=7.0,
                        help='Distance cutoff for clustering in Å (default: 7.0 Å)')
    parser.add_argument('--spatial_weight', type=float, default=1.2,
                        help='Weight for spatial distance in clustering metric (default: 1.2)')
    parser.add_argument('--orientation_weight', type=float, default=1.0,
                        help='Weight for orientation similarity in clustering metric (default: 1.0)')
    parser.add_argument('--clustering_threshold', type=float, default=2.5,
                        help='Threshold for clustering algorithm (default: 2.5)')
    # Add any other FFI-specific arguments as needed

def run(args):
    """
    Main function to run the FFI analysis.

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

    # Center and wrap the trajectory to handle PBC issues
    module_logger.info("Centering and wrapping trajectory...")
    center_and_wrap_trajectory(u, args.selection)

    # Initialize variables for analysis
    fiber_records = defaultdict(list)  # {fiber_id: [frame_numbers]}
    frame_results = []

    # Analyze each frame
    module_logger.info("Analyzing frames for fiber formation...")
    fiber_id_counter = 0
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
            if results.get('is_fiber'):
                fiber_id = f"fiber_{fiber_id_counter}"
                fiber_records[fiber_id].append(frame_number)
                fiber_id_counter += 1
                module_logger.debug(f"Frame {frame_number}: Aggregate classified as fiber (ID: {fiber_id})")
            else:
                module_logger.debug(f"Frame {frame_number}: Aggregate not classified as fiber.")
            frame_results.append(results)

    # Analyze fiber lifetimes
    fiber_lifetimes = analyze_lifetimes(fiber_records)
    module_logger.info(f"Total fibers detected: {len(fiber_lifetimes)}")

    # Save and plot results
    headers = ['frame', 'aggregate_size', 'shape_ratio1', 'shape_ratio2',
               'mean_angle', 'std_angle', 'fop', 'mean_cross_section_area',
               'std_cross_section_area', 'is_fiber']
    save_lifetimes(fiber_lifetimes, args.output, 'fiber')
    save_frame_results(frame_results, args.output, 'ffi', headers)
    plot_fiber_lifetimes(fiber_lifetimes, args.output)
    plot_number_of_structures_per_frame(frame_results, 'is_fiber', args.output)

    module_logger.info("FFI analysis completed successfully.")

    # Clean up temporary files if any
    temp_files = ['temp_topology.gro', 'temp_trajectory.xtc']
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            module_logger.debug(f"Removed temporary file: {temp_file}")

def analyze_aggregate(aggregate_atoms, frame_number, args):
    """
    Analyze a single aggregate for fiber properties.

    Args:
        aggregate_atoms (MDAnalysis.AtomGroup): AtomGroup of the aggregate.
        frame_number (int): Frame number.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        dict: Dictionary containing analysis results for the aggregate.
    """
    results = {}
    positions = aggregate_atoms.positions
    if len(positions) < args.min_fiber_size:
        results['is_fiber'] = False
        logging.debug(f"Frame {frame_number}: Aggregate too small to be a fiber (size={len(positions)}).")
        return results

    # Compute moments of inertia
    shape_ratio1, shape_ratio2, principal_axis = compute_moments_of_inertia(positions)

    # Debug print: shape ratios
    logging.debug(f"Frame {frame_number}: shape_ratio1={shape_ratio1:.3f}, shape_ratio2={shape_ratio2:.3f}")

    # Criteria for shape ratios
    if shape_ratio1 < 1.5 or shape_ratio2 < 1.5:
        results['is_fiber'] = False
        logging.debug(f"Frame {frame_number}: Shape ratios below threshold (shape_ratio1={shape_ratio1:.3f}, shape_ratio2={shape_ratio2:.3f}).")
        return results

    # Compute peptide orientations
    orientations = get_peptide_orientations(aggregate_atoms)
    logging.debug(f"Frame {frame_number}: Number of orientations: {len(orientations)}")
    if len(orientations) == 0:
        results['is_fiber'] = False
        logging.debug(f"Frame {frame_number}: No valid orientations found.")
        return results

    mean_angle, std_angle, angles = analyze_orientation_distribution(orientations, principal_axis)

    # Debug print: orientation distribution
    logging.debug(f"Frame {frame_number}: mean_angle={mean_angle:.3f}, std_angle={std_angle:.3f}")

    # Compute FOP
    fop = compute_fop(orientations, principal_axis)

    # Debug print: Fibrillar Order Parameter (FOP)
    logging.debug(f"Frame {frame_number}: FOP={fop:.3f}")

    # Cross-sectional profiling
    relative_positions = positions - positions.mean(axis=0)
    cross_section_areas = cross_sectional_profiling(relative_positions, principal_axis)
    mean_cross_section_area = np.mean(cross_section_areas)
    std_cross_section_area = np.std(cross_section_areas)

    # Classification criteria
    is_fiber = (
        std_angle < 50.0 and
        (fop > 0.1 or fop < -0.1)
    )

    # Debug print: final classification
    logging.debug(f"Frame {frame_number}: Is fiber={is_fiber} (std_angle={std_angle:.3f}, FOP={fop:.3f})")

    # Store results with explicit type casting
    results['frame'] = frame_number
    results['aggregate_size'] = len(positions)
    results['shape_ratio1'] = shape_ratio1
    results['shape_ratio2'] = shape_ratio2
    results['mean_angle'] = mean_angle
    results['std_angle'] = std_angle
    results['fop'] = fop
    results['mean_cross_section_area'] = mean_cross_section_area
    results['std_cross_section_area'] = std_cross_section_area
    results['is_fiber'] = is_fiber  # Store as boolean

    return results

if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_arguments('Fiber Formation Index (FFI) Analysis', add_arguments)
    # Run the analysis
    run(args)
