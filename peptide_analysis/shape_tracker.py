#!/usr/bin/env python3

"""
shape_tracker.py

Shape Tracker for Peptide Aggregates

This script tracks the evolution of shapes of peptide aggregates over time.
It identifies aggregates in each frame, classifies them into shapes (e.g., fiber, sheet, tube, vesicle),
and tracks transitions between shapes throughout the simulation.

Features:

- Identifies aggregates based on a distance cutoff.
- Classifies aggregates into different shapes using existing descriptor modules.
- Tracks aggregates over time, recording shape transitions.
- Saves detailed analysis results to CSV files.
- Generates plots to visualize shape evolution and transitions.

Usage:
    python shape_tracker.py -t topology.gro -x trajectory.xtc [options]

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
from util.clustering import (
    identify_aggregates,
    cluster_peptides
)
from util.data import (
    save_frame_results,
)
from util.logging import (
    setup_logging,
    get_logger,
)

# Import analysis functions from the des modules
from des.ffi import analyze_aggregate as ffi_analyze_aggregate
from des.sfi import analyze_cluster as sfi_analyze_cluster
from des.tfi import analyze_aggregate as tfi_analyze_aggregate
from des.vfi import analyze_aggregate as vfi_analyze_aggregate

# Import necessary functions from util.geometry
from util.geometry import compute_dipeptide_centroids, compute_dipeptide_orientations

def add_arguments(parser):
    """
    Add Shape Tracker-specific arguments to the parser.

    Args:
        parser (argparse.ArgumentParser): Argument parser object.
    """
    parser.add_argument('--distance_cutoff', type=float, default=7.0,
                        help='Distance cutoff for clustering in Å (default: 7.0 Å)')
    parser.add_argument('--min_aggregate_size', type=int, default=1000,
                        help='Minimum number of atoms to consider an aggregate (default: 1000)')
    parser.add_argument('--peptide_length', type=int, default=4,
                        help='Number of atoms per peptide (default: 4)')
    # Add any other arguments as needed

def run(args):
    """
    Main function to run the Shape Tracker analysis.

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
    aggregate_records = {}  # {aggregate_id: {'frames': [], 'shapes': []}}
    frame_results = []

    # A mapping to track aggregates across frames
    aggregate_id_counter = 0
    aggregate_mapping = {}  # {current_aggregate_key: aggregate_id}

    # Analyze each frame
    module_logger.info("Analyzing frames for shape tracking...")
    for frame_number, ts in enumerate(u.trajectory):
        module_logger.debug(f"Processing frame {frame_number + 1}/{n_frames}...")
        peptides = u.select_atoms(args.selection)
        positions = peptides.positions

        # Identify aggregates based on distance cutoff
        aggregates = identify_aggregates(positions, args.distance_cutoff)
        module_logger.debug(f"Frame {frame_number}: Found {len(aggregates)} aggregates")

        current_aggregate_keys = []
        for aggregate_indices in aggregates:
            aggregate_atoms = peptides[aggregate_indices]
            aggregate_size = len(aggregate_atoms)
            aggregate_key = tuple(sorted(aggregate_indices))

            # Initialize results dictionary
            results = {
                'frame': frame_number,
                'aggregate_size': aggregate_size,
                'aggregate_id': None,
                'shape': None
            }

            # Assign an aggregate ID
            if aggregate_key in aggregate_mapping:
                aggregate_id = aggregate_mapping[aggregate_key]
            else:
                aggregate_id = f"aggregate_{aggregate_id_counter}"
                aggregate_mapping[aggregate_key] = aggregate_id
                aggregate_id_counter += 1
                aggregate_records[aggregate_id] = {'frames': [], 'shapes': []}

            results['aggregate_id'] = aggregate_id

            # Update the aggregate records
            aggregate_records[aggregate_id]['frames'].append(frame_number)

            # Skip small aggregates
            if aggregate_size < args.min_aggregate_size:
                shape = 'non-aggregate'
                results['shape'] = shape
                aggregate_records[aggregate_id]['shapes'].append(shape)
                frame_results.append(results)
                continue

            # Try classifying the aggregate using existing descriptor modules
            shape = classify_aggregate(aggregate_atoms, aggregate_size, args)
            results['shape'] = shape
            aggregate_records[aggregate_id]['shapes'].append(shape)

            # Add to frame results
            frame_results.append(results)

            current_aggregate_keys.append(aggregate_key)

        # Clean up aggregate_mapping to remove aggregates not present in current frame
        keys_to_remove = [key for key in aggregate_mapping if key not in current_aggregate_keys]
        for key in keys_to_remove:
            del aggregate_mapping[key]

    # Save results
    headers = ['frame', 'aggregate_id', 'aggregate_size', 'shape']
    save_frame_results(frame_results, args.output, 'shape_tracker', headers)

    # Generate plots if needed

    module_logger.info("Shape tracking analysis completed successfully.")

def classify_aggregate(aggregate_atoms, aggregate_size, args):
    """
    Classify the aggregate into one of the defined shape categories.

    Args:
        aggregate_atoms (MDAnalysis.AtomGroup): AtomGroup of the aggregate.
        aggregate_size (int): Size of the aggregate.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        str: Shape classification of the aggregate.
    """
    # Try classifying as vesicle (VFI)
    vfi_results = vfi_analyze_aggregate(aggregate_atoms, 0, args)
    if vfi_results.get('is_vesicle'):
        return 'vesicle'

    # Try classifying as tube (TFI)
    tfi_results = tfi_analyze_aggregate(aggregate_atoms, 0, args)
    if tfi_results.get('is_tube'):
        return 'tube'

    # Try classifying as fiber (FFI)
    ffi_results = ffi_analyze_aggregate(aggregate_atoms, 0, args)
    if ffi_results.get('is_fiber'):
        return 'fiber'

    # For sheet classification, need positions and orientations
    positions = aggregate_atoms.positions
    peptide_length = args.peptide_length

    # Compute dipeptide centroids and orientations
    centroids = compute_dipeptide_centroids(aggregate_atoms, peptide_length)
    orientations = compute_dipeptide_orientations(aggregate_atoms, peptide_length)

    # Cluster peptides (SFI)
    labels = cluster_peptides(
        centroids,
        orientations,
        spatial_weight=1.0,
        orientation_weight=1.5,
        clustering_threshold=1.5
    )
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:
            continue  # Ignore noise
        cluster_indices = np.where(labels == label)[0]
        if len(cluster_indices) < args.min_aggregate_size:
            continue  # Ignore small clusters
        cluster_positions = centroids[cluster_indices]
        cluster_orientations = orientations[cluster_indices]
        sfi_results = sfi_analyze_cluster(cluster_positions, cluster_orientations, 0, args)
        if sfi_results.get('is_sheet'):
            return 'sheet'

    # If none of the shapes matched, classify as irregular
    return 'irregular'

if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_arguments('Shape Tracker Analysis', add_arguments)
    # Run the analysis
    run(args)
