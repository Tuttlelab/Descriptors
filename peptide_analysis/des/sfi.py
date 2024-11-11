#!/usr/bin/env python3

"""
sfi.py

Sheet Formation Index (SFI) Analysis

This script calculates the Sheet Formation Index (SFI) for peptide simulations.
It includes features such as:

- Identifying beta-sheets based on peptide orientations and proximity.
- Analyzing the size and persistence of sheets over the simulation time.
- Calculating shape descriptors to characterize sheet morphology.
- Saving analysis results to CSV files and generating plots.

Usage:
    python sfi.py -t topology.gro -x trajectory.xtc [options]

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
    compute_dipeptide_centroids,
    compute_dipeptide_orientations,
    compute_orientation_matrix,
    perform_pca,
    compute_shape_descriptors,
)
from util.clustering import (
    cluster_peptides,
)
from util.data import (
    save_lifetimes,
    save_frame_results,
    analyze_lifetimes,
)
from util.visualization import (
    plot_sheet_lifetimes,
    plot_number_of_structures_per_frame,
)
from util.logging import (
    setup_logging,
    get_logger,
)

def add_arguments(parser):
    """
    Add SFI-specific arguments to the parser.

    Args:
        parser (argparse.ArgumentParser): Argument parser object.
    """
    parser.add_argument('--min_sheet_size', type=int, default=5,
                        help='Minimum number of peptides to consider a sheet (default: 5)')
    parser.add_argument('--spatial_weight', type=float, default=1.0,
                        help='Weight for spatial distance in clustering metric (default: 1.0)')
    parser.add_argument('--orientation_weight', type=float, default=1.5,
                        help='Weight for orientation similarity in clustering metric (default: 1.5)')
    parser.add_argument('--clustering_threshold', type=float, default=1.5,
                        help='Threshold for clustering algorithm (default: 1.5)')
    parser.add_argument('--peptide_length', type=int, default=4,
                        help='Number of atoms per peptide (default: 4)')
    # Add any other SFI-specific arguments as needed

def run(args):
    """
    Main function to run the SFI analysis.

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
    sheet_records = defaultdict(list)  # {sheet_id: [frame_numbers]}
    frame_results = []

    # Analyze each frame
    module_logger.info("Analyzing frames for sheet formation...")
    sheet_id_counter = 0
    for frame_number, ts in enumerate(u.trajectory):
        module_logger.debug(f"Processing frame {frame_number + 1}/{n_frames}...")
        peptides = u.select_atoms(args.selection)
        if len(peptides) == 0:
            module_logger.warning(f"Frame {frame_number}: No peptides found with selection '{args.selection}'.")
            continue

        # Compute dipeptide centroids and orientations
        positions = compute_dipeptide_centroids(peptides, args.peptide_length)
        orientations = compute_dipeptide_orientations(peptides, args.peptide_length)
        module_logger.debug(f"Frame {frame_number}: Computed positions and orientations for {len(positions)} peptides.")

        # Perform clustering based on spatial proximity and orientation similarity
        labels = cluster_peptides(
            positions,
            orientations,
            args.spatial_weight,
            args.orientation_weight,
            args.clustering_threshold
        )
        unique_labels = set(labels)
        module_logger.debug(f"Frame {frame_number}: Found {len(unique_labels)} clusters.")

        # Analyze each cluster
        for label in unique_labels:
            if label == -1:
                continue  # Ignore noise
            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) < args.min_sheet_size:
                continue  # Ignore small clusters
            cluster_positions = positions[cluster_indices]
            cluster_orientations = orientations[cluster_indices]
            results = analyze_cluster(cluster_positions, cluster_orientations, frame_number, args)
            if results.get('is_sheet'):
                sheet_id = f"sheet_{sheet_id_counter}"
                sheet_records[sheet_id].append(frame_number)
                sheet_id_counter += 1
                module_logger.debug(f"Frame {frame_number}: Cluster classified as sheet (ID: {sheet_id})")
            else:
                module_logger.debug(f"Frame {frame_number}: Cluster not classified as sheet.")
            frame_results.append(results)

    # Analyze sheet lifetimes
    sheet_lifetimes = analyze_lifetimes(sheet_records)
    module_logger.info(f"Total sheets detected: {len(sheet_lifetimes)}")

    # Save and plot results
    headers = ['frame', 'sheet_size', 'asphericity', 'acylindricity', 'mean_angle', 'std_angle', 'is_sheet']
    save_lifetimes(sheet_lifetimes, args.output, 'sheet')
    save_frame_results(frame_results, args.output, 'sfi', headers)
    plot_sheet_lifetimes(sheet_lifetimes, args.output)
    plot_number_of_structures_per_frame(frame_results, 'is_sheet', args.output)

    module_logger.info("SFI analysis completed successfully.")

def analyze_cluster(cluster_positions, cluster_orientations, frame_number, args):
    """
    Analyze a single cluster for sheet properties.

    Args:
        cluster_positions (numpy.ndarray): Positions of peptides in the cluster.
        cluster_orientations (numpy.ndarray): Orientations of peptides in the cluster.
        frame_number (int): Frame number.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        dict: Dictionary containing analysis results for the cluster.
    """
    results = {}
    if len(cluster_positions) < args.min_sheet_size:
        results['is_sheet'] = False
        logging.debug(f"Frame {frame_number}: Cluster too small to be a sheet (size={len(cluster_positions)}).")
        return results

    # Perform PCA to get the normal vector of the best-fit plane
    normal_vector, _, rmsd, _, _ = perform_pca(cluster_positions)
    if normal_vector is None:
        results['is_sheet'] = False
        logging.debug(f"Frame {frame_number}: PCA failed to compute normal vector.")
        return results

    # Compute orientation angles relative to the normal vector
    cos_angles = np.dot(cluster_orientations, normal_vector)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    angles = np.arccos(np.abs(cos_angles)) * (180 / np.pi)  # Use absolute to consider both sides
    mean_angle = np.mean(angles)
    std_angle = np.std(angles)

    # Compute shape descriptors
    asphericity, acylindricity = compute_shape_descriptors(cluster_positions)

    # Classification criteria
    is_sheet = (
        rmsd < 5.0 and
        mean_angle < 30.0
    )

    # Debug print: final classification
    logging.debug(f"Frame {frame_number}: Is sheet={is_sheet} (rmsd={rmsd:.3f}, mean_angle={mean_angle:.3f})")

    # Store results with explicit type casting
    results['frame'] = frame_number
    results['sheet_size'] = len(cluster_positions)
    results['asphericity'] = asphericity
    results['acylindricity'] = acylindricity
    results['mean_angle'] = mean_angle
    results['std_angle'] = std_angle
    results['is_sheet'] = is_sheet  # Store as boolean

    return results

if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_arguments('Sheet Formation Index (SFI) Analysis', add_arguments)
    # Run the analysis
    run(args)
