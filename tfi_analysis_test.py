#!/usr/bin/env python3
"""
tfi_analysis.py

This script calculates the Tube Formation Index (TFI) for peptide simulations.
"""

import warnings
# Remove the import that causes the deprecation warning
# from Bio import BiopythonDeprecationWarning
# Modify the warnings filter to ignore the BiopythonDeprecationWarning
warnings.filterwarnings("ignore", ".*BiopythonDeprecationWarning.*")
warnings.filterwarnings("ignore", category=UserWarning)

import os
import csv
import argparse
import numpy as np
import MDAnalysis as mda
from scipy.spatial.distance import cdist
from scipy.signal import argrelextrema
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Modified Constants for PDB analysis
RADIAL_THRESHOLD = 15.0            # Adjusted for PDB scale
MIN_TUBE_SIZE = 10                 # Adjusted for typical PDB peptide size
SEGMENT_LENGTH = 10                # Adjusted segment length
STEP_SIZE = 10                     # Adjusted step size
ANGULAR_UNIFORMITY_THRESHOLD = 0.5  # Threshold for angular uniformity
ASPHERICITY_THRESHOLD = 0.2        # Lowered threshold for asphericity
CYLINDRICITY_THRESHOLD = 0.6       # New threshold for cylindrical shape
RATIO_THRESHOLD = 0.3              # Threshold for eigenvalue ratio
CSV_HEADERS = ['Frame', 'tube_count', 'total_atoms_in_tubes', 'avg_tube_size']

def parse_arguments():
    parser = argparse.ArgumentParser(description='Tube Formation Index (TFI) Analysis for PDB files')
    parser.add_argument('-t', '--topology', required=True, help='PDB file containing multiple frames')
    parser.add_argument('-x', '--trajectory', required=False, help='Not required for PDB files')
    parser.add_argument('-o', '--output', default='tfi_results', help='Output directory for results')
    parser.add_argument('--min_tube_size', type=int, default=MIN_TUBE_SIZE, help='Minimum number of atoms for tube')
    parser.add_argument('--first', type=int, default=0, help='First frame to analyze')
    parser.add_argument('--last', type=int, default=None, help='Last frame to analyze')
    parser.add_argument('--skip', type=int, default=1, help='Process every nth frame')
    args = parser.parse_args()

    if not args.topology.endswith('.pdb'):
        raise ValueError("This script is modified to work with PDB files only")

    return args

def ensure_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def setup_logging(output_dir):
    """Setup logging configuration"""
    timestamp = datetime.now().strftime("%m%d_%H%M")
    log_file = os.path.join(output_dir, f'tfi_analysis_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Keep console output as well
        ]
    )
    return log_file

def identify_aggregates(universe, selection_string):
    """Modified to return both aggregates and their peptide indices"""
    selection = universe.select_atoms(selection_string)
    positions = selection.positions

    if len(positions) == 0:
        return [], []

    # Perform clustering
    linkage_matrix = linkage(positions, method='single', metric='euclidean')
    labels = fcluster(linkage_matrix, t=6.0, criterion='distance') - 1

    # Group atoms by cluster
    aggregates = []
    aggregate_indices = []
    for cluster_id in np.unique(labels):
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) >= MIN_TUBE_SIZE:
            ag_atoms = selection.atoms[cluster_indices]
            aggregates.append(ag_atoms)
            aggregate_indices.append(cluster_indices)

    return aggregates, aggregate_indices

def connected_components(adjacency_matrix):
    n_nodes = adjacency_matrix.shape[0]
    visited = np.zeros(n_nodes, dtype=bool)
    labels = np.zeros(n_nodes, dtype=int) - 1
    label = 0
    for node in range(n_nodes):
        if not visited[node]:
            stack = [node]
            while stack:
                current = stack.pop()
                if not visited[current]:
                    visited[current] = True
                    labels[current] = label
                    neighbors = np.where(adjacency_matrix[current])[0]
                    stack.extend(neighbors)
            label += 1
    return labels, label

def perform_cylindrical_analysis(positions):
    positions_mean = positions.mean(axis=0)
    centered_positions = positions - positions_mean
    covariance_matrix = np.cov(centered_positions.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    principal_axis = eigenvectors[:, -1]

    projections = centered_positions - np.outer(np.dot(centered_positions, principal_axis), principal_axis)

    r = np.linalg.norm(projections, axis=1)
    theta = np.arctan2(projections[:, 1], projections[:, 0])
    z = np.dot(centered_positions, principal_axis)

    radial_std = np.std(r)
    angular_uniformity = compute_angular_uniformity(theta)

    return radial_std, angular_uniformity, r, theta, z, principal_axis

def compute_angular_uniformity(theta):
    histogram, _ = np.histogram(theta, bins=36, range=(-np.pi, np.pi))
    histogram_normalized = histogram / np.sum(histogram)
    uniformity = -np.sum(histogram_normalized * np.log(histogram_normalized + 1e-8))
    max_entropy = np.log(len(histogram))
    angular_uniformity = 1 - (uniformity / max_entropy)
    return angular_uniformity

def segment_based_analysis(positions, segment_length, step_size):
    positions_mean = positions.mean(axis=0)
    centered_positions = positions - positions_mean
    covariance_matrix = np.cov(centered_positions.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    principal_axis = eigenvectors[:, -1]
    z = np.dot(centered_positions, principal_axis)
    ordered_indices = np.argsort(z)
    positions_ordered = positions[ordered_indices]

    num_segments = 0
    tube_like_segments = 0

    for start in range(0, len(positions_ordered) - segment_length + 1, step_size):
        segment_positions = positions_ordered[start:start + segment_length]
        radial_std, angular_uniformity, _, _, _, _ = perform_cylindrical_analysis(segment_positions)
        if radial_std < RADIAL_THRESHOLD and angular_uniformity > ANGULAR_UNIFORMITY_THRESHOLD:
            tube_like_segments += 1
        num_segments += 1

    if num_segments == 0:
        return 0
    return tube_like_segments / num_segments

def compute_radial_density(r, num_bins=50):
    max_radius = r.max()
    bins = np.linspace(0, max_radius, num_bins)
    density, bin_edges = np.histogram(r, bins=bins, density=True)
    return density, bin_edges

def is_hollow_tube(density, bin_edges):
    """Modified to detect partial hollowness"""
    density_smooth = np.convolve(density, np.ones(5)/5, mode='same')
    maxima = argrelextrema(density_smooth, np.greater)[0]
    minima = argrelextrema(density_smooth, np.less)[0]

    if len(maxima) > 0 and len(minima) > 0:
        shell_peak = density_smooth[maxima[0]]
        core_min = density_smooth[minima[0]]
        # More lenient threshold: now only needs to be 20% lower than peak
        if core_min < shell_peak * 0.8:
            return True

    # Check for partial hollowness
    if len(density_smooth) > 5:
        # Check if any section shows significant density decrease
        for i in range(len(density_smooth)-5):
            section = density_smooth[i:i+5]
            if max(section) > 1.2 * min(section):  # 20% difference in any section
                return True

    return False

def compute_shape_anisotropy(positions):
    relative_positions = positions - positions.mean(axis=0)
    gyration_tensor = np.dot(relative_positions.T, relative_positions) / len(relative_positions)
    eigenvalues, _ = np.linalg.eigh(gyration_tensor)
    eigenvalues = np.sort(eigenvalues)

    # Calculate both asphericity and cylindricity
    asphericity = 1 - (2 * (eigenvalues[0] + eigenvalues[1]) / (2 * eigenvalues[2]))
    cylindricity = (eigenvalues[1] - eigenvalues[0]) / eigenvalues[2]  # New metric
    ratio = eigenvalues[0] / eigenvalues[2]

    return asphericity, cylindricity, ratio

def analyze_aggregate(aggregate_atoms, frame_number, peptide_indices, args):
    """Analyze aggregate for tube characteristics"""
    positions = aggregate_atoms.positions
    n_atoms = len(positions)

    if n_atoms < args.min_tube_size:
        return {'is_tube': False, 'size': 0}

    # Perform tube analysis
    tube_segment_ratio = segment_based_analysis(positions, SEGMENT_LENGTH, STEP_SIZE)
    radial_std, angular_uniformity, r, theta, z, principal_axis = perform_cylindrical_analysis(positions)
    density, bin_edges = compute_radial_density(r)
    hollow = is_hollow_tube(density, bin_edges)
    asphericity, cylindricity, ratio = compute_shape_anisotropy(positions)

    # Updated tube criteria including cylindricity
    is_tube = (
        tube_segment_ratio >= 0.4 and
        radial_std < RADIAL_THRESHOLD * 1.2 and
        hollow and
        ratio < RATIO_THRESHOLD * 1.2 and
        asphericity > ASPHERICITY_THRESHOLD and
        cylindricity > CYLINDRICITY_THRESHOLD
    )

    # Add debug printing
    logging.info("\nAggregate Analysis Values:")
    logging.info(f"Number of atoms: {n_atoms}")
    logging.info(f"Tube segment ratio: {tube_segment_ratio:.3f} (threshold ≥ 0.4)")
    logging.info(f"Radial std: {radial_std:.3f} (threshold < {RADIAL_THRESHOLD * 1.2})")
    logging.info(f"Angular uniformity: {angular_uniformity:.3f}")
    logging.info(f"Hollow: {hollow}")
    logging.info(f"Asphericity: {asphericity:.3f} (threshold > {ASPHERICITY_THRESHOLD})")
    logging.info(f"Cylindricity: {cylindricity:.3f} (threshold > {CYLINDRICITY_THRESHOLD})")
    logging.info(f"Eigenvalue ratio: {ratio:.3f} (threshold < {RATIO_THRESHOLD * 1.2})")
    logging.info(f"Is tube: {is_tube}\n")

    return {
        'frame': frame_number,
        'size': n_atoms,
        'radial_std': radial_std,
        'angular_uniformity': angular_uniformity,
        'tube_segment_ratio': tube_segment_ratio,
        'hollow': hollow,
        'asphericity': asphericity,
        'cylindricity': cylindricity,
        'eigenvalue_ratio': ratio,
        'is_tube': is_tube
    }

def main():
    args = parse_arguments()
    ensure_output_directory(args.output)
    log_file = setup_logging(args.output)
    args.selection = 'protein'

    logging.info("\nStarting TFI Analysis")
    logging.info(f"Log file: {log_file}")
    logging.info("\nLoading PDB file...")

    u = mda.Universe(args.topology)
    peptides = u.select_atoms('all')
    logging.info(f"Loaded {len(peptides)} peptide atoms.")

    frame_records = []

    total_frames = len(u.trajectory)
    if total_frames > 10:
        logging.info("Warning: Only processing first 10 frames of PDB file")
        total_frames = 10

    # Process each frame
    for frame_number in range(total_frames):
        u.trajectory[frame_number]
        logging.info(f"\n{'='*50}")
        logging.info(f"Processing frame {frame_number + 1}/{total_frames}")
        logging.info(f"{'='*50}")

        aggregates, peptide_indices = identify_aggregates(u, args.selection)
        logging.info(f"\nFound {len(aggregates)} potential aggregates in frame {frame_number}")

        frame_tubes = []
        total_atoms = 0

        for i, (aggregate, indices) in enumerate(zip(aggregates, peptide_indices)):
            logging.info(f"\nAnalyzing aggregate {i+1}/{len(aggregates)}")
            results = analyze_aggregate(aggregate, frame_number, indices, args)

            if results['is_tube']:
                frame_tubes.append(results)
                total_atoms += results['size']

        tube_count = len(frame_tubes)
        avg_tube_size = total_atoms / tube_count if tube_count > 0 else 0

        logging.info(f"\nFrame {frame_number} Summary:")
        logging.info(f"{'-'*30}")
        logging.info(f"Number of tubes found: {tube_count}")
        logging.info(f"Total atoms in tubes: {total_atoms}")
        logging.info(f"Average tube size: {avg_tube_size:.2f}\n")

        frame_record = {
            'Frame': frame_number,
            'tube_count': tube_count,
            'total_atoms_in_tubes': total_atoms,
            'avg_tube_size': avg_tube_size
        }
        frame_records.append(frame_record)

    save_frame_results(frame_records, args.output)
    logging.info("\nTFI analysis completed successfully.\n")

def save_frame_results(frame_records, output_dir):
    """Save TFI frame results to a CSV file."""
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_file = os.path.join(output_dir, f'tfi_frame_results_{timestamp}.csv')

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for record in frame_records:
            writer.writerow(record)

    logging.info(f"TFI frame results saved to {output_file}")

if __name__ == '__main__':
    main()
