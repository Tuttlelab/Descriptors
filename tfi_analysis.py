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

# Constants
RADIAL_THRESHOLD = 12.0             # Å, threshold for radial standard deviation
ANGULAR_UNIFORMITY_THRESHOLD = 0.04 # Threshold for angular uniformity metric
ASPHERICITY_THRESHOLD = 0.5        # Threshold for asphericity in gyration tensor analysis
RATIO_THRESHOLD = 0.3              # Threshold for eigenvalue ratio in shape analysis
MIN_TUBE_SIZE = 50                 # Minimum number of atoms to consider an aggregate as a tube
SEGMENT_LENGTH = 40                # Number of atoms in each segment
STEP_SIZE = 20                     # Step size for overlapping segments
CSV_HEADERS = ['Frame', 'Peptides', 'tube_count', 'total_peptides_in_tubes', 'avg_tube_size']


def parse_arguments():
    parser = argparse.ArgumentParser(description='Tube Formation Index (TFI) Analysis')
    parser.add_argument('-t', '--topology', required=True, help='Topology file (e.g., .gro, .pdb)')
    parser.add_argument('-x', '--trajectory', required=True, help='Trajectory file (e.g., .xtc, .trr)')
    parser.add_argument('-o', '--output', default='tfi_results', help='Output directory for results')
    parser.add_argument('--min_tube_size', type=int, default=MIN_TUBE_SIZE, help='Minimum number of atoms for tube')
    parser.add_argument('--first', type=int, default=0, help='First frame to analyze')
    parser.add_argument('--last', type=int, default=None, help='Last frame to analyze')
    parser.add_argument('--skip', type=int, default=1, help='Process every nth frame')
    args = parser.parse_args()
    return args

def ensure_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
    density_smooth = np.convolve(density, np.ones(5)/5, mode='same')
    maxima = argrelextrema(density_smooth, np.greater)[0]
    minima = argrelextrema(density_smooth, np.less)[0]
    if len(maxima) > 0 and len(minima) > 0:
        shell_peak = density_smooth[maxima[0]]
        core_min = density_smooth[minima[0]]
        if core_min < shell_peak * 0.5:
            return True
    return False

def compute_shape_anisotropy(positions):
    relative_positions = positions - positions.mean(axis=0)
    gyration_tensor = np.dot(relative_positions.T, relative_positions) / len(relative_positions)
    eigenvalues, _ = np.linalg.eigh(gyration_tensor)
    eigenvalues = np.sort(eigenvalues)
    asphericity = 1 - (2 * (eigenvalues[0] + eigenvalues[1]) / (2 * eigenvalues[2]))
    ratio = eigenvalues[0] / eigenvalues[2]
    return asphericity, ratio

def analyze_aggregate(aggregate_atoms, frame_number, peptide_indices, args):
    """Analyze aggregate for tube characteristics with peptide tracking"""
    positions = aggregate_atoms.positions
    n_atoms = len(positions)

    if n_atoms < args.min_tube_size:
        return {'is_tube': False, 'peptides': []}

    # Perform tube analysis
    tube_segment_ratio = segment_based_analysis(positions, SEGMENT_LENGTH, STEP_SIZE)
    radial_std, angular_uniformity, r, theta, z, principal_axis = perform_cylindrical_analysis(positions)
    density, bin_edges = compute_radial_density(r)
    hollow = is_hollow_tube(density, bin_edges)
    asphericity, ratio = compute_shape_anisotropy(positions)

    is_tube = (
        tube_segment_ratio >= 0.5 and
        radial_std < RADIAL_THRESHOLD and
        angular_uniformity > ANGULAR_UNIFORMITY_THRESHOLD and
        hollow and
        asphericity > ASPHERICITY_THRESHOLD and
        ratio < RATIO_THRESHOLD
    )

    return {
        'frame': frame_number,
        'size': n_atoms,
        'radial_std': radial_std,
        'angular_uniformity': angular_uniformity,
        'tube_segment_ratio': tube_segment_ratio,
        'hollow': hollow,
        'asphericity': asphericity,
        'eigenvalue_ratio': ratio,
        'is_tube': is_tube,
        'peptides': [f'PEP{idx+1}' for idx in peptide_indices]
    }

def main():
    args = parse_arguments()
    ensure_output_directory(args.output)
    args.selection = 'protein'  # Hardcoded for now
    print()
    print("Loading trajectory...")
    print()
    u = mda.Universe(args.topology, args.trajectory)
    peptides = u.select_atoms('all')  # Use all atoms as file is pre-processed
    print(f"Loaded {len(peptides)} peptide beads.")
    print()

    frame_records = []

    # Process each frame
    frames = range(args.first, args.last or len(u.trajectory), args.skip)
    for frame_number in frames:
        u.trajectory[frame_number]
        print(f"Processing frame {frame_number}...")
        print()

        # Get aggregates and analyze
        aggregates, peptide_indices = identify_aggregates(u, args.selection)

        # Track tubes and their peptides for this frame
        frame_tubes = []
        frame_peptides = []

        for aggregate, indices in zip(aggregates, peptide_indices):
            results = analyze_aggregate(aggregate, frame_number, indices, args)

            if results['is_tube']:
                frame_tubes.append(results)
                frame_peptides.extend(results['peptides'])

        # Create frame record
        tube_count = len(frame_tubes)
        print(f"Number of tubes found in frame {frame_number}: {tube_count}")
        print()
        total_peptides = sum(t['size'] for t in frame_tubes)
        avg_tube_size = total_peptides / tube_count if tube_count > 0 else 0

        frame_record = {
            'Frame': frame_number,
            'Peptides': str(sorted(frame_peptides)),
            'tube_count': tube_count,
            'total_peptides_in_tubes': total_peptides,
            'avg_tube_size': avg_tube_size
        }
        frame_records.append(frame_record)

    # Save results
    save_frame_results(frame_records, args.output)
    print("TFI analysis completed successfully.")
    print()

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
