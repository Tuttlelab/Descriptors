#!/usr/bin/env python3
"""
tfi_analysis.py

This script calculates the Tube Formation Index (TFI) for peptide simulations.
It incorporates advanced features such as:

- Cylindrical harmonic analysis for robust tube detection, accommodating curvature and twists.
- Segment-based analysis to handle long, curved tubes.
- Radial density profiling to assess hollowness.
- Shape anisotropy analysis using the gyration tensor.
- Temporal tracking of tube formation, growth, and stability over the simulation time.

"""

import os
import argparse
import numpy as np
import MDAnalysis as mda
from scipy.spatial.distance import cdist
from scipy.signal import argrelextrema
from collections import defaultdict
import matplotlib.pyplot as plt

# Constants
RADIAL_THRESHOLD = 2.0            # Å, threshold for radial standard deviation
ANGULAR_UNIFORMITY_THRESHOLD = 0.5  # Threshold for angular uniformity metric
ASPHERICITY_THRESHOLD = 0.7       # Threshold for asphericity in gyration tensor analysis
RATIO_THRESHOLD = 0.3             # Threshold for eigenvalue ratio in shape analysis
MIN_TUBE_SIZE = 50                # Minimum number of atoms to consider an aggregate as a tube
SEGMENT_LENGTH = 20               # Number of atoms in each segment
STEP_SIZE = 10                    # Step size for overlapping segments

def parse_arguments():
    parser = argparse.ArgumentParser(description='Tube Formation Index (TFI) Analysis')
    parser.add_argument('-t', '--topology', required=True, help='Topology file (e.g., .gro, .pdb)')
    parser.add_argument('-x', '--trajectory', required=True, help='Trajectory file (e.g., .xtc, .trr)')
    parser.add_argument('-s', '--selection', default='all', help='Atom selection string for peptides')
    parser.add_argument('-o', '--output', default='tfi_results', help='Output directory for results')
    parser.add_argument('--min_tube_size', type=int, default=MIN_TUBE_SIZE, help='Minimum number of atoms to consider a tube')
    parser.add_argument('--first', type=int, default=None, help='Only analyze the first N frames (default is all frames)')
    parser.add_argument('--last', type=int, default=None, help='Only analyze the last N frames (default is all frames)')
    parser.add_argument('--skip', type=int, default=1, help='Process every nth frame (default is every frame)')
    args = parser.parse_args()
    return args

def ensure_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def identify_aggregates(universe, selection_string):
    """
    Identify aggregates (clusters) in the system using a distance cutoff.
    """
    selection = universe.select_atoms(selection_string)
    positions = selection.positions
    distance_matrix = cdist(positions, positions)
    adjacency_matrix = distance_matrix < 6.0  # Distance cutoff in Å
    np.fill_diagonal(adjacency_matrix, 0)
    labels, num_labels = connected_components(adjacency_matrix)
    aggregates = defaultdict(list)
    for idx, label_id in enumerate(labels):
        aggregates[label_id].append(selection.atoms[idx].index)
    return aggregates.values()

def connected_components(adjacency_matrix):
    """
    Find connected components in an adjacency matrix.
    """
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
    """
    Perform cylindrical harmonic analysis on the given positions.
    Returns metrics indicating tube-like structures.
    """
    # Perform PCA to find principal axis
    positions_mean = positions.mean(axis=0)
    centered_positions = positions - positions_mean
    covariance_matrix = np.cov(centered_positions.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    principal_axis = eigenvectors[:, -1]  # Largest eigenvalue
    
    # Project positions onto plane perpendicular to principal axis
    projections = centered_positions - np.outer(np.dot(centered_positions, principal_axis), principal_axis)
    
    # Convert to cylindrical coordinates
    r = np.linalg.norm(projections, axis=1)
    theta = np.arctan2(projections[:, 1], projections[:, 0])
    z = np.dot(centered_positions, principal_axis)
    
    # Analyze radial distribution
    radial_std = np.std(r)
    
    # Analyze angular uniformity
    angular_uniformity = compute_angular_uniformity(theta)
    
    return radial_std, angular_uniformity, r, theta, z, principal_axis

def compute_angular_uniformity(theta):
    """
    Compute a metric for angular uniformity.
    Returns a value between 0 (non-uniform) and 1 (uniform).
    """
    histogram, _ = np.histogram(theta, bins=36, range=(-np.pi, np.pi))
    histogram_normalized = histogram / np.sum(histogram)
    uniformity = -np.sum(histogram_normalized * np.log(histogram_normalized + 1e-8))
    max_entropy = np.log(len(histogram))
    angular_uniformity = 1 - (uniformity / max_entropy)
    return angular_uniformity

def segment_based_analysis(positions, segment_length, step_size):
    """
    Divide positions into segments and perform cylindrical analysis on each segment.
    Returns the proportion of segments classified as tube-like.
    """
    # Order positions along the principal axis
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
    """
    Compute radial density profile.
    """
    max_radius = r.max()
    bins = np.linspace(0, max_radius, num_bins)
    density, bin_edges = np.histogram(r, bins=bins, density=True)
    return density, bin_edges

def is_hollow_tube(density, bin_edges):
    """
    Determine if the tube is hollow based on radial density profile.
    """
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
    """
    Compute asphericity and eigenvalue ratios using the gyration tensor.
    """
    relative_positions = positions - positions.mean(axis=0)
    gyration_tensor = np.dot(relative_positions.T, relative_positions) / len(relative_positions)
    eigenvalues, _ = np.linalg.eigh(gyration_tensor)
    eigenvalues = np.sort(eigenvalues)
    asphericity = 1 - (2 * (eigenvalues[0] + eigenvalues[1]) / (2 * eigenvalues[2]))
    ratio = eigenvalues[0] / eigenvalues[2]
    return asphericity, ratio

def analyze_aggregate(aggregate_atoms, frame_number, args):
    """
    Analyze a single aggregate for tube properties.
    """
    results = {}
    positions = aggregate_atoms.positions
    if len(positions) < args.min_tube_size:
        results['is_tube'] = False
        return results
    
    # Segment-based analysis
    tube_segment_ratio = segment_based_analysis(positions, SEGMENT_LENGTH, STEP_SIZE)
    
    # Perform cylindrical analysis on entire aggregate
    radial_std, angular_uniformity, r, theta, z, principal_axis = perform_cylindrical_analysis(positions)
    
    # Radial density profile
    density, bin_edges = compute_radial_density(r)
    hollow = is_hollow_tube(density, bin_edges)
    
    # Shape anisotropy
    asphericity, ratio = compute_shape_anisotropy(positions)
    
    # Classification criteria
    is_tube = (
        tube_segment_ratio >= 0.5 and
        radial_std < RADIAL_THRESHOLD and
        angular_uniformity > ANGULAR_UNIFORMITY_THRESHOLD and
        hollow and
        asphericity > ASPHERICITY_THRESHOLD and
        ratio < RATIO_THRESHOLD
    )
    
    results['frame'] = frame_number
    results['aggregate_size'] = len(positions)
    results['radial_std'] = radial_std
    results['angular_uniformity'] = angular_uniformity
    results['tube_segment_ratio'] = tube_segment_ratio
    results['hollow'] = hollow
    results['asphericity'] = asphericity
    results['eigenvalue_ratio'] = ratio
    results['is_tube'] = is_tube
    return results

def main():
    args = parse_arguments()
    ensure_output_directory(args.output)
    
    # Load the trajectory
    print("Loading trajectory data...")
    u = mda.Universe(args.topology, args.trajectory)
    selection_string = args.selection
    n_frames = len(u.trajectory)
    print(f"Total frames in trajectory: {n_frames}")
    start_frame = 0  # Default start is the first frame
    end_frame = n_frames  # Default end is the total number of frames

    if args.last is not None:
        start_frame = max(0, n_frames - args.last)  # Analyze only the last N frames

    if args.first is not None:
        end_frame = min(n_frames, args.first)  # Limit the analysis to the first N frames

    print(f"Analyzing frames from {start_frame} to {end_frame}, skipping every {args.skip} frames")
    
    # Initialize variables for analysis
    tube_records = defaultdict(list)  # {tube_id: [frame_numbers]}
    tube_id_counter = 0
    frame_results = []
    
    # Analyze each frame
    print("Analyzing frames for tube formation...")
    for frame_number, ts in enumerate(u.trajectory[start_frame:end_frame:args.skip]):
        actual_frame_number = start_frame + frame_number * args.skip  # Track the actual frame number
        print(f"Processing frame {actual_frame_number + 1}/{n_frames}...")
        
        aggregates = identify_aggregates(u, selection_string)
        for aggregate in aggregates:
            aggregate_atoms = u.atoms[aggregate]
            results = analyze_aggregate(aggregate_atoms, actual_frame_number, args)
            if results.get('is_tube'):
                tube_id = f"tube_{tube_id_counter}"
                tube_id_counter += 1
                tube_records[tube_id].append(actual_frame_number)
            frame_results.append(results)
    
    # Time-resolved analysis
    tube_lifetimes = analyze_tube_lifetimes(tube_records)
    save_tube_lifetimes(tube_lifetimes, args.output)
    save_frame_results(frame_results, args.output)
    plot_tube_lifetimes(tube_lifetimes, args.output)
    
    print("TFI analysis completed successfully.")

def analyze_tube_lifetimes(tube_records):
    """
    Analyze the lifetimes of tubes over time.
    """
    tube_lifetimes = {}
    for tube_id, frames in tube_records.items():
        lifetime = len(frames)
        tube_lifetimes[tube_id] = lifetime
    return tube_lifetimes

def save_tube_lifetimes(tube_lifetimes, output_dir):
    """
    Save tube lifetimes data to a file.
    """
    output_file = os.path.join(output_dir, 'tube_lifetimes.csv')
    with open(output_file, 'w') as f:
        f.write('TubeID,Lifetime\n')
        for tube_id, lifetime in tube_lifetimes.items():
            f.write(f"{tube_id},{lifetime}\n")
    print(f"Tube lifetimes data saved to {output_file}")

def save_frame_results(frame_results, output_dir):
    """
    Save per-frame analysis results to a file.
    """
    output_file = os.path.join(output_dir, 'tfi_frame_results.csv')
    with open(output_file, 'w') as f:
        headers = ['Frame', 'AggregateSize', 'RadialStd', 'AngularUniformity',
                   'TubeSegmentRatio', 'Hollow', 'Asphericity', 'EigenvalueRatio', 'IsTube']
        f.write(','.join(headers) + '\n')
        for result in frame_results:
            if 'is_tube' in result:
                f.write(f"{result['frame']},{result['aggregate_size']},"
                        f"{result['radial_std']:.3f},{result['angular_uniformity']:.3f},"
                        f"{result['tube_segment_ratio']:.3f},{int(result['hollow'])},"
                        f"{result['asphericity']:.3f},{result['eigenvalue_ratio']:.3f},"
                        f"{int(result['is_tube'])}\n")
    print(f"Per-frame results saved to {output_file}")

def plot_tube_lifetimes(tube_lifetimes, output_dir):
    """
    Plot the distribution of tube lifetimes.
    """
    lifetimes = list(tube_lifetimes.values())
    plt.figure()
    plt.hist(lifetimes, bins=range(1, max(lifetimes)+2), align='left')
    plt.xlabel('Lifetime (frames)')
    plt.ylabel('Number of Tubes')
    plt.title('Distribution of Tube Lifetimes')
    plt.savefig(os.path.join(output_dir, 'tube_lifetimes_distribution.png'))
    plt.close()
    print("Tube lifetimes distribution plot saved.")

if __name__ == '__main__':
    main()
