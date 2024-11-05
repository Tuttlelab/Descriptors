#!/usr/bin/env python3
"""
vfi_analysis.py

This script calculates the Vesicle Formation Index (VFI) for peptide simulations.
It incorporates advanced features such as:

- Radial density profiling for detailed hollowness assessment.
- Surface mesh generation using convex hulls for accurate sphericity calculations.
- Internal void analysis using voxelization and flood-fill algorithms.
- Shape descriptors beyond sphericity, including asphericity and acylindricity.
- Temporal tracking of vesicle formation and stability over the simulation time.

"""

import os
import argparse
import numpy as np
import MDAnalysis as mda
from scipy.spatial import ConvexHull
from scipy.ndimage import binary_fill_holes
from skimage.measure import label
from skimage.morphology import ball
from collections import defaultdict
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Constants
DENSITY_BINS = 100
SPHERICITY_THRESHOLD = 0.8
HOLLOWNESS_THRESHOLD = 0.3
ASPHERICITY_THRESHOLD = 0.2
ACYLINDRICITY_THRESHOLD = 0.2
MIN_VESICLE_SIZE = 50  # Minimum number of atoms to consider an aggregate as a vesicle

def parse_arguments():
    parser = argparse.ArgumentParser(description='Vesicle Formation Index (VFI) Analysis')
    parser.add_argument('-t', '--topology', required=True, help='Topology file (e.g., .gro, .pdb)')
    parser.add_argument('-x', '--trajectory', required=True, help='Trajectory file (e.g., .xtc, .trr)')
    parser.add_argument('-s', '--selection', default='all', help='Atom selection string for peptides')
    parser.add_argument('-o', '--output', default='vfi_results', help='Output directory for results')
    parser.add_argument('--min_vesicle_size', type=int, default=MIN_VESICLE_SIZE,
                        help='Minimum number of atoms to consider an aggregate as a vesicle')
    parser.add_argument('--first', type=int, default=None, help='Only analyze the first N frames (default is all frames)')
    parser.add_argument('--last', type=int, default=None, help='Only analyze the last N frames (default is all frames)')
    parser.add_argument('--skip', type=int, default=1, help='Process every nth frame (default is every frame)')
    args = parser.parse_args()
    return args

def ensure_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def setup_logging(output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(output_dir, f"vfi_{timestamp}.log")
    logging.basicConfig(filename=log_filename, level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("VFI Analysis started.")

def identify_aggregates(universe, selection_string):
    """
    Identify aggregates (clusters) in the system using a distance cutoff.
    """
    selection = universe.select_atoms(selection_string)
    positions = selection.positions
    distance_matrix = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1)
    adjacency_matrix = distance_matrix < 6.0  # Distance cutoff in Å
    np.fill_diagonal(adjacency_matrix, 0)
    labels, num_labels = connected_components(adjacency_matrix)
    aggregates = defaultdict(list)
    for idx, label_id in enumerate(labels):
        aggregates[label_id].append(selection.atoms[idx].index)
    return aggregates.values()

def connected_components(adjacency_matrix):
    """
    Custom connected components finder for the adjacency matrix.
    Returns labels indicating component membership.
    """
    n_nodes = adjacency_matrix.shape[0]
    visited = np.zeros(n_nodes, dtype=bool)
    labels = np.full(n_nodes, -1, dtype=int)
    label_id = 0

    for node in range(n_nodes):
        if not visited[node]:
            stack = [node]
            while stack:
                current = stack.pop()
                if not visited[current]:
                    visited[current] = True
                    labels[current] = label_id
                    neighbors = np.where(adjacency_matrix[current])[0]
                    stack.extend(neighbors)
            label_id += 1
    return labels, label_id

def compute_radial_density(positions, com, num_bins):
    distances = np.linalg.norm(positions - com, axis=1)
    max_distance = distances.max()
    bins = np.linspace(0, max_distance, num_bins)
    density, bin_edges = np.histogram(distances, bins=bins, density=True)
    return density, bin_edges

def is_hollow(density, bin_edges):
    from scipy.signal import argrelextrema
    density_smooth = np.convolve(density, np.ones(5)/5, mode='same')
    maxima = argrelextrema(density_smooth, np.greater)[0]
    minima = argrelextrema(density_smooth, np.less)[0]
    if len(maxima) > 0 and len(minima) > 0:
        shell_peak = density_smooth[maxima[0]]
        core_min = density_smooth[minima[0]]
        if core_min < shell_peak * 0.5:
            return True
    return False

def compute_sphericity(positions):
    if len(positions) < 4:
        return 0
    hull = ConvexHull(positions)
    surface_area = hull.area
    volume = hull.volume
    sphericity = (np.pi**(1/3)) * (6 * volume)**(2/3) / surface_area
    return sphericity


def compute_hollowness_ratio(positions, voxel_size=2.0):
    """
    Quantify hollowness using voxelization and flood fill algorithm.
    """
    # Define voxel grid dimensions
    min_coords = positions.min(axis=0) - voxel_size
    max_coords = positions.max(axis=0) + voxel_size
    grid_shape = np.ceil((max_coords - min_coords) / voxel_size).astype(int)
    print(f"Grid shape for hollowness calculation: {grid_shape}")  # Debugging output
    grid = np.zeros(grid_shape, dtype=bool)

    # Map positions to grid indices
    indices = np.floor((positions - min_coords) / voxel_size).astype(int)
    grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True

    # Try to fill internal voids
    try:
        filled_grid = binary_fill_holes(grid)
        if filled_grid is None:
            print("Warning: binary_fill_holes returned None. Defaulting hollowness_ratio to 0.")
            return 0  # Default hollowness ratio if filling fails

        # Compute volumes
        aggregate_volume = int(grid.sum())
        total_volume = int(filled_grid.sum())
        void_volume = total_volume - aggregate_volume

        hollowness_ratio = void_volume / total_volume if total_volume > 0 else 0
        return hollowness_ratio
    except MemoryError:
        print("Memory error during hollowness calculation. Consider adjusting voxel size or analyzing fewer frames.")
        return 0  # Return 0 as the default if memory error occurs

def compute_shape_descriptors(positions, com):
    relative_positions = positions - com
    gyration_tensor = np.dot(relative_positions.T, relative_positions) / len(relative_positions)
    eigenvalues, _ = np.linalg.eigh(gyration_tensor)
    lambda_avg = eigenvalues.mean()
    asphericity = ((eigenvalues - lambda_avg)**2).sum() / (2 * lambda_avg**2)
    acylindricity = ((eigenvalues[1] - eigenvalues[0])**2 + (eigenvalues[2] - eigenvalues[1])**2 +
                     (eigenvalues[0] - eigenvalues[2])**2) / (2 * lambda_avg**2)
    return asphericity, acylindricity

def analyze_aggregate(aggregate_atoms, frame_number, args):
    results = {}
    positions = aggregate_atoms.positions
    com = positions.mean(axis=0)
    density, bin_edges = compute_radial_density(positions, com, DENSITY_BINS)
    hollow = is_hollow(density, bin_edges)
    sphericity = compute_sphericity(positions)
    hollowness_ratio = compute_hollowness_ratio(positions)
    asphericity, acylindricity = compute_shape_descriptors(positions, com)

    is_vesicle = (
        len(positions) >= args.min_vesicle_size and
        sphericity >= SPHERICITY_THRESHOLD and
        hollow and
        hollowness_ratio >= HOLLOWNESS_THRESHOLD and
        asphericity <= ASPHERICITY_THRESHOLD and
        acylindricity <= ACYLINDRICITY_THRESHOLD
    )

    logging.debug(f"Frame {frame_number}: sphericity={sphericity:.3f}, hollowness_ratio={hollowness_ratio:.3f}, "
                  f"asphericity={asphericity:.3f}, acylindricity={acylindricity:.3f}, is_vesicle={is_vesicle}")

    results['frame'] = frame_number
    results['aggregate_size'] = len(positions)
    results['sphericity'] = sphericity
    results['hollowness_ratio'] = hollowness_ratio
    results['asphericity'] = asphericity
    results['acylindricity'] = acylindricity
    results['is_vesicle'] = is_vesicle
    return results

def main():
    args = parse_arguments()
    ensure_output_directory(args.output)
    setup_logging(args.output)

    u = mda.Universe(args.topology, args.trajectory)
    selection_string = args.selection
    n_frames = len(u.trajectory)
    print(f"Total frames in trajectory: {n_frames}")
    start_frame, end_frame = 0, n_frames

    if args.last is not None:
        start_frame = max(0, n_frames - args.last)
    if args.first is not None:
        end_frame = min(n_frames, args.first)

    print(f"Analyzing frames from {start_frame} to {end_frame}, skipping every {args.skip} frames")

    vesicle_records = defaultdict(list)
    vesicle_id_counter = 0
    frame_results = []

    print("Analyzing frames for vesicle formation...")
    for frame_number, ts in enumerate(u.trajectory[start_frame:end_frame:args.skip]):
        actual_frame_number = start_frame + frame_number * args.skip
        logging.info(f"Processing frame {actual_frame_number + 1}/{n_frames}")
        print(f"Processing frame {actual_frame_number + 1}/{n_frames}")

        aggregates = identify_aggregates(u, selection_string)
        for aggregate in aggregates:
            aggregate_atoms = u.select_atoms(selection_string)[aggregate]
            results = analyze_aggregate(aggregate_atoms, actual_frame_number, args)
            if results['is_vesicle']:
                vesicle_id = f"vesicle_{vesicle_id_counter}"
                vesicle_id_counter += 1
                vesicle_records[vesicle_id].append(actual_frame_number)
            frame_results.append(results)

    vesicle_lifetimes = analyze_vesicle_lifetimes(vesicle_records)
    save_vesicle_lifetimes(vesicle_lifetimes, args.output)
    save_frame_results(frame_results, args.output)
    plot_vesicle_lifetimes(vesicle_lifetimes, args.output)

    print("VFI analysis completed successfully.")

def analyze_vesicle_lifetimes(vesicle_records):
    """
    Analyze the lifetimes of vesicles over time.
    """
    vesicle_lifetimes = {}
    for vesicle_id, frames in vesicle_records.items():
        lifetime = len(frames)
        vesicle_lifetimes[vesicle_id] = lifetime
    return vesicle_lifetimes

def save_vesicle_lifetimes(vesicle_lifetimes, output_dir):
    """
    Save vesicle lifetimes data to a file.
    """
    output_file = os.path.join(output_dir, 'vesicle_lifetimes.csv')
    with open(output_file, 'w') as f:
        f.write('VesicleID,Lifetime\n')
        for vesicle_id, lifetime in vesicle_lifetimes.items():
            f.write(f"{vesicle_id},{lifetime}\n")
    print(f"Vesicle lifetimes data saved to {output_file}")

def save_frame_results(frame_results, output_dir):
    """
    Save per-frame analysis results to a file.
    """
    output_file = os.path.join(output_dir, 'vfi_frame_results.csv')
    with open(output_file, 'w') as f:
        headers = ['Frame', 'AggregateSize', 'Sphericity', 'HollownessRatio',
                   'Asphericity', 'Acylindricity', 'IsVesicle']
        f.write(','.join(headers) + '\n')
        for result in frame_results:
            f.write(f"{result['frame']},{result['aggregate_size']},"
                    f"{result['sphericity']:.3f},{result['hollowness_ratio']:.3f},"
                    f"{result['asphericity']:.3f},{result['acylindricity']:.3f},"
                    f"{int(result['is_vesicle'])}\n")
    print(f"Per-frame results saved to {output_file}")

def plot_vesicle_lifetimes(vesicle_lifetimes, output_dir):
    """
    Plot the distribution of vesicle lifetimes.
    """
    lifetimes = list(vesicle_lifetimes.values())
    plt.figure()
    plt.hist(lifetimes, bins=range(1, max(lifetimes)+2), align='left')
    plt.xlabel('Lifetime (frames)')
    plt.ylabel('Number of Vesicles')
    plt.title('Distribution of Vesicle Lifetimes')
    plt.savefig(os.path.join(output_dir, 'vesicle_lifetimes_distribution.png'))
    plt.close()
    print("Vesicle lifetimes distribution plot saved.")

if __name__ == '__main__':
    main()
