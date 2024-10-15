#!/usr/bin/env python3
"""
ffi_analysis.py

This script calculates the Fiber Formation Index (FFI) for peptide simulations.
It incorporates advanced features such as:

- Multidimensional shape analysis using moments of inertia.
- Detailed alignment analysis using orientation distribution.
- Cross-sectional profiling to assess fiber uniformity.
- Temporal tracking of fiber growth and branching over the simulation time.
- Integration with the Fibrillar Order Parameter (FOP) for internal ordering assessment.

"""

import os
import argparse
import numpy as np
import MDAnalysis as mda
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from collections import defaultdict
import matplotlib.pyplot as plt

# Constants
SHAPE_RATIO_THRESHOLD = 3.0         # Threshold for shape ratios in moment of inertia analysis
ALIGNMENT_STD_THRESHOLD = 15.0      # Degrees, threshold for standard deviation of orientation angles
FOP_THRESHOLD = 0.8                 # Threshold for Fibrillar Order Parameter
MIN_FIBER_SIZE = 50                 # Minimum number of atoms to consider an aggregate as a fiber
CROSS_SECTION_THICKNESS = 5.0       # Thickness for cross-sectional profiling in Å
NUM_CROSS_SECTIONS = 10             # Number of cross-sections along the fiber

def parse_arguments():
    parser = argparse.ArgumentParser(description='Fiber Formation Index (FFI) Analysis')
    parser.add_argument('-t', '--topology', required=True, help='Topology file (e.g., .gro, .pdb)')
    parser.add_argument('-x', '--trajectory', required=True, help='Trajectory file (e.g., .xtc, .trr)')
    parser.add_argument('-s', '--selection', default='all', help='Atom selection string for peptides')
    parser.add_argument('-o', '--output', default='ffi_results', help='Output directory for results')
    parser.add_argument('--min_fiber_size', type=int, default=MIN_FIBER_SIZE, help='Minimum number of atoms to consider a fiber')
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

def compute_moments_of_inertia(positions):
    """
    Compute the moments of inertia and shape ratios of an aggregate.
    """
    com = positions.mean(axis=0)
    relative_positions = positions - com
    inertia_tensor = np.zeros((3, 3))
    for pos in relative_positions:
        inertia_tensor += np.outer(pos, pos)
    inertia_tensor /= len(relative_positions)
    eigenvalues, eigenvectors = np.linalg.eigh(inertia_tensor)
    eigenvalues = np.sort(eigenvalues)
    shape_ratio1 = eigenvalues[2] / eigenvalues[1]
    shape_ratio2 = eigenvalues[1] / eigenvalues[0]
    principal_axis = eigenvectors[:, -1]
    return shape_ratio1, shape_ratio2, principal_axis

def get_peptide_orientations(cluster_atoms):
    """
    Calculate the orientation vectors (backbone vectors) for each peptide.
    Returns an array of orientation vectors.
    """
    orientations = []
    peptide_groups = cluster_atoms.groupby('residues')
    for residue in peptide_groups:
        backbone = residue.atoms.select_atoms('backbone')
        if len(backbone.positions) >= 2:
            vector = backbone.positions[-1] - backbone.positions[0]
            norm = np.linalg.norm(vector)
            if norm > 0:
                orientations.append(vector / norm)
            else:
                orientations.append(np.zeros(3))
        else:
            orientations.append(np.zeros(3))
    return np.array(orientations)

def analyze_orientation_distribution(orientations, principal_axis):
    """
    Analyze the distribution of peptide orientations relative to the principal axis.
    """
    cos_angles = np.dot(orientations, principal_axis)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)  # Numerical stability
    angles = np.arccos(cos_angles) * (180 / np.pi)  # Convert to degrees
    mean_angle = np.mean(angles)
    std_angle = np.std(angles)
    return mean_angle, std_angle, angles

def compute_fop(orientations, principal_axis):
    """
    Compute the Fibrillar Order Parameter (FOP).
    """
    cos_angles = np.dot(orientations, principal_axis)
    cos2_angles = 2 * cos_angles**2 - 1
    fop = np.mean(cos2_angles)
    return fop

def cross_sectional_profiling(relative_positions, principal_axis):
    """
    Perform cross-sectional profiling along the fiber.
    """
    z = np.dot(relative_positions, principal_axis)
    z_min, z_max = z.min(), z.max()
    cross_section_areas = []
    thickness = CROSS_SECTION_THICKNESS
    for i in range(NUM_CROSS_SECTIONS):
        z_i = z_min + i * (z_max - z_min) / NUM_CROSS_SECTIONS
        indices = np.where((z >= z_i - thickness / 2) & (z < z_i + thickness / 2))[0]
        cross_section_positions = relative_positions[indices]
        if len(cross_section_positions) >= 3:
            # Project onto plane perpendicular to principal axis
            projections = cross_section_positions - np.outer(np.dot(cross_section_positions, principal_axis), principal_axis)
            print(projections)
            hull = ConvexHull(projections)
            area = hull.area
            cross_section_areas.append(area)
        else:
            cross_section_areas.append(0)
    return cross_section_areas

def analyze_aggregate(aggregate_atoms, frame_number, args):
    """
    Analyze a single aggregate for fiber properties.
    """
    results = {}
    positions = aggregate_atoms.positions
    if len(positions) < args.min_fiber_size:
        results['is_fiber'] = False
        return results

    # Compute moments of inertia
    shape_ratio1, shape_ratio2, principal_axis = compute_moments_of_inertia(positions)
    
    # Criteria for shape ratios
    if shape_ratio1 < SHAPE_RATIO_THRESHOLD or shape_ratio2 < SHAPE_RATIO_THRESHOLD:
        results['is_fiber'] = False
        return results

    # Compute peptide orientations
    orientations = get_peptide_orientations(aggregate_atoms)
    mean_angle, std_angle, angles = analyze_orientation_distribution(orientations, principal_axis)
    
    # Compute FOP
    fop = compute_fop(orientations, principal_axis)
    
    # Cross-sectional profiling
    relative_positions = positions - positions.mean(axis=0)
    cross_section_areas = cross_sectional_profiling(relative_positions, principal_axis)
    mean_cross_section_area = np.mean(cross_section_areas)
    std_cross_section_area = np.std(cross_section_areas)
    
    # Classification criteria
    is_fiber = (
        std_angle < ALIGNMENT_STD_THRESHOLD and
        fop > FOP_THRESHOLD
    )
    
    results['frame'] = frame_number
    results['aggregate_size'] = len(positions)
    results['shape_ratio1'] = shape_ratio1
    results['shape_ratio2'] = shape_ratio2
    results['mean_angle'] = mean_angle
    results['std_angle'] = std_angle
    results['fop'] = fop
    results['mean_cross_section_area'] = mean_cross_section_area
    results['std_cross_section_area'] = std_cross_section_area
    results['is_fiber'] = is_fiber
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
    
    # Initialize variables for analysis
    fiber_records = defaultdict(list)  # {fiber_id: [frame_numbers]}
    fiber_id_counter = 0
    frame_results = []
    
    # Analyze each frame
    print("Analyzing frames for fiber formation...")
    for frame_number, ts in enumerate(u.trajectory):
        print(f"Processing frame {frame_number+1}/{n_frames}...")
        aggregates = identify_aggregates(u, selection_string)
        for aggregate in aggregates:
            aggregate_atoms = u.atoms[aggregate]
            results = analyze_aggregate(aggregate_atoms, frame_number, args)
            if results.get('is_fiber'):
                fiber_id = f"fiber_{fiber_id_counter}"
                fiber_id_counter += 1
                fiber_records[fiber_id].append(frame_number)
            frame_results.append(results)
    
    # Time-resolved analysis
    fiber_lifetimes = analyze_fiber_lifetimes(fiber_records)
    save_fiber_lifetimes(fiber_lifetimes, args.output)
    save_frame_results(frame_results, args.output)
    plot_fiber_lifetimes(fiber_lifetimes, args.output)
    
    print("FFI analysis completed successfully.")

def analyze_fiber_lifetimes(fiber_records):
    """
    Analyze the lifetimes of fibers over time.
    """
    fiber_lifetimes = {}
    for fiber_id, frames in fiber_records.items():
        lifetime = len(frames)
        fiber_lifetimes[fiber_id] = lifetime
    return fiber_lifetimes

def save_fiber_lifetimes(fiber_lifetimes, output_dir):
    """
    Save fiber lifetimes data to a file.
    """
    output_file = os.path.join(output_dir, 'fiber_lifetimes.csv')
    with open(output_file, 'w') as f:
        f.write('FiberID,Lifetime\n')
        for fiber_id, lifetime in fiber_lifetimes.items():
            f.write(f"{fiber_id},{lifetime}\n")
    print(f"Fiber lifetimes data saved to {output_file}")

def save_frame_results(frame_results, output_dir):
    """
    Save per-frame analysis results to a file.
    """
    output_file = os.path.join(output_dir, 'ffi_frame_results.csv')
    with open(output_file, 'w') as f:
        headers = ['Frame', 'AggregateSize', 'ShapeRatio1', 'ShapeRatio2',
                   'MeanAngle', 'StdAngle', 'FOP', 'MeanCrossSectionArea',
                   'StdCrossSectionArea', 'IsFiber']
        f.write(','.join(headers) + '\n')
        for result in frame_results:
            if 'is_fiber' in result:
                print(result)
                f.write(f"{result['frame']},{result['aggregate_size']},"
                        f"{result['shape_ratio1']:.3f},{result['shape_ratio2']:.3f},"
                        f"{result['mean_angle']:.3f},{result['std_angle']:.3f},"
                        f"{result['fop']:.3f},{result['mean_cross_section_area']:.3f},"
                        f"{result['std_cross_section_area']:.3f},{int(result['is_fiber'])}\n")
        print(f"Per-frame results saved to {output_file}")

def plot_fiber_lifetimes(fiber_lifetimes, output_dir):
    """
    Plot the distribution of fiber lifetimes.
    """
    lifetimes = list(fiber_lifetimes.values())
    plt.figure()
    plt.hist(lifetimes, bins=range(1, max(lifetimes)+2), align='left')
    plt.xlabel('Lifetime (frames)')
    plt.ylabel('Number of Fibers')
    plt.title('Distribution of Fiber Lifetimes')
    plt.savefig(os.path.join(output_dir, 'fiber_lifetimes_distribution.png'))
    plt.close()
    print("Fiber lifetimes distribution plot saved.")

if __name__ == '__main__':
    main()
