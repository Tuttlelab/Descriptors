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
from MDAnalysis.analysis import align
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.spatial import ConvexHull
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime

import warnings
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import logging
from datetime import datetime

# Set up logging
timestamp = datetime.now().strftime("%m%d_%H%M")
log_filename = f"ffi_{timestamp}.log"
logging.basicConfig(filename=log_filename, level=logging.DEBUG, format='%(message)s')

# Constants
DEFAULT_MIN_FIBER_SIZE = 5            # Minimum number of atoms to consider a fiber
SHAPE_RATIO_THRESHOLD = 1.5          # Threshold for shape ratios in moment of inertia analysis
ALIGNMENT_STD_THRESHOLD = 20.0        # Degrees, threshold for standard deviation of orientation angles
FOP_THRESHOLD = 0.7                   # Threshold for Fibrillar Order Parameter
CROSS_SECTION_THICKNESS = 5.0         # Thickness for cross-sectional profiling in Å
NUM_CROSS_SECTIONS = 10               # Number of cross-sections along the fiber
DEFAULT_DISTANCE_CUTOFF = 8.0         # Distance cutoff for clustering in Å

def parse_arguments():
    parser = argparse.ArgumentParser(description='Fiber Formation Index (FFI) Analysis')
    parser.add_argument('-t', '--topology', required=True, help='Topology file (e.g., .gro, .pdb)')
    parser.add_argument('-x', '--trajectory', required=True, help='Trajectory file (e.g., .xtc, .trr)')
    parser.add_argument('-s', '--selection', default='protein', help='Atom selection string for peptides')
    parser.add_argument('-o', '--output', default='ffi_results', help='Output directory for results')
    parser.add_argument('--min_fiber_size', type=int, default=DEFAULT_MIN_FIBER_SIZE, help='Minimum number of atoms to consider a fiber')
    parser.add_argument('--distance_cutoff', type=float, default=DEFAULT_DISTANCE_CUTOFF, help='Distance cutoff for clustering in Å')
    parser.add_argument('--first', type=int, default=0, help='First frame to analyze (default is 0)')
    parser.add_argument('--last', type=int, default=None, help='Last frame to analyze (default is all frames)')
    parser.add_argument('--skip', type=int, default=1, help='Process every nth frame (default is every frame)')
    args = parser.parse_args()
    return args

def ensure_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def load_and_crop_trajectory(topology, trajectory, first, last, skip, selection="protein"):
    u = mda.Universe(topology, trajectory)

    # Set last frame if not specified
    total_frames = len(u.trajectory)
    if last is None or last > total_frames:
        last = total_frames
    if first < 0 or first >= total_frames:
        raise ValueError(f"Invalid first frame: {first}.")

    # Select the specified atoms
    selection_atoms = u.select_atoms(selection)
    if len(selection_atoms) == 0:
        raise ValueError(f"Selection '{selection}' returned no atoms.")

    indices = list(range(first, last, skip))

    # Create temporary file names for cropped trajectory
    temp_gro = "temp_protein_slice.gro"
    temp_xtc = "temp_protein_slice.xtc"

    # Write the selected atoms to a temporary trajectory
    with mda.Writer(temp_gro, selection_atoms.n_atoms) as W:
        W.write(selection_atoms)
    with mda.Writer(temp_xtc, selection_atoms.n_atoms) as W:
        for ts in u.trajectory[indices]:
            W.write(selection_atoms)

    # Reload the cropped trajectory
    cropped_u = mda.Universe(temp_gro, temp_xtc)
    return cropped_u

def identify_aggregates(universe, selection_string, distance_cutoff):
    """
    Identify aggregates (clusters) in the system using a distance cutoff.
    """
    selection = universe.select_atoms(selection_string)
    positions = selection.positions
    distance_matrix = cdist(positions, positions)
    adjacency_matrix = distance_matrix < distance_cutoff  # Distance cutoff in Å
    np.fill_diagonal(adjacency_matrix, False)
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
    labels = np.full(n_nodes, -1, dtype=int)
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
    peptide_groups = cluster_atoms.residues
    for residue in peptide_groups:
        backbone = residue.atoms.select_atoms('name BB')
        if len(backbone.positions) >= 2:
            vector = backbone.positions[-1] - backbone.positions[0]
            norm = np.linalg.norm(vector)
            if norm > 0:
                orientations.append(vector / norm)
            else:
                orientations.append(np.zeros(3))
        else:
            logging.debug(f"Residue {residue} has insufficient backbone atoms for orientation calculation.")
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
            hull = ConvexHull(projections[:, :2])  # Use first two coordinates
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
        logging.debug(f"Frame {frame_number}: Aggregate too small to be a fiber (size={len(positions)}).")
        return results

    # Compute moments of inertia
    shape_ratio1, shape_ratio2, principal_axis = compute_moments_of_inertia(positions)

    # Debug print: shape ratios
    logging.debug(f"Frame {frame_number}: shape_ratio1={shape_ratio1:.3f}, shape_ratio2={shape_ratio2:.3f}")

    # Criteria for shape ratios
    if shape_ratio1 < SHAPE_RATIO_THRESHOLD or shape_ratio2 < SHAPE_RATIO_THRESHOLD:
        results['is_fiber'] = False
        logging.debug(f"Frame {frame_number}: Shape ratios below threshold (shape_ratio1={shape_ratio1:.3f}, shape_ratio2={shape_ratio2:.3f}).")
        return results

    # Compute peptide orientations
    orientations = get_peptide_orientations(aggregate_atoms)
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
        std_angle < ALIGNMENT_STD_THRESHOLD and
        fop > FOP_THRESHOLD
    )

    # Debug print: final classification decision
    logging.debug(f"Frame {frame_number}: Is fiber={is_fiber} (std_angle={std_angle:.3f}, FOP={fop:.3f})")

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

def analyze_fiber_lifetimes(fiber_records):
    """
    Analyze the lifetimes of fibers over time.
    """
    fiber_lifetimes = {}
    for fiber_id, frames in fiber_records.items():
        start_frame = min(frames)
        end_frame = max(frames)
        lifetime = end_frame - start_frame + 1  # Inclusive
        fiber_lifetimes[fiber_id] = {
            'start_frame': start_frame,
            'end_frame': end_frame,
            'lifetime': lifetime
        }
    return fiber_lifetimes

def save_fiber_lifetimes(fiber_lifetimes, output_dir):
    """
    Save fiber lifetimes data to a file.
    """
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_file = os.path.join(output_dir, f'fiber_lifetimes_{timestamp}.csv')
    with open(output_file, 'w') as f:
        f.write('FiberID,StartFrame,EndFrame,Lifetime\n')
        for fiber_id, data in fiber_lifetimes.items():
            f.write(f"{fiber_id},{data['start_frame']},{data['end_frame']},{data['lifetime']}\n")
    print(f"Fiber lifetimes data saved to {output_file}")

def save_frame_results(frame_results, output_dir):
    """
    Save per-frame analysis results to a file.
    """
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_file = os.path.join(output_dir, f'ffi_frame_results_{timestamp}.csv')
    with open(output_file, 'w') as f:
        headers = ['Frame', 'AggregateSize', 'ShapeRatio1', 'ShapeRatio2',
                   'MeanAngle', 'StdAngle', 'FOP', 'MeanCrossSectionArea',
                   'StdCrossSectionArea', 'IsFiber']
        f.write(','.join(headers) + '\n')
        for result in frame_results:
            f.write(f"{result.get('frame', '')},{result.get('aggregate_size', '')},"
                    f"{result.get('shape_ratio1', '')},{result.get('shape_ratio2', '')},"
                    f"{result.get('mean_angle', '')},{result.get('std_angle', '')},"
                    f"{result.get('fop', '')},{result.get('mean_cross_section_area', '')},"
                    f"{result.get('std_cross_section_area', '')},{int(result.get('is_fiber', False))}\n")
    print(f"Per-frame results saved to {output_file}")

def plot_fiber_lifetimes(fiber_lifetimes, output_dir):
    """
    Plot the distribution of fiber lifetimes.
    """
    lifetimes = [data['lifetime'] for data in fiber_lifetimes.values()]
    if not lifetimes:
        print("No fibers detected; skipping plot.")
        return
    plt.figure()
    plt.hist(lifetimes, bins=range(1, max(lifetimes)+2), align='left')
    plt.xlabel('Lifetime (frames)')
    plt.ylabel('Number of Fibers')
    plt.title('Distribution of Fiber Lifetimes')
    timestamp = datetime.now().strftime("%m%d_%H%M")
    plt.savefig(os.path.join(output_dir, f'fiber_lifetimes_distribution_{timestamp}.png'))
    plt.close()
    print("Fiber lifetimes distribution plot saved.")

def main():
    args = parse_arguments()
    ensure_output_directory(args.output)

    # Load and crop trajectory
    print("Loading and processing trajectory...")
    u = load_and_crop_trajectory(args.topology, args.trajectory, args.first, args.last, args.skip, args.selection)
    print(f"Total frames in cropped trajectory: {len(u.trajectory)}")

    selection_string = args.selection
    n_frames = len(u.trajectory)
    print(f"Analyzing {n_frames} frames with selection '{selection_string}'")

    # Initialize variables for analysis
    fiber_records = defaultdict(list)  # {fiber_id: [frame_numbers]}
    fiber_id_counter = 0
    frame_results = []

    # Analyze each frame
    print("Analyzing frames for fiber formation...")
    for frame_number, ts in enumerate(u.trajectory):
        print(f"Processing frame {frame_number+1}/{n_frames}...")

        aggregates = identify_aggregates(u, selection_string, args.distance_cutoff)
        for aggregate in aggregates:
            aggregate_atoms = u.select_atoms('index ' + ' '.join(map(str, aggregate)))
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

    # Clean up temporary files
    os.remove("temp_protein_slice.gro")
    os.remove("temp_protein_slice.xtc")

if __name__ == '__main__':
    main()
