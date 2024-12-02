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

import warnings
# Suppress BiopythonDeprecationWarning messages before any Bio modules are imported
warnings.filterwarnings("ignore", message=".*Bio.Application modules and modules relying on it have been deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning)

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

import logging
from datetime import datetime
import csv

# Set up logging
timestamp = datetime.now().strftime("%m%d_%H%M")
log_filename = f"ffi_results/ffi_{timestamp}.log"
logging.basicConfig(filename=log_filename, level=logging.DEBUG, format='%(message)s')

# Constants
DEFAULT_MIN_FIBER_SIZE = 1000            # Minimum number of beads to consider a fiber
SHAPE_RATIO_THRESHOLD = 1.5           # Threshold for shape ratios in moment of inertia analysis
ALIGNMENT_STD_THRESHOLD = 50.0        # Degrees, threshold for standard deviation of orientation angles
FOP_THRESHOLD = 0.7                   # Threshold for Fibrillar Order Parameter
CROSS_SECTION_THICKNESS = 5.0         # Thickness for cross-sectional profiling in Å
NUM_CROSS_SECTIONS = 10               # Number of cross-sections along the fiber
DEFAULT_DISTANCE_CUTOFF = 7.0         # Distance cutoff for clustering in Å
FOP_THRESHOLD_POSITIVE = 0.1  # For alignment
FOP_THRESHOLD_NEGATIVE = -0.1  # For anti-alignment
CSV_HEADERS = ['Frame', 'Peptides', 'fiber_count', 'total_peptides_in_fibers', 'avg_fiber_size']

def parse_arguments():
    parser = argparse.ArgumentParser(description='Fiber Formation Index (FFI) Analysis')
    parser.add_argument('-t', '--topology', required=True, help='Topology file (e.g., .gro, .pdb)')
    parser.add_argument('-x', '--trajectory', required=True, help='Trajectory file (e.g., .xtc, .trr)')
    parser.add_argument('-s', '--selection', default='protein', help='Atom selection string for peptides')
    parser.add_argument('-o', '--output', default='ffi_results', help='Output directory for results')
    parser.add_argument('--min_fiber_size', type=int, default=DEFAULT_MIN_FIBER_SIZE, help='Minimum number of beads to consider a fiber')
    parser.add_argument('--distance_cutoff', type=float, default=DEFAULT_DISTANCE_CUTOFF, help='Distance cutoff for clustering in Å')
    parser.add_argument('--first', type=int, default=0, help='First frame to analyze (default is 0)')
    parser.add_argument('--last', type=int, default=None, help='Last frame to analyze (default is all frames)')
    parser.add_argument('--skip', type=int, default=1, help='Process every nth frame (default is every frame)')
    args = parser.parse_args()
    return args

def ensure_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def load_and_crop_trajectory(topology, trajectory, first, last, skip, selection):
    u = mda.Universe(topology, trajectory)

    # Set the total number of frames
    total_frames = len(u.trajectory)
    print()
    print(f"Total frames: {total_frames}")
    print()
    print(f"First frame: {first}")
    print()
    print(f"Last frame: {last}")
    print()
    print(f"Skip: {skip}")
    print()

    # Validate and set 'first' and 'last'
    if last is None or last > total_frames:
        last = total_frames
    if first < 0 or first >= total_frames:
        raise ValueError(f"Invalid first frame: {first}.")

    # Ensure that 'last' is greater than 'first'
    if last <= first:
        raise ValueError(f"'last' frame must be greater than 'first' frame. Got first={first}, last={last}.")

    # Select the specified beads
    selection_atoms = u.select_atoms(selection)
    if len(selection_atoms) == 0:
        raise ValueError(f"Selection '{selection}' returned no beads.")

    # Create the list of frame indices to process
    indices = list(range(first, last, skip))
    logging.debug(f"Indices to be processed: {indices}")

    return u, selection_atoms, indices

def center_and_wrap_trajectory(universe, selection_string):
    """
    Center the selected group in the simulation box and wrap all atoms to handle PBC issues.
    """
    selection = universe.select_atoms(selection_string)

    # Calculate the center of mass of the selection
    com = selection.center_of_mass()

    # Get the simulation box dimensions
    box_dimensions = universe.dimensions[:3]  # [lx, ly, lz]

    # Calculate the center of the box
    box_center = box_dimensions / 2

    # Calculate the shift vector needed to move COM to box center
    shift = box_center - com

    # Translate the entire system by the shift vector
    universe.atoms.translate(shift)

    # Wrap all atoms back into the primary simulation box
    universe.atoms.wrap()

    # Optional: Recompute the center of mass after translation and wrapping
    new_com = selection.center_of_mass()
    logging.debug(f"Initial COM: {com}, Shift Applied: {shift}, New COM after wrapping: {new_com}")

def identify_aggregates(universe, selection_string, distance_cutoff, min_fiber_size):
    """Modified to return both aggregates and their peptide indices"""
    selection = universe.select_atoms(selection_string)
    positions = selection.positions
    distance_matrix = cdist(positions, positions)
    adjacency_matrix = distance_matrix < distance_cutoff
    np.fill_diagonal(adjacency_matrix, False)

    labels, num_labels = connected_components(adjacency_matrix)

    aggregates = []
    aggregate_indices = []
    for label in range(num_labels):
        indices = np.where(labels == label)[0]
        if len(indices) >= min_fiber_size:
            aggregates.append(selection.atoms[indices])
            aggregate_indices.append(indices)

    return aggregates, aggregate_indices

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

def get_peptide_orientations(cluster_atoms, frame_number):
    """
    Calculate the orientation vectors (backbone vectors) for each dipeptide.
    Returns an array of orientation vectors.
    """
    orientations = []
    peptide_groups = cluster_atoms.residues  # Group atoms by residue (peptide)

    # Debug: Print the number of residues
    logging.debug(f"Frame {frame_number}: Number of residues: {len(peptide_groups)}")

    # Iterate over pairs of residues to form dipeptides
    for i in range(0, len(peptide_groups) - 1, 2):
        # len(peptide_groups) - 1
        residue1 = peptide_groups[i]
        residue2 = peptide_groups[i + 1]

        # Select backbone atoms
        backbone1 = residue1.atoms.select_atoms('name BB')
        backbone2 = residue2.atoms.select_atoms('name BB')

        if len(backbone1.positions) == 1 and len(backbone2.positions) == 1:  # Check if each residue has exactly 1 backbone atom
            vector = backbone2.positions[0] - backbone1.positions[0]  # Calculate vector from residue1 BB to residue2 BB
            norm = np.linalg.norm(vector)  # Calculate the norm (length) of the vector

            if norm > 0:
                orientations.append(vector / norm)  # Normalize the vector and append to orientations
            else:
                orientations.append(np.zeros(3))  # Append zero vector if norm is zero
        else:
            logging.debug(f"Residue pair {residue1.resid}-{residue2.resid} has insufficient backbone beads for orientation calculation.")
            orientations.append(np.zeros(3))  # Append zero vector if there are not exactly 1 backbone atom in each residue

    return np.array(orientations)  # Return orientations as a NumPy array

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
    cos2_angles = (3 * cos_angles**2 - 1) / 2  # Standard P2(cosθ)
    fop = np.mean(cos2_angles)
    # FOP = 1: Perfect alignment.
    # FOP = -0.5: Perfect anti-alignment.
    # FOP = 0: Random orientation.
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

def analyze_aggregate(aggregate_atoms, frame_number, peptide_indices, args):
    """Modified to include peptide tracking"""
    results = {}
    positions = aggregate_atoms.positions

    if len(positions) < args.min_fiber_size:
        results['is_fiber'] = False
        results['peptides'] = []
        return results

    # Compute moments of inertia
    shape_ratio1, shape_ratio2, principal_axis = compute_moments_of_inertia(positions)

    if shape_ratio1 < SHAPE_RATIO_THRESHOLD or shape_ratio2 < SHAPE_RATIO_THRESHOLD:
        results['is_fiber'] = False
        results['peptides'] = []
        return results

    # Get orientations
    orientations = get_peptide_orientations(aggregate_atoms, frame_number)

    # Compute metrics
    mean_angle, std_angle, angles = analyze_orientation_distribution(orientations, principal_axis)
    fop = compute_fop(orientations, principal_axis)
    relative_positions = positions - positions.mean(axis=0)
    cross_section_areas = cross_sectional_profiling(relative_positions, principal_axis)

    # Classification criteria
    is_fiber = (
        std_angle < ALIGNMENT_STD_THRESHOLD and
        (fop > FOP_THRESHOLD_POSITIVE or fop < FOP_THRESHOLD_NEGATIVE)
    )

    results = {
        'frame': frame_number,
        'size': len(positions),
        'shape_ratio1': shape_ratio1,
        'shape_ratio2': shape_ratio2,
        'mean_angle': mean_angle,
        'std_angle': std_angle,
        'fop': fop,
        'is_fiber': is_fiber,
        'peptides': [f'PEP{idx+1}' for idx in peptide_indices]
    }

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
    print()

def save_frame_results(frame_records, output_dir):
    """Save FFI frame results to a CSV file."""
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_file = os.path.join(output_dir, f'ffi_frame_results_{timestamp}.csv')

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for record in frame_records:
            writer.writerow(record)

    logging.info(f"FFI frame results saved to {output_file}")

def plot_fiber_lifetimes(fiber_lifetimes, output_dir):
    """
    Plot the distribution of fiber lifetimes.
    """
    lifetimes = [data['lifetime'] for data in fiber_lifetimes.values()]
    if not lifetimes:
        print("No fibers detected; skipping plot.")
        print()
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
    print()

def plot_number_of_fibers_per_frame(frame_results, output_dir):
    """
    Plot the number of fibers in each frame.
    """
    from collections import defaultdict

    fiber_counts = defaultdict(int)
    for result in frame_results:
        if result.get('is_fiber', False):
            frame = result.get('frame')
            fiber_counts[frame] += 1

    frames = sorted(fiber_counts.keys())
    counts = [fiber_counts[frame] for frame in frames]

    plt.figure(figsize=(10, 6))
    plt.plot(frames, counts, marker='o', linestyle='-')
    plt.xlabel('Frame')
    plt.ylabel('Number of Fibers')
    plt.title('Number of Fibers per Frame')
    plt.grid(True)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    plot_filename = os.path.join(output_dir, f'number_of_fibers_per_frame_{timestamp}.png')
    plt.savefig(plot_filename)
    plt.close()
    print(f"Number of fibers per frame plot saved to {plot_filename}")
    print()

def track_fibers_across_frames(fiber_records, frame_results, distance_threshold=5.0):
    """
    Track fibers across frames by comparing their properties.
    """
    tracked_fibers = defaultdict(list)
    fiber_id_counter = 0

    for frame_number, frame_result in enumerate(frame_results):
        current_fibers = [result for result in frame_result if result.get('is_fiber', False)]

        if frame_number == 0:
            # Initialize tracking for the first frame
            for fiber in current_fibers:
                tracked_fibers[fiber_id_counter].append((frame_number, fiber))
                fiber_id_counter += 1
        else:
            # Track fibers in subsequent frames
            previous_fibers = [tracked_fibers[fiber_id][-1][1] for fiber_id in tracked_fibers]
            for fiber in current_fibers:
                matched = False
                for prev_fiber in previous_fibers:
                    if np.linalg.norm(fiber['positions'].mean(axis=0) - prev_fiber['positions'].mean(axis=0)) < distance_threshold:
                        fiber_id = [fid for fid in tracked_fibers if tracked_fibers[fid][-1][1] == prev_fiber][0]
                        tracked_fibers[fiber_id].append((frame_number, fiber))
                        matched = True
                        break
                if not matched:
                    tracked_fibers[fiber_id_counter].append((frame_number, fiber))
                    fiber_id_counter += 1

    return tracked_fibers

def plot_tracked_fibers(tracked_fibers, output_dir):
    """
    Plot the tracked fibers over time.
    """
    plt.figure(figsize=(10, 6))

    for fiber_id, fiber_data in tracked_fibers.items():
        frames = [data[0] for data in fiber_data]
        sizes = [data[1]['aggregate_size'] for data in fiber_data]
        plt.plot(frames, sizes, marker='o', linestyle='-', label=f'Fiber {fiber_id}')

    plt.xlabel('Frame')
    plt.ylabel('Fiber Size')
    plt.title('Tracked Fibers Over Time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Save the plot with a timestamped filename
    timestamp = datetime.now().strftime("%m%d_%H%M")
    plot_filename = os.path.join(output_dir, f'tracked_fibers_{timestamp}.png')
    plt.savefig(plot_filename)
    plt.close()
    print(f"Tracked fibers plot saved to {plot_filename}")
    print()

def main():
    args = parse_arguments()
    ensure_output_directory(args.output)

    # # Initialize MPI
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # size = comm.Get_size()

    # Load trajectory and setup variables
    u, selection_atoms, indices = load_and_crop_trajectory(
        args.topology,
        args.trajectory,
        args.first,
        args.last,
        args.skip,
        args.selection
    )

    # Center and wrap the trajectory if needed
    center_and_wrap_trajectory(u, args.selection)

    # Initialize sheet tracking variables
    sheet_records = defaultdict(list)
    fiber_records = defaultdict(list)
    fiber_id_counter = 0
    frame_records = []

    min_fiber_frames = 1  # Minimum frames a fiber must persist

    # Process each specified frame
    for frame_idx, frame_number in enumerate(indices):
        u.trajectory[frame_number]
        print(f"Processing frame {frame_number + 1}/{len(u.trajectory)}...")
        print()
        logging.info(f"Processing frame {frame_number + 1}/{len(u.trajectory)}...")

        # Select current frame's atoms
        current_atoms = selection_atoms

        # Identify aggregates (clusters)
        aggregates, peptide_indices = identify_aggregates(u, args.selection, args.distance_cutoff, args.min_fiber_size)

        # Track fibers and their peptides for this frame
        frame_fibers = []
        frame_peptides = []

        for aggregate, indices in zip(aggregates, peptide_indices):
            results = analyze_aggregate(aggregate, frame_number, indices, args)

            if results['is_fiber']:
                fiber_id = f"fiber_{fiber_id_counter}"
                fiber_records[fiber_id].append(frame_number)
                fiber_id_counter += 1
                frame_fibers.append(results)
                frame_peptides.extend(results['peptides'])

        # Create frame record
        fiber_count = len(frame_fibers)
        total_peptides = sum(f['size'] for f in frame_fibers)
        avg_fiber_size = total_peptides / fiber_count if fiber_count > 0 else 0

        print(f"{fiber_count} fibers found.")
        print()

        frame_record = {
            'Frame': frame_number,
            'Peptides': str(sorted(frame_peptides)),
            'fiber_count': fiber_count,
            'total_peptides_in_fibers': total_peptides,
            'avg_fiber_size': avg_fiber_size
        }
        frame_records.append(frame_record)

    # Save results
    save_frame_results(frame_records, args.output)

    # Analyze fiber lifetimes
    fiber_lifetimes = analyze_fiber_lifetimes(fiber_records)
    save_fiber_lifetimes(fiber_lifetimes, args.output)

    print("FFI analysis completed successfully.")
    print()
    logging.info("FFI analysis completed successfully.")

if __name__ == '__main__':
    main()
