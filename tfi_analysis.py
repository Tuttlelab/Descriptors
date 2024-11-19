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
import csv
import argparse
import numpy as np
import MDAnalysis as mda
from scipy.spatial.distance import cdist
from scipy.signal import argrelextrema
from collections import defaultdict
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


def parse_arguments():
    parser = argparse.ArgumentParser(description='Tube Formation Index (TFI) Analysis')
    parser.add_argument('-t', '--topology', default="eq_FF1200.gro", help='Topology file (e.g., .gro, .pdb)')
    parser.add_argument('-x', '--trajectory', default="eq_FF1200.xtc", help='Trajectory file (e.g., .xtc, .trr)')
    parser.add_argument('-s', '--selection', default='protein', help='Bead selection string for peptides')
    parser.add_argument('-o', '--output', default='tfi_results', help='Output directory for results')
    parser.add_argument('--min_tube_size', type=int, default=MIN_TUBE_SIZE, help='Minimum number of atoms to consider a tube')
    parser.add_argument('--first', type=int, default=3024, help='Only analyze the first N frames (default is all frames)')
    parser.add_argument('--last', type=int, default=3025, help='Only analyze the last N frames (default is all frames)')
    parser.add_argument('--skip', type=int, default=1, help='Process every nth frame (default is every frame)')
    args = parser.parse_args()
    return args

def ensure_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# def center_and_wrap_trajectory(universe, selection_string):
#     """
#     Center and wrap trajectory while preserving molecular connectivity.
#     Specialized for MARTINI CG dipeptides.
#     """
#     selection = universe.select_atoms(selection_string)
#     all_atoms = universe.atoms
#     box = universe.dimensions[:3]
#     box_center = box / 2

#     # Use existing identify_aggregates function to find connected components
#     aggregates = identify_aggregates(universe, selection_string, min_aggregate_size=1)

#     # Convert aggregates to list and sort by size
#     aggregates = sorted(aggregates, key=len, reverse=True)

#     if not aggregates:
#         return universe

#     # Get positions of largest aggregate
#     largest_aggregate = aggregates[0]
#     largest_positions = selection.atoms[largest_aggregate].positions
#     com = np.mean(largest_positions, axis=0)

#     # Calculate shift to center the largest aggregate
#     shift = box_center - com

#     # Apply shift to all atoms
#     all_atoms.positions += shift

#     # Handle each aggregate for PBC
#     for aggregate_indices in aggregates:
#         aggregate_atoms = selection.atoms[aggregate_indices]
#         positions = aggregate_atoms.positions

#         # Check if aggregate spans PBC
#         spans_pbc = False
#         for i in range(len(positions)):
#             for j in range(i+1, len(positions)):
#                 diff = positions[i] - positions[j]
#                 if np.any(np.abs(diff) > box/2):
#                     spans_pbc = True
#                     break
#             if spans_pbc:
#                 break

#         if spans_pbc:
#             # Unwrap this aggregate
#             ref_pos = positions[0]
#             for i in range(1, len(positions)):
#                 diff = positions[i] - ref_pos
#                 for dim in range(3):
#                     if diff[dim] > box[dim]/2:
#                         positions[i] -= box[dim]
#                     elif diff[dim] < -box[dim]/2:
#                         positions[i] += box[dim]

#             # Update positions
#             aggregate_atoms.positions = positions

#     # Final wrap to primary unit cell
#     wrapped_positions = np.copy(all_atoms.positions)
#     wrapped_positions -= np.floor(wrapped_positions/box)*box
#     all_atoms.positions = wrapped_positions

#     return universe

# def visualize_centered_trajectory(universe, selection_string, frame, output_dir, traj_id=None):
#     """Save centered trajectory visualization as PNG images and PDB files."""
#     import os
#     import subprocess

#     vis_dir = os.path.join(output_dir, 'trajectory_visualizations')
#     os.makedirs(vis_dir, exist_ok=True)

#     # Get timestep information
#     timestep = universe.trajectory[frame].time  # Get actual simulation time

#     traj_tag = f"t{traj_id}_" if traj_id else ""
#     pdb_file = os.path.join(vis_dir, f"trajectory_{traj_tag}frame_{frame}_time_{timestep:.1f}ps.pdb")
#     png_file = os.path.join(vis_dir, f"trajectory_{traj_tag}frame_{frame}_time_{timestep:.1f}ps.png")
#     vmd_script = os.path.join(vis_dir, f"render_{traj_tag}frame_{frame}.tcl")

#     try:
#         # Move to the correct frame
#         universe.trajectory[frame]

#         # Use already centered universe
#         selection = universe.select_atoms(selection_string)
#         selection.write(pdb_file)
#         print(f"Saved PDB file to {pdb_file}")

#         # Create VMD script with modified settings
#         with open(vmd_script, 'w') as f:
#             f.write(f"""
# # Load molecule
# mol new {pdb_file} type pdb waitfor all

# # Set display size
# display resize 800 600

# # Set representation
# mol delrep 0 top
# mol representation VDW 1.0 12.0
# mol color Name
# mol material Opaque
# mol addrep top

# # Basic display settings
# display projection Orthographic
# display depthcue off
# display shadows off
# display ambientocclusion off
# axes location off
# color Display Background white

# # Center view
# mol center top
# display resetview
# rotate y by 90
# rotate x by 90

# # Force display update
# display update
# after idle

# # Wait for rendering
# after 1000

# # Render with snapshot
# render snapshot {png_file} 800 600 false %

# quit
# """)
#         print(f"Created VMD script at {vmd_script}")

#         # Set environment variables for software rendering
#         env = os.environ.copy()
#         env['LIBGL_ALWAYS_SOFTWARE'] = '1'
#         env['VMDNOCUDA'] = '1'
#         env['VMDNOOPTIX'] = '1'

#         # Run VMD with software OpenGL
#         vmd_command = f"vmd -dispdev opengl -size 800 600 -eofexit < {vmd_script}"
#         result = subprocess.run(vmd_command, shell=True, capture_output=True, text=True, env=env)

#         if result.returncode != 0:
#             print(f"VMD Output: {result.stdout}")
#             print(f"VMD Error: {result.stderr}")
#             raise RuntimeError(f"VMD failed: {result.stderr}")

#         if not os.path.exists(png_file):
#             raise RuntimeError("PNG file was not created")

#         if os.path.getsize(png_file) == 0:
#             raise RuntimeError("PNG file is empty")

#         print(f"Successfully saved PNG file to {png_file}")
#         logging.info(f"Saved visualization to {png_file}")

#     except Exception as e:
#         logging.error(f"Failed to visualize trajectory: {str(e)}")
#         print(f"Error: {str(e)}")
#     finally:
#         if os.path.exists(vmd_script):
#             os.remove(vmd_script)

def identify_aggregates(universe, selection_string, min_aggregate_size=20, output_csv=None):
    print("Selecting atoms based on the selection string...")
    selection = universe.select_atoms(selection_string)
    print(f"Number of selected atoms: {len(selection)}")

    print("Calculating positions of selected atoms...")
    positions = selection.positions
    print(f"Positions shape: {positions.shape}")

    print("Calculating distance matrix...")
    distance_matrix = cdist(positions, positions)
    print(f"Distance matrix shape: {distance_matrix.shape}")

    print("Creating adjacency matrix with distance cutoff...")
    adjacency_matrix = distance_matrix < 7.0  # Distance cutoff in Å
    np.fill_diagonal(adjacency_matrix, 0)
    print(f"Adjacency matrix:\n{adjacency_matrix}")

    print("Finding connected components...")
    labels, num_labels = connected_components(adjacency_matrix)
    print(f"Number of connected components: {num_labels}")

    print("Grouping atoms into aggregates...")
    aggregates = defaultdict(list)
    for idx, label_id in enumerate(labels):
        aggregates[label_id].append(selection.atoms[idx].index)

    # Filter aggregates by minimum size
    filtered_aggregates = {k: v for k, v in aggregates.items() if len(v) >= min_aggregate_size}
    print(f"Number of aggregates found: {len(filtered_aggregates)}")

    # Save aggregates to CSV if output_csv is specified
    if output_csv:
        with open(output_csv, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['AggregateID', 'AtomIndices'])
            for aggregate_id, atom_indices in filtered_aggregates.items():
                csvwriter.writerow([aggregate_id, ','.join(map(str, atom_indices))])
        print(f"Aggregates saved to {output_csv}")

    return filtered_aggregates.values()

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

def analyze_aggregate(aggregate_atoms, frame_number, args):
    # Initialize results with frame number first
    results = {
        'frame': frame_number,
        'aggregate_size': len(aggregate_atoms),
        'is_tube': False  # default value
    }

    # Early return for small aggregates
    if len(aggregate_atoms) < args.min_tube_size:
        logging.debug(f"Frame {frame_number}: Aggregate too small to be a tube (size={len(aggregate_atoms)}).")
        return results

    # Rest of analysis...
    positions = aggregate_atoms.positions

    tube_segment_ratio = segment_based_analysis(positions, SEGMENT_LENGTH, STEP_SIZE)
    radial_std, angular_uniformity, r, theta, z, principal_axis = perform_cylindrical_analysis(positions)
    density, bin_edges = compute_radial_density(r)
    hollow = is_hollow_tube(density, bin_edges)
    asphericity, ratio = compute_shape_anisotropy(positions)

    # Update results with all metrics
    results.update({
        'radial_std': radial_std,
        'angular_uniformity': angular_uniformity,
        'tube_segment_ratio': tube_segment_ratio,
        'hollow': hollow,
        'asphericity': asphericity,
        'eigenvalue_ratio': ratio,
        'is_tube': (
            tube_segment_ratio >= 0.5 and
            radial_std < RADIAL_THRESHOLD and
            angular_uniformity > ANGULAR_UNIFORMITY_THRESHOLD and
            hollow and
            asphericity > ASPHERICITY_THRESHOLD and
            ratio < RATIO_THRESHOLD
        )
    })

    return results
def main():
    args = parse_arguments()
    ensure_output_directory(args.output)

    # Set up logging
    timestamp = datetime.now().strftime("%m%d_%H%M")
    log_filename = os.path.join(args.output, f"tfi_{timestamp}.log")
    logging.basicConfig(filename=log_filename, level=logging.DEBUG, format='%(message)s')
    print("Loading trajectory data...")
    u = mda.Universe(args.topology, args.trajectory)
    # u = center_and_wrap_trajectory(u, args.selection)

    # # Move to the specified frame and visualize
    # u.trajectory[args.first]  # Move to the correct frame
    # visualize_centered_trajectory(u, args.selection, args.first, args.output)

    selection_string = args.selection
    n_frames = len(u.trajectory)
    print(f"Total frames in trajectory: {n_frames}")
    start_frame = 0  # Default start is the first frame
    end_frame = n_frames  # Default end is the total number of frames

    if args.first is not None:
        start_frame = max(0, args.first)  # Start from the specified first frame or 0

    if args.last is not None:
        end_frame = min(n_frames, args.last)  # End at the specified last frame or the total number of frames

    print(f"Analyzing frames from {start_frame} to {end_frame}, skipping every {args.skip} frames")

    # Initialize variables for analysis
    tube_records = defaultdict(list)  # {tube_id: [frame_numbers]}
    tube_id_counter = 0
    frame_results = []

    # Analyze each frame
    print("Analyzing frames for tube formation...")

    for frame_number, ts in enumerate(u.trajectory[start_frame:end_frame:args.skip]):
        actual_frame_number = start_frame + frame_number * args.skip  # Track the actual frame number
        print(f"Processing frame {actual_frame_number + 1}/{n_frames}...")  # Debug print for each processed frame
        logging.debug(f"Processing frame {actual_frame_number + 1}/{n_frames}...")

        aggregates = identify_aggregates(u, selection_string, min_aggregate_size=20, output_csv=f'{args.output}/aggregates_{timestamp}.csv')
        for aggregate in aggregates:
            aggregate_atoms = u.select_atoms(selection_string)[aggregate]
            results = analyze_aggregate(aggregate_atoms, actual_frame_number, args)
            if results.get('is_tube'):
                tube_id = f"tube_{tube_id_counter}"
                tube_id_counter += 1
                tube_records[tube_id].append(actual_frame_number)
            frame_results.append(results)

    # Time-resolved analysis
    tube_lifetimes = analyze_tube_lifetimes(tube_records)
    save_tube_lifetimes(tube_lifetimes, args.output, timestamp)
    save_frame_results(frame_results, args.output, timestamp)
    # plot_tube_lifetimes(tube_lifetimes, args.output)

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

def save_tube_lifetimes(tube_lifetimes, output_dir, timestamp):
    """
    Save tube lifetimes data to a file.
    """
    output_file = os.path.join(output_dir, f'tube_lifetimes_{timestamp}.csv')
    with open(output_file, 'w') as f:
        f.write('TubeID,Lifetime\n')
        for tube_id, lifetime in tube_lifetimes.items():
            f.write(f"{tube_id},{lifetime}\n")
    print(f"Tube lifetimes data saved to {output_file}")

def save_frame_results(frame_results, output_dir, timestamp):
    """Save per-frame analysis results to a file."""
    output_file = os.path.join(output_dir, f'tfi_frame_results_{timestamp}.csv')
    with open(output_file, 'w') as f:
        headers = ['Frame', 'AggregateSize', 'RadialStd', 'AngularUniformity',
                  'TubeSegmentRatio', 'Hollow', 'Asphericity', 'EigenvalueRatio', 'IsTube']
        f.write(','.join(headers) + '\n')
        for result in frame_results:
            f.write(f"{result['frame']},{result['aggregate_size']},"
                   f"{result.get('radial_std', 0):.3f},{result.get('angular_uniformity', 0):.3f},"
                   f"{result.get('tube_segment_ratio', 0):.3f},{int(result.get('hollow', False))},"
                   f"{result.get('asphericity', 0):.3f},{result.get('eigenvalue_ratio', 0):.3f},"
                   f"{int(result.get('is_tube', False))}\n")
            logging.debug(f"Wrote result to CSV: Frame {result['frame']}, "
                          f"Size {result['aggregate_size']}, Is tube: {result['is_tube']}")
    print(f"Per-frame results saved to {output_file}")

# # def plot_tube_lifetimes(tube_lifetimes, output_dir):
#     """
#     Plot the distribution of tube lifetimes.
#     """
#     lifetimes = list(tube_lifetimes.values())
#     plt.figure()
#     plt.hist(lifetimes, bins=range(1, max(lifetimes)+2), align='left')
#     plt.xlabel('Lifetime (frames)')
#     plt.ylabel('Number of Tubes')
#     plt.title('Distribution of Tube Lifetimes')
#     plt.savefig(os.path.join(output_dir, 'tube_lifetimes_distribution.png'))
#     plt.close()
#     print("Tube lifetimes distribution plot saved.")

if __name__ == '__main__':
    main()
