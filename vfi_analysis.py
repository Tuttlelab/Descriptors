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
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.cluster.hierarchy import fcluster, linkage
import csv

import warnings
# Remove the import that causes the deprecation warning
# from Bio import BiopythonDeprecationWarning
# Modify the warnings filter to ignore the BiopythonDeprecationWarning
warnings.filterwarnings("ignore", ".*BiopythonDeprecationWarning.*")
warnings.filterwarnings("ignore", category=UserWarning)

# Constants
DENSITY_BINS = 50  # Reduced from 100
SPHERICITY_THRESHOLD = 0.5  # Increased from 0.4 for more reliable detection
HOLLOWNESS_THRESHOLD = 0.05  # More sensitive threshold
MIN_VESICLE_SIZE = 30  # Reduced from 50
VOXEL_SIZE = 0.5  # Increased from 0.15 for faster calculation
ACYLINDRICITY_THRESHOLD = 2.5 # adjacency_matrix = distance_matrix < acylindricity for vesicle classification
MIN_GRID_SIZE = 10    # Minimum grid dimension for hollowness calculation
MIN_VOLUME = 500      # Minimum volume in voxels for vesicle detection
MIN_ATOMS_SPHERICITY = 10  # Minimum atoms needed for meaningful sphericity
PERFECT_SPHERE_RATIO = (np.pi**(1/3)) * 6**(2/3)  # Pre-calculate constant term
MIN_VOLUME_SPHERICITY = 100.0  # Minimum volume in nm³
MIN_SPHERICITY_THRESHOLD = 0.2  # Minimum meaningful sphericity value
MAX_COMPONENTS = 5000  # Safety limit for number of components
MIN_COMPONENT_SIZE = 5  # Minimum atoms per component
MAX_RADIAL_BINS = 200  # Maximum number of bins for density calculation
ASPHERICITY_THRESHOLD = 1.1  # Maximum asphericity for vesicle classification
CSV_HEADERS = ['Frame', 'Peptides', 'vesicle_count', 'total_peptides_in_vesicles', 'avg_vesicle_size']

def parse_arguments():
    parser = argparse.ArgumentParser(description='Vesicle Formation Index (VFI) Analysis')
    parser.add_argument('-t', '--topology', required=True, help='Topology file (e.g., .gro, .pdb)')
    parser.add_argument('-x', '--trajectory', required=True, help='Trajectory file (e.g., .xtc, .trr)')
    parser.add_argument('-o', '--output', default='vfi_results', help='Output directory for results')
    parser.add_argument('--min_vesicle_size', type=int, default=MIN_VESICLE_SIZE,
                        help='Minimum number of atoms to consider an aggregate as a vesicle')
    parser.add_argument('--first', type=int, default=0, help='First frame to analyze')
    parser.add_argument('--last', type=int, default=None, help='Last frame to analyze')
    parser.add_argument('--skip', type=int, default=1, help='Process every nth frame')
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
    print()
    print(f"Logging to {log_filename}")

'''
def load_and_crop_trajectory():
    # Comment out this function as we'll use direct loading
    # ...existing function code...
'''

'''
def center_and_wrap_trajectory():
    # Comment out this function as files are pre-processed
    # ...existing function code...
'''

def identify_aggregates(universe):
    """Identify aggregates using hierarchical clustering"""
    positions = universe.atoms.positions

    if len(positions) == 0:
        return [], []

    # Perform hierarchical clustering
    linkage_matrix = linkage(positions, method='single', metric='euclidean')
    # Cut the dendrogram at 6.0 Angstroms distance
    labels = fcluster(linkage_matrix, t=6.0, criterion='distance') - 1

    # Count unique clusters
    unique_clusters = np.unique(labels)

    # Group atoms by cluster
    aggregates = []
    aggregate_indices = []
    for cluster_id in unique_clusters:
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) >= MIN_VESICLE_SIZE:
            ag_atoms = universe.atoms[cluster_indices]
            aggregates.append(ag_atoms)
            aggregate_indices.append(cluster_indices)

    return aggregates, aggregate_indices

def connected_components(adjacency_matrix):
    """
    Find connected components in adjacency matrix using depth-first search.

    Parameters:
        adjacency_matrix (np.ndarray): Boolean matrix where True indicates connection

    Returns:
        tuple: (labels array, number of components)

    Raises:
        ValueError: If matrix is invalid or too large
    """
    if not isinstance(adjacency_matrix, np.ndarray) or adjacency_matrix.ndim != 2:
        raise ValueError("Input must be 2D numpy array")

    n_nodes = adjacency_matrix.shape[0]
    logging.debug(f"Finding components for {n_nodes} nodes")

    visited = np.zeros(n_nodes, dtype=bool)
    labels = np.full(n_nodes, -1, dtype=int)
    label_id = 0
    component_sizes = []

    for node in range(n_nodes):
        if not visited[node]:
            stack = [node]
            component_size = 0
            while stack:
                current = stack.pop()
                if not visited[current]:
                    visited[current] = True
                    labels[current] = label_id
                    component_size += 1
                    neighbors = np.where(adjacency_matrix[current])[0]
                    stack.extend(neighbors)

            if component_size >= MIN_COMPONENT_SIZE:
                component_sizes.append(component_size)
                label_id += 1

            if label_id > MAX_COMPONENTS:
                raise ValueError(f"Too many components (>{MAX_COMPONENTS})")

    logging.debug(f"Found {label_id} components with sizes: {component_sizes}")
    return labels, label_id

def compute_radial_density(positions, com, num_bins):
    """
    Calculate radial density profile around center of mass.

    Parameters:
        positions (np.ndarray): Atomic positions
        com (np.ndarray): Center of mass coordinates
        num_bins (int): Number of radial bins

    Returns:
        tuple: (density array, bin edges array)
    """
        # Input validation
    if len(positions) == 0:
        logging.error("Empty positions array in radial density calculation")
        return np.zeros(num_bins), np.zeros(num_bins + 1)

    if num_bins > MAX_RADIAL_BINS:
        logging.warning(f"Reducing bins from {num_bins} to {MAX_RADIAL_BINS}")
        num_bins = MAX_RADIAL_BINS

    try:
        # Calculate distances from COM
        distances = np.linalg.norm(positions - com, axis=1)

        # Validate distances
        if len(distances) == 0:
            logging.error("No valid distances calculated")
            return np.zeros(num_bins), np.zeros(num_bins + 1)

        max_distance = distances.max()

        if max_distance == 0:
                logging.error("Zero maximum distance in radial calculation")
                return np.zeros(num_bins), np.zeros(num_bins + 1)

        # Create histogram
        bins = np.linspace(0, max_distance, num_bins)
        density, bin_edges = np.histogram(distances, bins=bins, density=True)
        return density, bin_edges

    except Exception as e:
        logging.error(f"Density calculation failed: {str(e)}")
        return np.zeros(num_bins), np.zeros(num_bins + 1)

def is_hollow(density, bin_edges, window_size=7, hollow_ratio=HOLLOWNESS_THRESHOLD):
    """
    Detect hollow structures using radial density profile.

    Parameters:
    - density: radial density profile
    - bin_edges: radial distance bins
    - window_size: smoothing window (odd number)
    - hollow_ratio: maximum core/shell density ratio for hollow classification
    """
    from scipy.signal import argrelextrema

    # Wider smoothing window
    kernel = np.ones(window_size)/window_size
    density_smooth = np.convolve(density, kernel, mode='same')

    # Find extrema
    maxima = argrelextrema(density_smooth, np.greater)[0]
    minima = argrelextrema(density_smooth, np.less)[0]

    if len(maxima) > 0 and len(minima) > 0:
        shell_peak = density_smooth[maxima[0]]
        core_min = density_smooth[minima[0]]

        # Check if minimum occurs before maximum (core before shell)
        if minima[0] > maxima[0]:
            return False

        # Stricter hollow criterion
        if core_min < shell_peak * hollow_ratio:
            return True

    return False

def compute_sphericity(positions):
    """
    Calculate sphericity using convex hull.
    Sphericity = (π^(1/3)) * (6V)^(2/3) / A
    where V is volume and A is surface area.
    Perfect sphere = 1.0
    """
    # Validate input size
    if len(positions) < MIN_ATOMS_SPHERICITY:
        logging.debug(f"Too few atoms ({len(positions)}) for sphericity calculation")
        return 0.0

    try:
        # Calculate convex hull
        hull = ConvexHull(positions)

        # Validate volume
        if hull.volume < MIN_VOLUME_SPHERICITY:
            logging.debug(f"Volume too small ({hull.volume:.2f}) for meaningful sphericity")
            return 0.0

        # Calculate sphericity
        sphericity = PERFECT_SPHERE_RATIO * (hull.volume**(2/3)) / hull.area

        # Validate result
        if sphericity < MIN_SPHERICITY_THRESHOLD:
            logging.debug(f"Calculated sphericity too low: {sphericity:.3f}")
            return 0.0

        return sphericity

    except Exception as e:
        logging.error(f"Error in sphericity calculation: {str(e)}")
        return 0.0

def compute_hollowness_ratio(positions, voxel_size=VOXEL_SIZE):
    """Calculate hollowness using both voxelization and radial density methods"""
    try:
        # 1. Voxel-based hollowness
        pos = positions - positions.min(axis=0)
        voxel_size = min(voxel_size, np.ptp(pos, axis=0).min() / 20)

        # Create 3D grid with higher resolution
        grid_size = np.ceil(np.ptp(pos, axis=0) / voxel_size).astype(int) + 4  # More padding
        grid = np.zeros(grid_size, dtype=bool)

        # Map positions to grid
        indices = (pos / voxel_size).astype(int)
        grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True

        # Use smaller structure for dilation to preserve holes better
        from scipy.ndimage import binary_dilation, binary_fill_holes
        grid = binary_dilation(grid, structure=ball(1))

        # Fill holes
        filled = binary_fill_holes(grid)

        if filled is None:
            logging.warning("binary_fill_holes returned None")
            return 0.001  # Return minimum non-zero value on error

        # Calculate voxel-based hollowness
        shell_volume = np.sum(grid)
        total_volume = np.sum(filled.astype(int))
        voxel_hollowness = 0.0
        if total_volume > shell_volume and total_volume > 0:
            voxel_hollowness = (total_volume - shell_volume) / total_volume

        # 2. Radial density-based hollowness
        com = np.mean(positions, axis=0)
        density, bin_edges = compute_radial_density(positions, com, DENSITY_BINS)

        # Analyze density profile
        if len(density) > 3:  # Ensure enough points for analysis
            max_density = np.max(density)
            if (max_density > 0):
                # Normalize density
                density = density / max_density
                # Check for hollow core (lower density in center)
                core_density = np.mean(density[:len(density)//4])  # Inner quarter
                shell_density = np.mean(density[len(density)//4:3*len(density)//4])  # Middle half
                if shell_density > core_density:
                    radial_hollowness = (shell_density - core_density) / shell_density
                else:
                    radial_hollowness = 0.0
            else:
                radial_hollowness = 0.0
        else:
            radial_hollowness = 0.0

        # Combine both methods (weighted average)
        combined_hollowness = 0.7 * voxel_hollowness + 0.3 * radial_hollowness

        # Apply gentler sigmoid scaling
        from scipy.special import expit
        scaled_ratio = expit(2 * combined_hollowness - 0.5)  # Less aggressive scaling

        return max(scaled_ratio, 0.001)  # Ensure minimum non-zero value for nearly-hollow structures

    except Exception as e:
        logging.warning(f"Hollowness calculation failed: {e}")
        return 0.001  # Return minimum non-zero value on error

def voxelize_positions(positions, voxel_size):
    """Convert atomic positions to voxel grid"""
    # Get bounding box
    min_coords = positions.min(axis=0) - voxel_size
    max_coords = positions.max(axis=0) + voxel_size

    # Create grid dimensions
    grid_shape = np.ceil((max_coords - min_coords) / voxel_size).astype(int)
    grid = np.zeros(grid_shape, dtype=bool)

    # Map positions to grid indices
    indices = np.floor((positions - min_coords) / voxel_size).astype(int)
    grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True

    return grid

def compute_shape_descriptors(positions, com):
    relative_positions = positions - com
    gyration_tensor = np.dot(relative_positions.T, relative_positions) / len(relative_positions)
    eigenvalues, _ = np.linalg.eigh(gyration_tensor)
    lambda_avg = eigenvalues.mean()
    asphericity = ((eigenvalues - lambda_avg)**2).sum() / (2 * lambda_avg**2)
    acylindricity = ((eigenvalues[1] - eigenvalues[0])**2 + (eigenvalues[2] - eigenvalues[1])**2 +
                     (eigenvalues[0] - eigenvalues[2])**2) / (2 * lambda_avg**2)
    return asphericity, acylindricity

def analyze_aggregate(aggregate_atoms, frame_number, peptide_indices):
    """Modified aggregate analysis to include peptide tracking"""
    positions = aggregate_atoms.positions
    n_atoms = len(positions)

    if n_atoms < MIN_VESICLE_SIZE:
        return {'is_vesicle': False, 'peptides': []}

    # Calculate center of mass and shape descriptors
    com = np.mean(positions, axis=0)
    sphericity = compute_sphericity(positions)
    density, bin_edges = compute_radial_density(positions, com, DENSITY_BINS)
    hollowness = compute_hollowness_ratio(positions)
    asphericity, acylindricity = compute_shape_descriptors(positions, com)

    # Determine if structure is a vesicle
    is_vesicle = (
        sphericity >= SPHERICITY_THRESHOLD and
        hollowness >= HOLLOWNESS_THRESHOLD and
        asphericity <= ASPHERICITY_THRESHOLD and
        acylindricity <= ACYLINDRICITY_THRESHOLD
    )

    return {
        'frame': frame_number,
        'size': n_atoms,
        'sphericity': sphericity,
        'hollowness': hollowness,
        'asphericity': asphericity,
        'acylindricity': acylindricity,
        'is_vesicle': is_vesicle,
        'peptides': [f'PEP{idx+1}' for idx in peptide_indices]
    }

def main():
    args = parse_arguments()
    ensure_output_directory(args.output)
    setup_logging(args.output)

    # Direct loading of pre-processed trajectory
    print()
    print("Loading trajectory...")
    print()
    u = mda.Universe(args.topology, args.trajectory)
    peptides = u.select_atoms('all')  # Use all atoms as file is pre-processed
    print(f"Loaded {len(peptides)} atoms")
    print()

    # Initialize tracking variables
    vesicle_records = defaultdict(list)
    vesicle_id_counter = 0
    frame_records = []

    # Process frames
    frames = range(args.first, args.last or len(u.trajectory), args.skip)
    for frame_number in frames:
        u.trajectory[frame_number]
        print(f"Processing frame {frame_number}...")
        print()

        # Get aggregates
        aggregates, peptide_indices = identify_aggregates(u)

        # Track vesicles and their peptides for this frame
        frame_vesicles = []
        frame_peptides = []

        for aggregate, indices in zip(aggregates, peptide_indices):
            results = analyze_aggregate(aggregate, frame_number, indices)

            if results['is_vesicle']:
                vesicle_id = f"vesicle_{vesicle_id_counter}"
                vesicle_id_counter += 1
                vesicle_records[vesicle_id].append(frame_number)
                frame_vesicles.append(results)
                frame_peptides.extend(results['peptides'])

        # Create frame record
        vesicle_count = len(frame_vesicles)
        total_peptides = sum(v['size'] for v in frame_vesicles)
        avg_vesicle_size = total_peptides / vesicle_count if vesicle_count > 0 else 0

        frame_record = {
            'Frame': frame_number,
            'Peptides': str(sorted(frame_peptides)),
            'vesicle_count': vesicle_count,
            'total_peptides_in_vesicles': total_peptides,
            'avg_vesicle_size': avg_vesicle_size
        }
        frame_records.append(frame_record)

    # Save results
    save_frame_results(frame_records, args.output)
    # ...existing analysis and plotting code...

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
    print()

def save_frame_results(frame_results, output_dir):
    """Save VFI frame results to a CSV file."""
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_file = os.path.join(output_dir, f'vfi_frame_results_{timestamp}.csv')

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for record in frame_results:
            writer.writerow(record)

    logging.info(f"VFI frame results saved to {output_file}")
    print(f"VFI frame results saved to {output_file}")
    print()

def plot_vesicle_lifetimes(vesicle_lifetimes, output_dir):
    """
    Plot the distribution of vesicle lifetimes.
    """
    if not vesicle_lifetimes:
        print("No vesicles found in the analyzed frames - skipping histogram creation")

        return

    lifetimes = list(vesicle_lifetimes.values())  # Use dictionary values
    plt.figure()
    plt.hist(lifetimes, bins=range(1, max(lifetimes)+2), align='left')
    plt.xlabel('Lifetime (frames)')
    plt.ylabel('Number of Vesicles')
    plt.title('Distribution of Vesicle Lifetimes')
    plot_path = os.path.join(output_dir, 'vesicle_lifetimes_distribution.png')
    plt.savefig(plot_path)
    plt.close()
    print("Vesicle lifetimes distribution plot saved.")


# IMPROVEMENTS:

# def visualize_vesicle(universe, aggregate_atoms, frame, output_dir, vesicle_id=None):
#     """
#     Save vesicle visualization as PNG images and PDB files for viewing in VMD.

#     Parameters:
#         universe: MDAnalysis Universe
#         aggregate_atoms: AtomGroup containing vesicle atoms
#         frame: Frame number
#         output_dir: Output directory
#         vesicle_id: Optional identifier for tracking the same vesicle across frames
#     """
#     import os
#     import subprocess
#     import warnings

#     # Create visualization subdirectory
#     vis_dir = os.path.join(output_dir, 'vesicle_visualizations')
#     os.makedirs(vis_dir, exist_ok=True)

#     # Generate unique filenames
#     vesicle_tag = f"v{vesicle_id}_" if vesicle_id else ""
#     pdb_file = os.path.join(vis_dir, f"vesicle_{vesicle_tag}frame_{frame}.pdb")
#     png_file = os.path.join(vis_dir, f"vesicle_{vesicle_tag}frame_{frame}.png")
#     vmd_script = os.path.join(vis_dir, f"render_{vesicle_tag}frame_{frame}.tcl")

#     try:
#         # Center vesicle within the box
#         com = aggregate_atoms.center_of_mass()
#         box_center = universe.dimensions[:3] / 2
#         shift = box_center - com

#         # Save centered structure
#         temp_ag = aggregate_atoms.copy()
#         temp_ag.translate(shift)
#         temp_ag.write(pdb_file)

#         # Create VMD script for rendering
#         with open(vmd_script, 'w') as f:
#             f.write(f"""
# # Load the molecule
# mol new {pdb_file} type pdb

# # Set representation
# mol delrep 0 top
# mol representation VDW 1.0 12.0
# mol color Name
# mol material Opaque
# mol addrep top

# # Display settings
# display projection Orthographic
# display depthcue off
# axes location off
# color Display Background white

# # Set display size and rendering parameters
# display resize 800 600
# display antialias on

# # Update display
# display update ui

# # Render to PNG using Snapshot renderer
# render snapshot {png_file}

# quit
# """)

#         # Set environment variable for software rendering
#         env = os.environ.copy()
#         env['LIBGL_ALWAYS_SOFTWARE'] = '1'

#         # Command to run VMD off-screen
#         vmd_command = f"vmd -dispdev ogl -e {vmd_script}"

#         # Run VMD with software rendering
#         subprocess.run(vmd_command, shell=True, check=True, env=env)

#         logging.info(f"Saved vesicle visualization to {png_file}")
#         logging.info(f"Saved vesicle structure to {pdb_file}")

#     except Exception as e:
#         logging.error(f"Failed to visualize vesicle: {str(e)}")
#     finally:
#         # Cleanup temporary files (keep the PDB file)
#         if os.path.exists(vmd_script):
#             os.remove(vmd_script)

def track_vesicle_evolution(vesicle_records):
    """Track how vesicles merge/split over time"""

def analyze_vesicle_composition(aggregate_atoms):
    """Analyze peptide arrangements in vesicles"""

def generate_summary_report(results, output_dir):
    """Create detailed PDF report with plots and statistics"""

if __name__ == '__main__':
    main()
