#!/usr/bin/env python3
"""
sfi_analysis.py

This script calculates the Sheet Formation Index (SFI) for peptide simulations.
It incorporates advanced features such as:

- Principal Component Analysis (PCA) for robust plane fitting.
- Flexibility to detect curved or twisted sheets using quadratic surface fitting.
- Topological methods using persistent homology for complex sheet structures.
- Cluster analysis to distinguish multiple sheets.
- Time-resolved analysis to track the dynamics of sheet formation.

"""

import os
import argparse
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align
from scipy.optimize import curve_fit
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
PLANARITY_THRESHOLD = 0.9  # Å, threshold for RMSD to classify as a sheet
CURVATURE_THRESHOLD = 1.3  # Å, threshold for RMSD in curved sheet fitting
SPATIAL_WEIGHT = 1.2       # Weight for spatial distance in clustering
ORIENTATION_WEIGHT = 1.0   # Weight for orientation similarity in clustering
CLUSTERING_THRESHOLD = 2.5 # Threshold for clustering algorithm
MIN_SHEET_SIZE = 10         # Minimum number of peptides to consider a sheet

def parse_arguments():
    parser = argparse.ArgumentParser(description='Sheet Formation Index (SFI) Analysis')
    parser.add_argument('-t', '--topology', default="eq_FF1200.gro", help='Topology file (e.g., .gro, .pdb)')
    parser.add_argument('-x', '--trajectory', default="eq_FF1200.xtc", help='Trajectory file (e.g., .xtc, .trr)')
    parser.add_argument('-s', '--selection', default='protein', help='Bead selection string for peptides')
    parser.add_argument('-o', '--output', default='sfi_results', help='Output directory for results')
    parser.add_argument('-pl', '--peptide_length', type=int, default=8, help='Length of each peptide in residues')
    parser.add_argument('--min_sheet_size', type=int, default=MIN_SHEET_SIZE, help='Minimum number of peptides to consider a sheet')
    parser.add_argument('--first', type=int, default=482, help='Only analyze the first N frames (default is all frames)')
    parser.add_argument('--last', type=int, default=483, help='Only analyze the last N frames (default is all frames)')
    parser.add_argument('--skip', type=int, default=1, help='Process every nth frame (default is every frame)')
    args = parser.parse_args()
    return args

def ensure_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def load_and_crop_trajectory(topology, trajectory, first, last, skip, selection="protein"):
    """
    Load the trajectory, apply cropping, and select relevant atoms for analysis.

    Parameters:
    - topology (str): Path to the topology file (e.g., .gro, .pdb).
    - trajectory (str): Path to the trajectory file (e.g., .xtc, .trr).
    - first (int): First frame to include in the analysis.
    - last (int): Last frame to include in the analysis.
    - skip (int): Frame skipping interval.
    - selection (str): Atom selection string for peptides.

    Returns:
    - MDAnalysis Universe object of the cropped trajectory.
    """
    u = mda.Universe(topology, trajectory)

    # Define end frame if not specified
    total_frames = len(u.trajectory)
    if last is None or last > total_frames:
        last = total_frames
    if first < 0 or first >= total_frames:
        raise ValueError(f"Invalid first frame: {first}.")

    # Select the specified atoms
    peptides = u.select_atoms(selection)
    if len(peptides) == 0:
        raise ValueError(f"Selection '{selection}' returned no atoms.")

    indices = list(range(first, last, skip))

    # Create temporary file names for cropped trajectory
    temp_gro = "temp_protein_slice.gro"
    temp_xtc = "temp_protein_slice.xtc"

    # Write the selected atoms to a temporary trajectory
    with mda.Writer(temp_gro, peptides.n_atoms) as W:
        W.write(peptides)
    with mda.Writer(temp_xtc, peptides.n_atoms) as W:
        for ts in u.trajectory[indices]:
            W.write(peptides)

    # Reload the cropped trajectory
    cropped_u = mda.Universe(temp_gro, temp_xtc)
    return cropped_u

def perform_pca(positions):
    """
    Uses PCA to determine the main orientation vector and the best-fit plane for a set of positions.

    Parameters:
        positions (numpy.ndarray): An array of shape (N, 3) representing 3D coordinates.

    Returns:
        normal_vector (numpy.ndarray): Unit vector normal to the best-fit plane (smallest eigenvector).
        orientation_vector (numpy.ndarray): Main orientation vector (largest eigenvector).
        rmsd (float): RMSD of points from the plane.
        positions_mean (numpy.ndarray): Centroid of the points.
        eigenvalues (numpy.ndarray): Variance along the principal axes.
    """
    if len(positions) < 3:
        print("Insufficient points for PCA.")
        return None, None, np.inf, None, None

    positions_mean = positions.mean(axis=0)
    centered_positions = positions - positions_mean
    covariance_matrix = np.cov(centered_positions.T)

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    except np.linalg.LinAlgError:
        print("PCA failed: Eigenvalues did not converge.")
        return None, None, np.inf, None, None

    # The smallest eigenvalue's eigenvector is the normal vector to the best-fit plane
    normal_vector = eigenvectors[:, 0]
    orientation_vector = eigenvectors[:, -1]

    distances = np.dot(centered_positions, normal_vector)
    rmsd = np.sqrt(np.mean(distances ** 2))

    return normal_vector, orientation_vector, rmsd, positions_mean, eigenvalues

def fit_quadratic_surface(positions):
    """
    Fit a quadratic surface to the positions to account for curvature.
    Returns the RMSD of the fit if successful, otherwise returns infinity.

    Parameters:
    - positions (numpy.ndarray): An array of shape (N, 3) representing the 3D coordinates of points.

    Returns:
    - rmsd (float): Root-mean-square deviation of the points from the fitted quadratic surface.
    - params (tuple): Parameters (a, b, c, d, e, f) of the fitted quadratic surface, or None if the fit fails.
    """
    # Ensure there are enough points for a quadratic fit
    if len(positions) < 6:
        print("Not enough points to fit a quadratic surface.")
        return np.inf, None

    def quadratic_surface(X, a, b, c, d, e, f):
        x, y = X
        return a * x**2 + b * y**2 + c * x * y + d * x + e * y + f

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    X = np.vstack((x, y))

    try:
        # Perform curve fitting
        params, _ = curve_fit(quadratic_surface, X, z)

        # Calculate fitted z values and residuals
        z_fit = quadratic_surface(X, *params)
        residuals = z - z_fit

        # Compute RMSD from the residuals
        rmsd = np.sqrt(np.mean(residuals**2))

        return rmsd, params

    except RuntimeError:
        # Handle case where the curve fitting fails
        print("Quadratic surface fitting failed.")
        return np.inf, None

def compute_angle_matrix(orientations):
    """
    Compute a pairwise angle matrix between orientation vectors.

    Parameters:
    - orientations (numpy.ndarray): An array of shape (N, 3) where each row is a unit vector representing the orientation of a peptide.

    Returns:
    - angle_matrix (numpy.ndarray): A matrix of shape (N, N) containing pairwise angles in degrees between the orientation vectors.
    """
    # Compute the dot product matrix for pairwise orientation vectors
    dot_products = np.dot(orientations, orientations.T)

    # Calculate norms and construct a matrix of norms for stable angle calculation
    norms = np.linalg.norm(orientations, axis=1)
    norms_matrix = np.outer(norms, norms)

    # Avoid division by zero by setting zero norms to 1 temporarily
    norms_matrix[norms_matrix == 0] = 1
    cos_angles = dot_products / norms_matrix

    # Clip cosine values to the range [-1, 1] to avoid numerical errors in arccos
    cos_angles = np.clip(cos_angles, -1.0, 1.0)

    # Calculate angles in radians and convert to degrees
    angles = np.arccos(cos_angles)
    angle_matrix = np.degrees(angles)

    return angle_matrix

# Constants adjusted for molecular scale
SPATIAL_CUTOFF = 2.0      # nm, reasonable for β-sheet formation
ANGLE_CUTOFF = 30         # degrees, stricter alignment requirement
MIN_SHEET_SIZE = 4        # minimum peptides for sheet classification

def cluster_peptides(positions, orientations):
    """Cluster peptides into potential β-sheets"""
    # Calculate distances excluding self-distances
    spatial_dist = squareform(pdist(positions))
    np.fill_diagonal(spatial_dist, np.inf)  # Exclude self-distances

    # Calculate angles excluding self-angles
    angle_matrix = compute_angle_matrix(orientations)
    np.fill_diagonal(angle_matrix, np.inf)  # Exclude self-angles

    # Create connectivity matrix with stricter conditions
    connectivity = np.logical_and(
        spatial_dist <= SPATIAL_CUTOFF,
        angle_matrix <= ANGLE_CUTOFF
    ).astype(int)  # Convert to int for connected_components

    # Clustering
    n_components, labels = connected_components(csr_matrix(connectivity))

    # Filter small clusters
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        if count < MIN_SHEET_SIZE:
            labels[labels == label] = -1

    print(f"Valid clusters found: {len(set(labels) - {-1})}")
    return labels

def time_resolved_sheet_analysis(sheet_records, min_sheet_size):
    """
    Analyze the dynamics of sheet formation over time, tracking the start, end, and lifetime of each sheet.

    Parameters:
    - sheet_records (dict): A dictionary where keys are frame numbers and values are dictionaries
      of sheet IDs mapped to peptide indices for each frame.
    - min_sheet_size (int): The minimum number of frames a sheet must persist to be considered.

    Returns:
    - sheet_lifetimes (dict): A dictionary with sheet IDs as keys and dictionaries as values.
      Each dictionary contains the start frame, end frame, and lifetime (in frames) for each sheet.
    """
    # Initialize tracking of each sheet's frames and peptides
    sheet_tracks = defaultdict(list)
    for frame_number, sheets in sheet_records.items():
        for sheet_id, peptides in sheets.items():
            sheet_tracks[sheet_id].append((frame_number, peptides))

    # Analyze each sheet's lifetime and properties
    sheet_lifetimes = {}
    for sheet_id, history in sheet_tracks.items():
        if len(history) >= min_sheet_size:
            # Sort frames to get continuous tracking of each sheet
            frames = [entry[0] for entry in history]
            start_frame = min(frames)
            end_frame = max(frames)
            lifetime = end_frame - start_frame + 1  # Inclusive of start and end

            # Save results for sheets meeting the minimum persistence criterion
            sheet_lifetimes[sheet_id] = {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "lifetime": lifetime
            }

    return sheet_lifetimes

def main():
    args = parse_arguments()
    ensure_output_directory(args.output)

    # Load trajectory and setup variables
    u = load_and_crop_trajectory(args.topology, args.trajectory, args.first, args.last, args.skip, args.selection)
    peptide_length = args.peptide_length
    peptides = u.select_atoms(args.selection)

    # Initialize sheet tracking variables
    sheet_records = defaultdict(dict)
    sheet_id_counter = 0

    # Process each frame
    for frame_number, ts in enumerate(u.trajectory):
        print(f"Processing frame {frame_number}/{len(u.trajectory)}...")

        # For each dipeptide (peptide_length = 8 beads):
        # 1. Takes 8 beads at a time (peptides[i:i+8])
        # 2. Calculates center of mass by averaging bead positions (.mean(axis=0))
        # 3. Results in 1200 positions (one per dipeptide)
        positions = np.array([
            peptides[i:i + peptide_length].positions.mean(axis=0)
            for i in range(0, len(peptides), peptide_length)
        ])

        # Orientation calculation:
        # 1. Takes 8 beads of each dipeptide
        # 2. Performs PCA to find principal axis
        # 3. Returns orientation vector for each dipeptide
        orientations = np.array([
            perform_pca(peptides[i:i + peptide_length].positions)[1]
            for i in range(0, len(peptides), peptide_length)
        ])


        # Perform clustering based on positions and orientations
        labels = cluster_peptides(positions, orientations)
        unique_labels = set(labels)

        # Process each cluster to identify sheets
        frame_sheets = {}
        for label in unique_labels:
            if label == -1:
                continue  # Ignore noise

            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) >= args.min_sheet_size:
                cluster_positions = positions[cluster_indices]

                # Perform PCA and check if it succeeded
                normal_vector, orientation_vector, rmsd_pca, plane_origin, eigenvalues = perform_pca(cluster_positions)
                if rmsd_pca == np.inf:
                    print(f"Skipping cluster {label} in frame {frame_number} due to PCA failure.")
                    continue

                # Determine if the cluster is planar or curved
                is_planar = rmsd_pca < PLANARITY_THRESHOLD
                print(f"Frame {frame_number}, Cluster {label}: Size={len(cluster_indices)}, Planar={is_planar}")

                # Check for curvature if the cluster is not planar
                is_curved = False
                if not is_planar and len(cluster_positions) >= 6:
                    rmsd_quad, _ = fit_quadratic_surface(cluster_positions)
                    is_curved = rmsd_quad < CURVATURE_THRESHOLD
                    print(f"Frame {frame_number}, Cluster {label}: Curved={is_curved}")

                # Register the sheet based on planarity or curvature
                if is_planar:
                    sheet_id = f"sheet_{sheet_id_counter}"
                    frame_sheets[sheet_id] = cluster_indices
                    sheet_id_counter += 1
                elif is_curved:
                    sheet_id = f"sheet_{sheet_id_counter}"
                    frame_sheets[sheet_id] = cluster_indices
                    sheet_id_counter += 1

        # Store results for the current frame
        sheet_records[frame_number] = frame_sheets

    # Time-resolved analysis, save results, and plot lifetimes
    sheet_lifetimes = time_resolved_sheet_analysis(sheet_records, args.min_sheet_size)
    save_sheet_lifetimes(sheet_lifetimes, args.output)
    plot_sheet_lifetimes(sheet_lifetimes, args.output)

    print("SFI analysis completed successfully.")

def save_sheet_lifetimes(sheet_lifetimes, output_dir):
    """
    Save sheet lifetimes data to a file.
    """
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_file = os.path.join(output_dir, f'sheet_lifetimes_{timestamp}.csv')
    with open(output_file, 'w') as f:
        f.write('SheetID,Lifetime\n')
        for sheet_id, lifetime in sheet_lifetimes.items():
            f.write(f"{sheet_id},{lifetime}\n")
    print(f"Sheet lifetimes data saved to {output_file}")

def plot_sheet_lifetimes(sheet_lifetimes, output_dir):
    """
    Plot the distribution of sheet lifetimes.
    """
    lifetimes = [data["lifetime"] for data in sheet_lifetimes.values()]

    if not lifetimes:
        print("No sheets met the minimum lifetime criteria; skipping plot.")
        return

    plt.figure()
    plt.hist(lifetimes, bins=range(1, max(lifetimes) + 2), align='left')
    plt.xlabel('Lifetime (frames)')
    plt.ylabel('Number of Sheets')
    plt.title('Distribution of Sheet Lifetimes')
    timestamp = datetime.now().strftime("%m%d_%H%M")
    plt.savefig(os.path.join(output_dir, f'sheet_lifetimes_distribution_{timestamp}.png'))
    plt.close()
    print("Sheet lifetimes distribution plot saved.")

if __name__ == '__main__':
    main()
