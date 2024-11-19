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

import logging
import os
import argparse
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform, cdist
from collections import defaultdict
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import DBSCAN

# Constants
PLANARITY_THRESHOLD = 0.9   # Å, threshold for RMSD to classify as a sheet
CURVATURE_THRESHOLD = 2.0   # Å, increased to detect more curved sheets
SPATIAL_WEIGHT = 1.2        # Weight for spatial distance in clustering
ORIENTATION_WEIGHT = 1.0    # Weight for orientation similarity in clustering
SPATIAL_CUTOFF = 15        # nm, adjusted for smaller sheet detection
ANGLE_CUTOFF = 45           # degrees, increased to allow for curvature
MIN_SHEET_SIZE = 5         # Reduced to detect smaller sheets

def parse_arguments():
    parser = argparse.ArgumentParser(description='Sheet Formation Index (SFI) Analysis')
    parser.add_argument('-t', '--topology', default="eq_FF1200.gro", help='Topology file (e.g., .gro, .pdb)')
    parser.add_argument('-x', '--trajectory', default="eq_FF1200.xtc", help='Trajectory file (e.g., .xtc, .trr)')
    parser.add_argument('-s', '--selection', default='protein', help='Bead selection string for peptides')
    parser.add_argument('-o', '--output', default='sfi_results', help='Output directory for results')
    parser.add_argument('-pl', '--peptide_length', type=int, default=8, help='Length of each peptide in residues')
    parser.add_argument('--min_sheet_size', type=int, default=MIN_SHEET_SIZE, help='Minimum number of peptides to consider a sheet')
    parser.add_argument('--first', type=int, default=482, help='First frame to analyze')
    parser.add_argument('--last', type=int, default=483, help='Last frame to analyze')
    parser.add_argument('--skip', type=int, default=1, help='Process every nth frame')
    args = parser.parse_args()
    return args

def ensure_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def load_and_crop_trajectory(topology, trajectory, first, last, skip, selection="protein"):
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
    if len(positions) < 3:
        # For less than 3 points, define default outputs
        positions_mean = positions.mean(axis=0)
        normal_vector = np.array([0, 0, 1])
        orientation_vector = np.array([1, 0, 0])
        rmsd = 0.0
        eigenvalues = np.array([0, 0, 0])
        return normal_vector, orientation_vector, rmsd, positions_mean, eigenvalues

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
    if len(positions) < 6:
        # Not enough points to fit a quadratic surface
        return np.inf, None

    def quadratic_surface(X, a, b, c, d, e, f):
        x, y = X
        return a * x**2 + b * y**2 + c * x * y + d * x + e * y + f

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    X = np.vstack((x, y))

    try:
        params, _ = curve_fit(quadratic_surface, X, z)

        z_fit = quadratic_surface(X, *params)
        residuals = z - z_fit

        rmsd = np.sqrt(np.mean(residuals**2))

        return rmsd, params

    except RuntimeError:
        # Curve fitting failed
        return np.inf, None

def compute_angle_matrix(orientations):
    dot_products = np.dot(orientations, orientations.T)
    norms = np.linalg.norm(orientations, axis=1)
    norms_matrix = np.outer(norms, norms)
    norms_matrix[norms_matrix == 0] = 1
    cos_angles = dot_products / norms_matrix
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    angles = np.degrees(np.arccos(cos_angles))
    return angles

def cluster_peptides(positions, orientations):
    spatial_dist = squareform(pdist(positions))
    np.fill_diagonal(spatial_dist, np.inf)

    angle_matrix = compute_angle_matrix(orientations)
    np.fill_diagonal(angle_matrix, np.inf)

    connectivity = np.logical_and(
        spatial_dist <= SPATIAL_CUTOFF,
        angle_matrix <= ANGLE_CUTOFF
    ).astype(int)

    n_components, labels = connected_components(csr_matrix(connectivity))

    # Adjust labels for small clusters
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        if count < MIN_SHEET_SIZE:
            labels[labels == label] = -1

    valid_clusters = set(labels) - {-1}
    print(f"Valid clusters found: {len(valid_clusters)}")
    return labels

# def cluster_peptides(positions, orientations):
#     """
#     Cluster peptides using DBSCAN based on spatial proximity and orientation similarity.
#     """
#     # Normalize orientations
#     orientations_normalized = orientations / np.linalg.norm(orientations, axis=1).reshape(-1, 1)

#     # Combine positional and orientation data
#     # You can adjust the weighting factors as needed
#     spatial_weight = 1.0
#     orientation_weight = 1.0
#     features = np.hstack((positions * spatial_weight, orientations_normalized * orientation_weight))

#     # Perform DBSCAN clustering
#     clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(features)
#     labels = clustering.labels_

#     # Adjust labels for small clusters
#     unique_labels, counts = np.unique(labels, return_counts=True)
#     for label, count in zip(unique_labels, counts):
#         if count < MIN_SHEET_SIZE:
#             labels[labels == label] = -1

#     valid_clusters = set(labels) - {-1}
#     print(f"Valid clusters found: {len(valid_clusters)}")
#     return labels

def time_resolved_sheet_analysis(sheet_records, min_sheet_frames):
    sheet_lifetimes = {}
    for sheet_id, frames in sheet_records.items():
        frames_sorted = sorted(frames)
        lifetime = len(frames)
        if lifetime >= min_sheet_frames:
            sheet_lifetimes[sheet_id] = {
                "start_frame": frames_sorted[0],
                "end_frame": frames_sorted[-1],
                "lifetime": lifetime
            }
    return sheet_lifetimes

def save_sheet_lifetimes(sheet_lifetimes, output_dir):
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_file = os.path.join(output_dir, f'sheet_lifetimes_{timestamp}.csv')
    with open(output_file, 'w') as f:
        f.write('SheetID,StartFrame,EndFrame,Lifetime\n')
        for sheet_id, data in sheet_lifetimes.items():
            f.write(f"{sheet_id},{data['start_frame']},{data['end_frame']},{data['lifetime']}\n")
    print(f"Sheet lifetimes data saved to {output_file}")

def plot_sheet_lifetimes(sheet_lifetimes, output_dir):
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

def main():
    args = parse_arguments()
    ensure_output_directory(args.output)

    # Setup logging with timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(args.output, f'sfi_analysis_{timestamp}.log')
    logging.basicConfig(
        filename=log_filename,
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s:%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Load trajectory and setup variables
    u = load_and_crop_trajectory(args.topology, args.trajectory, args.first, args.last, args.skip, args.selection)
    peptide_length = args.peptide_length
    peptides = u.select_atoms(args.selection)

    # Initialize sheet tracking variables
    sheet_records = defaultdict(list)
    sheet_id_counter = 0

    min_sheet_frames = 1  # Minimum frames a sheet must persist

    # Process each frame
    for frame_number, ts in enumerate(u.trajectory):
        print(f"Processing frame {frame_number + args.first}/{len(u.trajectory) + args.first - 1}...")

        # Calculate positions and orientations for each peptide
        positions = []
        orientations = []
        for i in range(0, len(peptides), peptide_length):
            peptide = peptides[i:i + peptide_length]
            if len(peptide) < peptide_length:
                continue
            positions.append(peptide.positions.mean(axis=0))
            _, orientation_vector, _, _, _ = perform_pca(peptide.positions)
            orientations.append(orientation_vector)

        positions = np.array(positions)
        orientations = np.array(orientations)

        # Perform clustering based on positions and orientations
        labels = cluster_peptides(positions, orientations)
        unique_labels = set(labels)

        # Process each cluster to identify sheets
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

                # Initialize rmsd_quad
                rmsd_quad = None

                # Check for curvature if the cluster is not planar
                is_curved = False
                if not is_planar and len(cluster_positions) >= 6:
                    rmsd_quad, _ = fit_quadratic_surface(cluster_positions)
                    is_curved = rmsd_quad < CURVATURE_THRESHOLD

                # Register the sheet based on planarity or curvature
                if is_planar or is_curved:
                    sheet_id = f"sheet_{label}"
                    sheet_records[sheet_id].append(frame_number + args.first)
                    rmsd_quad_str = f"{rmsd_quad:.3f} Å" if rmsd_quad is not None else "N/A"
                    logging.debug(
                        f"Frame {frame_number + args.first}, Sheet {sheet_id}: "
                        f"Size={len(cluster_indices)}, "
                        f"Planar={is_planar}, Curved={is_curved}, "
                        f"RMSD_PCA={rmsd_pca:.3f} Å, "
                        f"Eigenvalues={eigenvalues}, "
                        f"RMSD_Quadratic={rmsd_quad_str}"
                    )

    # Time-resolved analysis, save results, and plot lifetimes
    sheet_lifetimes = time_resolved_sheet_analysis(sheet_records, min_sheet_frames)
    save_sheet_lifetimes(sheet_lifetimes, args.output)
    plot_sheet_lifetimes(sheet_lifetimes, args.output)

    print("SFI analysis completed successfully.")

if __name__ == '__main__':
    main()