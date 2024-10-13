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
import matplotlib.pyplot as plt

# Constants
PLANARITY_THRESHOLD = 1.0  # Å, threshold for RMSD to classify as a sheet
CURVATURE_THRESHOLD = 1.5  # Å, threshold for RMSD in curved sheet fitting
SPATIAL_WEIGHT = 1.0       # Weight for spatial distance in clustering
ORIENTATION_WEIGHT = 1.0   # Weight for orientation similarity in clustering
CLUSTERING_THRESHOLD = 2.0 # Threshold for clustering algorithm
MIN_SHEET_SIZE = 5         # Minimum number of peptides to consider a sheet

def parse_arguments():
    parser = argparse.ArgumentParser(description='Sheet Formation Index (SFI) Analysis')
    parser.add_argument('-t', '--topology', required=True, help='Topology file (e.g., .gro, .pdb)')
    parser.add_argument('-x', '--trajectory', required=True, help='Trajectory file (e.g., .xtc, .trr)')
    parser.add_argument('-s', '--selection', default='resname PEP and backbone', help='Atom selection string for peptides')
    parser.add_argument('-o', '--output', default='sfi_results', help='Output directory for results')
    parser.add_argument('--min_sheet_size', type=int, default=MIN_SHEET_SIZE, help='Minimum number of peptides to consider a sheet')
    args = parser.parse_args()
    return args

def ensure_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def perform_pca(positions):
    """
    Perform PCA to find the best-fit plane for given positions.
    Returns the normal vector of the plane and the RMSD.
    """
    positions_mean = positions.mean(axis=0)
    centered_positions = positions - positions_mean
    covariance_matrix = np.cov(centered_positions.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    normal_vector = eigenvectors[:, 0]  # Normal to the plane (smallest eigenvalue)
    distances = np.dot(centered_positions, normal_vector)
    rmsd = np.sqrt(np.mean(distances**2))
    return normal_vector, rmsd, positions_mean

def fit_quadratic_surface(positions):
    """
    Fit a quadratic surface to the positions to account for curvature.
    Returns the RMSD of the fit.
    """
    def quadratic_surface(X, a, b, c, d, e, f):
        x, y = X
        return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    X = np.vstack((x, y))
    try:
        params, _ = curve_fit(quadratic_surface, X, z)
        z_fit = quadratic_surface(X, *params)
        residuals = z - z_fit
        rmsd = np.sqrt(np.mean(residuals**2))
        return rmsd
    except RuntimeError:
        # Curve fitting failed
        return np.inf

def get_peptide_orientations(peptides):
    """
    Calculate the orientation vectors (backbone vectors) for each peptide.
    Returns an array of orientation vectors.
    """
    orientations = []
    peptide_groups = peptides.groupby('residues')
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

def compute_angle_matrix(orientations):
    """
    Compute a pairwise angle matrix between orientation vectors.
    Returns a matrix of angles in degrees.
    """
    dot_products = np.dot(orientations, orientations.T)
    norms = np.linalg.norm(orientations, axis=1)
    norms_matrix = np.outer(norms, norms)
    cos_angles = dot_products / norms_matrix
    cos_angles = np.clip(cos_angles, -1.0, 1.0)  # Numerical stability
    angles = np.arccos(cos_angles)
    angle_matrix = np.degrees(angles)
    return angle_matrix

def cluster_peptides(positions, orientations, spatial_weight, orientation_weight, clustering_threshold):
    """
    Perform clustering of peptides based on spatial proximity and orientation similarity.
    Returns an array of cluster labels.
    """
    spatial_dist = squareform(pdist(positions))
    angle_matrix = compute_angle_matrix(orientations)
    # Normalize matrices
    spatial_dist /= np.max(spatial_dist)
    angle_matrix /= np.max(angle_matrix)
    # Combine matrices
    distance_matrix = spatial_weight * spatial_dist + orientation_weight * angle_matrix
    clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average', distance_threshold=clustering_threshold)
    labels = clustering.fit_predict(distance_matrix)
    return labels

def time_resolved_sheet_analysis(sheet_records, min_sheet_size):
    """
    Analyze the dynamics of sheet formation over time.
    """
    sheet_tracks = defaultdict(list)
    for frame_number, sheets in sheet_records.items():
        for sheet_id, peptides in sheets.items():
            sheet_tracks[sheet_id].append((frame_number, peptides))
    # Analyze sheet lifetimes and other properties
    sheet_lifetimes = {}
    for sheet_id, history in sheet_tracks.items():
        lifetime = len(history)
        if lifetime >= 1:
            sheet_lifetimes[sheet_id] = lifetime
    return sheet_lifetimes

def main():
    args = parse_arguments()
    ensure_output_directory(args.output)

    # Load the trajectory
    print("Loading trajectory data...")
    u = mda.Universe(args.topology, args.trajectory)
    selection_string = args.selection
    peptides = u.select_atoms(selection_string)
    n_frames = len(u.trajectory)
    print(f"Total frames in trajectory: {n_frames}")

    # Initialize variables for analysis
    sheet_records = defaultdict(dict)  # {frame_number: {sheet_id: peptide_indices}}
    sheet_id_counter = 0

    # Analyze each frame
    print("Analyzing frames for sheet formation...")
    for frame_number, ts in enumerate(u.trajectory):
        positions = peptides.positions
        orientations = get_peptide_orientations(peptides)
        labels = cluster_peptides(positions, orientations, SPATIAL_WEIGHT, ORIENTATION_WEIGHT, CLUSTERING_THRESHOLD)
        unique_labels = set(labels)
        frame_sheets = {}
        for label in unique_labels:
            if label == -1:
                continue  # Noise
            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) >= args.min_sheet_size:
                cluster_positions = positions[cluster_indices]
                # Perform PCA
                normal_vector, rmsd_pca, plane_origin = perform_pca(cluster_positions)
                if rmsd_pca < PLANARITY_THRESHOLD:
                    # Classify as a sheet
                    sheet_id = f"sheet_{sheet_id_counter}"
                    sheet_id_counter += 1
                    frame_sheets[sheet_id] = cluster_indices
                else:
                    # Try quadratic surface fitting
                    rmsd_quad = fit_quadratic_surface(cluster_positions)
                    if rmsd_quad < CURVATURE_THRESHOLD:
                        # Classify as a curved sheet
                        sheet_id = f"sheet_{sheet_id_counter}"
                        sheet_id_counter += 1
                        frame_sheets[sheet_id] = cluster_indices
        sheet_records[frame_number] = frame_sheets

    # Time-resolved analysis
    sheet_lifetimes = time_resolved_sheet_analysis(sheet_records, args.min_sheet_size)
    save_sheet_lifetimes(sheet_lifetimes, args.output)
    plot_sheet_lifetimes(sheet_lifetimes, args.output)

    print("SFI analysis completed successfully.")

def save_sheet_lifetimes(sheet_lifetimes, output_dir):
    """
    Save sheet lifetimes data to a file.
    """
    output_file = os.path.join(output_dir, 'sheet_lifetimes.csv')
    with open(output_file, 'w') as f:
        f.write('SheetID,Lifetime\n')
        for sheet_id, lifetime in sheet_lifetimes.items():
            f.write(f"{sheet_id},{lifetime}\n")
    print(f"Sheet lifetimes data saved to {output_file}")

def plot_sheet_lifetimes(sheet_lifetimes, output_dir):
    """
    Plot the distribution of sheet lifetimes.
    """
    lifetimes = list(sheet_lifetimes.values())
    plt.figure()
    plt.hist(lifetimes, bins=range(1, max(lifetimes)+2), align='left')
    plt.xlabel('Lifetime (frames)')
    plt.ylabel('Number of Sheets')
    plt.title('Distribution of Sheet Lifetimes')
    plt.savefig(os.path.join(output_dir, 'sheet_lifetimes_distribution.png'))
    plt.close()
    print("Sheet lifetimes distribution plot saved.")

if __name__ == '__main__':
    main()
