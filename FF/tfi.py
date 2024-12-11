"""Module for analyzing tube characteristics in peptide clusters."""

import os
import numpy as np
import MDAnalysis as mda
from scipy.signal import argrelextrema

# Constant
RADIAL_THRESHOLD = 12.0
ANGULAR_UNIFORMITY_THRESHOLD = 0.04
ASPHERICITY_THRESHOLD = 0.5
RATIO_THRESHOLD = 0.3
SEGMENT_LENGTH = 40
STEP_SIZE = 20
TUBE_SEGMENT_RATIO_THRESHOLD = 0.5

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

def segment_based_analysis(positions, segment_length=SEGMENT_LENGTH, step_size=STEP_SIZE):
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

    return tube_like_segments / num_segments if num_segments > 0 else 0

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

def calculate_peptide_length(universe):
    """Calculate number of beads in a dipeptide unit by finding 1XXX and 2XXX pairs."""
    dipeptide_beads = 0
    current_dipeptide = None

    for atom in universe.atoms:
        # Check if residue name starts with 1 or 2
        resname_prefix = atom.resname[0]
        if resname_prefix in ('1', '2'):
            if current_dipeptide != atom.resid:
                if resname_prefix == '1':  # Start of new dipeptide
                    current_dipeptide = atom.resid
                    # Count beads in this dipeptide (1XXX + 2XXX)
                    dipeptide_atoms = [a for a in universe.atoms if a.resid in (atom.resid, atom.resid + 1)]
                    if dipeptide_beads == 0:  # Only need to calculate once
                        dipeptide_beads = len(dipeptide_atoms)
                        break

    return dipeptide_beads if dipeptide_beads > 0 else 8  # Default fallback

def analyze_clusters(cluster_files, min_peptides):
    """Analyze clusters for tube characteristics."""
    results = []
    for cluster_file in cluster_files:
        try:
            u = mda.Universe(cluster_file)
            filename = os.path.basename(cluster_file)
            cluster_num = int(''.join(filter(str.isdigit, filename.split('_')[0])))

            if u.atoms is None:
                continue

            positions = u.atoms.positions
            if len(positions) < min_peptides * 8:  # Assuming 8 beads per peptide as minimum
                continue

            # Perform tube analysis matching tfi_analysis.py logic
            tube_segment_ratio = segment_based_analysis(positions)
            radial_std, angular_uniformity, r, theta, z, principal_axis = perform_cylindrical_analysis(positions)
            density, bin_edges = compute_radial_density(r)
            hollow = is_hollow_tube(density, bin_edges)
            asphericity, ratio = compute_shape_anisotropy(positions)

            # Simplified metrics reporting
            metrics = {
                'radial_std': round(float(radial_std), 1),
                'angular_uniformity': round(float(angular_uniformity), 1),
                'tube_segment_ratio': round(float(tube_segment_ratio), 1),
                'hollow': bool(hollow),
                'asphericity': round(float(asphericity), 1),
                'ratio': round(float(ratio), 1),
                'total_beads': len(positions)
            }

            # Determine if it's a tube based on criteria
            is_tube = bool(
                tube_segment_ratio >= TUBE_SEGMENT_RATIO_THRESHOLD and
                radial_std < RADIAL_THRESHOLD and
                angular_uniformity > ANGULAR_UNIFORMITY_THRESHOLD and
                hollow and
                asphericity > ASPHERICITY_THRESHOLD and
                ratio < RATIO_THRESHOLD
            )

            results.append({
                'size': len(positions),
                'is_tube': is_tube,
                'cluster_num': cluster_num,
                'metrics': metrics
            })

        except Exception as e:
            print(f"Error analyzing cluster {cluster_file}: {str(e)}")
            continue

    return results
