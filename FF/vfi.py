"""Module for analyzing vesicle characteristics in peptide clusters."""

import os
import numpy as np
import MDAnalysis as mda
from scipy.spatial import ConvexHull
from scipy.ndimage import binary_fill_holes, binary_dilation
from scipy.signal import argrelextrema
from scipy.special import expit
from skimage.morphology import ball

# Constants
SPHERICITY_THRESHOLD = 0.85
HOLLOWNESS_THRESHOLD = 0.2
ASPHERICITY_THRESHOLD = 0.3
ACYLINDRICITY_THRESHOLD = 1.0
VOXEL_SIZE = 0.5
DENSITY_BINS = 50
MIN_ATOMS_SPHERICITY = 10
MIN_VOLUME_SPHERICITY = 100.0
PERFECT_SPHERE_RATIO = (np.pi**(1/3)) * 6**(2/3)
MAX_RADIAL_BINS = 200

def compute_sphericity(positions):
    """Calculate sphericity using convex hull"""
    if len(positions) < MIN_ATOMS_SPHERICITY:
        return 0.0

    try:
        hull = ConvexHull(positions)
        if hull.volume < MIN_VOLUME_SPHERICITY:
            return 0.0

        sphericity = PERFECT_SPHERE_RATIO * (hull.volume**(2/3)) / hull.area
        return max(sphericity, 0.0)
    except Exception:
        return 0.0

def compute_radial_density(positions, com, num_bins=DENSITY_BINS):
    """Calculate radial density profile around center of mass."""
    if len(positions) == 0:
        return np.zeros(num_bins), np.zeros(num_bins + 1)

    if num_bins > MAX_RADIAL_BINS:
        num_bins = MAX_RADIAL_BINS

    try:
        distances = np.linalg.norm(positions - com, axis=1)
        if len(distances) == 0 or distances.max() == 0:
            return np.zeros(num_bins), np.zeros(num_bins + 1)

        bins = np.linspace(0, distances.max(), num_bins)
        density, bin_edges = np.histogram(distances, bins=bins, density=True)
        return density, bin_edges

    except Exception:
        return np.zeros(num_bins), np.zeros(num_bins + 1)

# Update hollowness calculation
def compute_hollowness_ratio(positions, voxel_size=VOXEL_SIZE):
    """Calculate hollowness with improved accuracy"""
    try:
        # 1. Voxel-based hollowness
        pos = positions - positions.min(axis=0)
        voxel_size = min(voxel_size, np.ptp(pos, axis=0).min() / 20)

        grid_size = np.ceil(np.ptp(pos, axis=0) / voxel_size).astype(int) + 4
        grid = np.zeros(grid_size, dtype=bool)

        indices = (pos / voxel_size).astype(int)
        grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True

        grid = binary_dilation(grid, structure=ball(1))
        filled = binary_fill_holes(grid)

        if filled is None:
            return 0.001

        shell_volume = np.sum(grid)
        total_volume = np.sum(filled)
        voxel_hollowness = 0.0
        if total_volume > shell_volume and total_volume > 0:
            voxel_hollowness = (total_volume - shell_volume) / total_volume

        # 2. Radial density-based hollowness
        com = np.mean(positions, axis=0)
        density, _ = compute_radial_density(positions, com)

        if len(density) > 3:
            max_density = np.max(density)
            if (max_density > 0):
                density = density / max_density
                core_density = np.mean(density[:len(density)//4])
                shell_density = np.mean(density[len(density)//4:3*len(density)//4])
                radial_hollowness = (shell_density - core_density) / shell_density if shell_density > core_density else 0.0
            else:
                radial_hollowness = 0.0
        else:
            radial_hollowness = 0.0

        # Increase weight of radial density analysis for vesicles
        combined_hollowness = 0.3 * voxel_hollowness + 0.7 * radial_hollowness
        scaled_ratio = expit(2 * combined_hollowness - 0.5)

        hollowness = max(scaled_ratio, 0.001)
        # Stricter threshold for vesicle detection
        return combined_hollowness > 0.4  # Increased from 0.3

    except Exception:
        return 0.001

def compute_shape_descriptors(positions):
    """Calculate shape descriptors (asphericity and acylindricity)"""
    com = positions.mean(axis=0)
    relative_positions = positions - com
    gyration_tensor = np.dot(relative_positions.T, relative_positions) / len(relative_positions)
    eigenvalues, _ = np.linalg.eigh(gyration_tensor)
    lambda_avg = eigenvalues.mean()

    asphericity = ((eigenvalues - lambda_avg)**2).sum() / (2 * lambda_avg**2)
    acylindricity = ((eigenvalues[1] - eigenvalues[0])**2 +
                     (eigenvalues[2] - eigenvalues[1])**2 +
                     (eigenvalues[0] - eigenvalues[2])**2) / (2 * lambda_avg**2)

    return asphericity, acylindricity

def calculate_dipeptide_length(universe):
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
    """Analyze clusters for vesicle characteristics."""
    results = []
    for cluster_file in cluster_files:
        try:
            u = mda.Universe(cluster_file)
            # Extract cluster number from filename (e.g., cluster1_size7424.gro -> 1)
            filename = os.path.basename(cluster_file)
            try:
                cluster_num = int(''.join(filter(str.isdigit, filename.split('_')[0])))
            except ValueError:
                print(f"Could not extract cluster number from filename: {filename}")
                continue

            dipeptide_length = calculate_dipeptide_length(u)
            if u.atoms is None:
                print(f"Error: No atoms found in {cluster_file}")
                continue

            print(f"Analyzing {filename} - dipeptide length: {dipeptide_length}, "
                  f"total beads: {len(u.atoms)}, estimated dipeptides: {len(u.atoms)/dipeptide_length:.1f}")

            positions = u.atoms.positions
            if len(positions) < min_peptides * dipeptide_length:
                continue

            sphericity = compute_sphericity(positions)
            com = positions.mean(axis=0)
            density, _ = compute_radial_density(positions, com)
            hollowness = compute_hollowness_ratio(positions)
            asphericity, acylindricity = compute_shape_descriptors(positions)

            # Simplified metrics reporting
            metrics = {
                'sphericity': round(float(sphericity), 1),
                'hollowness': bool(hollowness),
                'asphericity': round(float(asphericity), 1),
                'acylindricity': round(float(acylindricity), 1),
                'total_beads': len(positions)
            }

            # Determine if it's a vesicle based on criteria
            is_vesicle = bool(
                sphericity >= SPHERICITY_THRESHOLD and
                hollowness >= HOLLOWNESS_THRESHOLD and
                asphericity <= ASPHERICITY_THRESHOLD and
                acylindricity <= ACYLINDRICITY_THRESHOLD
            )

            results.append({
                'size': len(positions) // dipeptide_length,
                'is_vesicle': is_vesicle,
                'cluster_num': cluster_num,
                'metrics': metrics
            })

        except Exception as e:
            print(f"Error analyzing cluster {cluster_file}: {str(e)}")
            continue

    return results
