#!/usr/bin/env python3
"""
Module for analyzing sheet characteristics in peptide clusters.
"""

import warnings
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import numpy as np
import MDAnalysis as mda
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull
import logging

logger = logging.getLogger('sfi')

# Constants
PLANARITY_THRESHOLD = 1.5  # Increased from 1.2
CURVATURE_THRESHOLD = 4.0  # Increased from 3.0
LOCAL_THICKNESS_THRESHOLD = 2.5  # Increased from 2.0
MIN_SURFACE_AREA = 80.0  # Decreased from 100.0
SURFACE_DENSITY_THRESHOLD = 0.5  # Decreased from 0.7
SPATIAL_WEIGHT = 1.2
ORIENTATION_WEIGHT = 1.0
SPATIAL_CUTOFF = 15
ANGLE_CUTOFF = 45
MIN_SHEET_SIZE = 5
MIN_ATOMS_SPHERICITY = 10
PERFECT_SPHERE_RATIO = 1.0
MIN_VOLUME_SPHERICITY = 0.1  # Rolled back

def perform_pca(positions):
    """Perform PCA with robust error handling."""
    if len(positions) < 3:
        return None, None, np.inf, None, None

    try:
        # Center positions
        positions_mean = positions.mean(axis=0)
        centered_positions = positions - positions_mean

        # Compute covariance matrix
        covariance_matrix = np.cov(centered_positions.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort eigenvalues and corresponding eigenvectors
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Normal vector is eigenvector with smallest eigenvalue
        normal_vector = eigenvectors[:, 0]
        orientation_vector = eigenvectors[:, 2]  # Largest eigenvalue direction

        # Calculate RMSD from best-fit plane
        distances = np.abs(np.dot(centered_positions, normal_vector))
        rmsd = np.sqrt(np.mean(distances**2))

        return normal_vector, orientation_vector, rmsd, positions_mean, eigenvalues

    except np.linalg.LinAlgError:
        return None, None, np.inf, None, None

def analyze_bb_sc_distribution(positions, atoms):
    """Analyze backbone and sidechain distribution to detect bilayer characteristics."""
    # Placeholder implementation
    bb_positions = positions[atoms.names == 'BB']
    sc_positions = positions[atoms.names != 'BB']
    bb_sc_separation = np.mean(np.linalg.norm(bb_positions - sc_positions, axis=1))
    is_bilayer = bb_sc_separation > 5.0  # Example threshold
    return is_bilayer, bb_sc_separation

def analyze_local_structure(positions, k=10):
    """Analyze local planarity in smaller neighborhoods"""
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=k).fit(positions)
    _, indices = nbrs.kneighbors(positions)

    local_planarities = []
    local_normals = []

    for idx_group in indices:
        local_pos = positions[idx_group]
        _, _, local_rmsd, _, eigenvals = perform_pca(local_pos)
        local_planarities.append(local_rmsd)
        if eigenvals is not None:
            thickness = np.sqrt(eigenvals[0])  # Smallest eigenvalue = thickness²
            if thickness > LOCAL_THICKNESS_THRESHOLD:
                local_planarities[-1] = np.inf

    return np.array(local_planarities)

def is_sheet_like(eigenvalues, rmsd, positions=None, atoms=None):
    """Enhanced sheet detection with more lenient criteria"""
    if eigenvalues is None or len(eigenvalues) < 3:
        return False

    try:
        eigenvalues = np.sort(eigenvalues)
        thickness = np.sqrt(eigenvalues[0])
        width = np.sqrt(eigenvalues[1])
        length = np.sqrt(eigenvalues[2])

        # More lenient thickness ratio
        thickness_ratio = thickness / width
        if thickness_ratio > 0.7:  # Increased from 0.5
            return False

        # Check curvature with more lenient criteria
        curvature_rmsd, params = fit_quadratic_surface(positions)
        if params is not None:
            max_curvature = max(abs(params[0]), abs(params[1]))
            if max_curvature > 0.15:  # Increased from 0.1
                return curvature_rmsd < CURVATURE_THRESHOLD * 2.0  # More lenient

        # Additional sheet criteria
        aspect_ratio = length / width
        if aspect_ratio < 1.2:  # Added minimum aspect ratio
            return False

        # Allow more curved structures
        if curvature_rmsd < CURVATURE_THRESHOLD * 1.5:
            return True

        return rmsd < PLANARITY_THRESHOLD

    except Exception as e:
        logger.error(f"Error in is_sheet_like: {str(e)}")
        return False

def compute_thickness_variation(positions):
    """Compute the variation in thickness of the sheet."""
    # Placeholder implementation
    return np.std(positions[:, 2] - positions[:, 2].mean())

def fit_quadratic_surface(positions):
    """Enhanced surface fitting for curved bilayer detection"""
    if len(positions) < 6:
        return np.inf, None

    try:
        # Center and get principal components
        center = positions.mean(axis=0)
        centered_pos = positions - center
        cov_matrix = np.cov(centered_pos.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)

        # Project onto plane of largest variance
        proj_matrix = np.column_stack((eigenvecs[:, 1], eigenvecs[:, 2]))
        proj_points = np.dot(centered_pos, proj_matrix)
        heights = np.dot(centered_pos, eigenvecs[:, 0])

        # Check thickness distribution for bilayer characteristics
        height_std = np.std(heights)
        if height_std < 3.0:  # Too thin to be a bilayer
            return np.inf, None

        # Fit surface
        x, y = proj_points[:, 0], proj_points[:, 1]
        X = np.vstack((x, y))

        def quadratic_surface(X, a, b, c, d, e, f):
            x, y = X
            return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f

        params, _ = curve_fit(quadratic_surface, X, heights, p0=[0, 0, 0, 0, 0, 0])
        z_fit = quadratic_surface(X, *params)
        residuals = heights - z_fit
        rmsd = np.sqrt(np.mean(residuals**2))

        # Adjust RMSD based on curvature
        a, b = params[0], params[1]
        max_curvature = max(abs(a), abs(b))
        if max_curvature > 0.05:  # Significant curvature
            rmsd *= 0.5  # Reduce RMSD penalty for curved structures

        return rmsd, params

    except Exception:
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

def calculate_peptide_length(universe):
    """Calculate the number of beads per peptide from the universe"""
    if not hasattr(universe, 'residues') or not universe.residues:
        return 0  # No residues found

    residues = universe.residues
    if not hasattr(residues[0], 'atoms') or not residues[0].atoms:
        return 0  # No atoms in residue

    # Count beads in first residue
    beads_per_residue = len(residues[0].atoms)
    # Multiply by 2 for dipeptide
    return beads_per_residue * 2
    # For dipeptides: Look at first residue
    first_res = residues[0]
    beads_per_res = len(first_res.atoms)
    if beads_per_res == 4:  # BB + SC1 + SC2 + SC3
        return beads_per_res * 2  # 8 beads total for dipeptide
    else:
        logging.warning(f"Unexpected number of beads per residue: {beads_per_res}")
        return beads_per_res * 2

def analyze_clusters(cluster_files, min_peptides):
    """Analyze clusters with improved curved sheet detection"""
    results = []
    for cluster_file in cluster_files:
        try:
            u = mda.Universe(cluster_file)
            filename = os.path.basename(cluster_file)
            cluster_num = int(''.join(filter(str.isdigit, filename.split('_')[0])))

            if u.atoms is None:
                continue

            positions = u.atoms.positions
            if len(positions) < min_peptides * 8:  # Assuming 8 beads per peptide minimum
                continue

            # Enhanced sheet analysis matching sfi_analysis.py
            normal_vector, orientation_vector, rmsd, _, eigenvalues = perform_pca(positions)

            if eigenvalues is None:
                continue

            curvature_rmsd, params = fit_quadratic_surface(positions)
            sphericity = compute_sphericity(positions)

            # Enhanced sheet criteria
            is_sheet = bool(
                is_sheet_like(eigenvalues, rmsd, positions, u.atoms) and
                (sphericity < 0.95) and  # Not too spherical
                any([  # Must meet at least one of these criteria
                    rmsd < PLANARITY_THRESHOLD,  # Flat sheet
                    (curvature_rmsd < CURVATURE_THRESHOLD and  # Curved sheet
                     params is not None and
                     max(abs(params[0]), abs(params[1])) > 0.05)
                ])
            )

            is_curved = False
            if params is not None:
                a, b = params[0], params[1]
                is_curved = (abs(a) > 0.1 and abs(b) < 0.05) or (abs(b) > 0.1 and abs(a) < 0.05)

            # Simplified metrics reporting
            metrics = {
                'planarity_rmsd': round(float(rmsd), 1),
                'curvature_rmsd': round(float(curvature_rmsd), 1),
                'sphericity': round(float(sphericity), 1),
                'total_beads': len(positions)
            }

            # Enhanced sheet detection criteria
            is_sheet = bool(
                (rmsd < PLANARITY_THRESHOLD or
                 (curvature_rmsd < CURVATURE_THRESHOLD * 2.0 and  # More lenient for curved sheets
                  params is not None and
                  max(abs(params[0]), abs(params[1])) > 0.05)) and  # Check for curvature
                sphericity < 0.95 and
                eigenvalues is not None and
                (np.sqrt(eigenvalues[0]/eigenvalues[1]) < 0.5 or  # Standard thickness ratio
                 (params is not None and  # Alternative criterion for curved sheets
                  max(abs(params[0]), abs(params[1])) > 0.05 and
                  np.sqrt(eigenvalues[0]/eigenvalues[1]) < 0.7))  # More lenient for curved sheets
            )

            # Add bilayer check
            if u.atoms is not None:
                is_bilayer, bb_sc_sep = analyze_bb_sc_distribution(positions, u.atoms)
                if is_bilayer:
                    is_sheet = True  # Override if bilayer characteristics are detected

            results.append({
                'size': len(positions),
                'is_sheet': is_sheet,
                'cluster_num': cluster_num,
                'metrics': metrics
            })

        except Exception as e:
            logger.error(f"Error analyzing cluster {cluster_file}: {str(e)}")
            continue

    return results

def compute_sphericity(positions):
    """Calculate sphericity using convex hull with enhanced criteria."""
    if len(positions) < MIN_ATOMS_SPHERICITY:

        return 0.0

    try:
        hull = ConvexHull(positions)
        if hull.volume < MIN_VOLUME_SPHERICITY:
            return 0.0

        # Enhanced sphericity calculation
        sphericity = PERFECT_SPHERE_RATIO * (hull.volume ** (2/3)) / hull.area
        sphericity = max(sphericity, 0.0)

        # Additional validation to exclude vesicle-like structures
        if sphericity < 0.8:
            return 0.0

        return sphericity
    except Exception:
        return 0.0
