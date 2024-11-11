# util/geometry.py

"""
Geometry Utilities for Peptide Analysis

This module provides utility functions for geometric calculations, such as PCA,
surface fitting, orientation computations, and shape descriptors.

Functions:
- perform_pca: Perform Principal Component Analysis on positions.
- fit_quadratic_surface: Fit a quadratic surface to positions.
- compute_orientation_matrix: Compute pairwise orientation differences.
- compute_dipeptide_centroids: Compute centroids of dipeptides.
- compute_dipeptide_orientations: Compute orientations of dipeptides.
- compute_radial_density: Compute radial density profile.
- compute_sphericity: Compute sphericity of a set of positions.
- compute_hollowness_ratio: Compute hollowness ratio using voxelization.
- compute_shape_descriptors: Compute asphericity and acylindricity.
- perform_cylindrical_analysis: Perform cylindrical harmonic analysis.
- compute_angular_uniformity: Compute angular uniformity metric.
- compute_shape_anisotropy: Compute shape anisotropy using gyration tensor.
- compute_moments_of_inertia: Compute moments of inertia and shape ratios.
- compute_fop: Compute Fibrillar Order Parameter.
- analyze_orientation_distribution: Analyze peptide orientation distribution.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.spatial import ConvexHull
from scipy.ndimage import binary_fill_holes
from scipy.signal import argrelextrema
from skimage.measure import label
from sklearn.decomposition import PCA
import warnings

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

def compute_orientation_matrix(orientations):
    """
    Compute a pairwise orientation difference matrix between orientation vectors.

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

def compute_dipeptide_centroids(peptides, peptide_length):
    """
    Compute the centroids of dipeptides (or peptides of specified length).

    Parameters:
    - peptides (MDAnalysis.AtomGroup): AtomGroup containing the peptide atoms.
    - peptide_length (int): Number of atoms per peptide.

    Returns:
    - positions (numpy.ndarray): Array of centroid positions for each peptide.
    """
    num_peptides = len(peptides) // peptide_length
    positions = np.zeros((num_peptides, 3))
    for i in range(num_peptides):
        start = i * peptide_length
        end = start + peptide_length
        positions[i] = peptides[start:end].positions.mean(axis=0)
    return positions

def compute_dipeptide_orientations(peptides, peptide_length):
    """
    Compute the orientations of dipeptides (or peptides of specified length) using PCA.

    Parameters:
    - peptides (MDAnalysis.AtomGroup): AtomGroup containing the peptide atoms.
    - peptide_length (int): Number of atoms per peptide.

    Returns:
    - orientations (numpy.ndarray): Array of orientation vectors for each peptide.
    """
    num_peptides = len(peptides) // peptide_length
    orientations = np.zeros((num_peptides, 3))
    for i in range(num_peptides):
        start = i * peptide_length
        end = start + peptide_length
        peptide_positions = peptides[start:end].positions
        _, orientation_vector, _, _, _ = perform_pca(peptide_positions)
        if orientation_vector is not None:
            orientations[i] = orientation_vector
        else:
            orientations[i] = np.zeros(3)
    return orientations

def compute_radial_density(positions, com, num_bins):
    """
    Compute the radial density profile of positions relative to a center of mass.

    Parameters:
    - positions (numpy.ndarray): Positions of atoms.
    - com (numpy.ndarray): Center of mass.
    - num_bins (int): Number of bins for the histogram.

    Returns:
    - density (numpy.ndarray): Radial density values.
    - bin_edges (numpy.ndarray): Edges of the bins.
    """
    distances = np.linalg.norm(positions - com, axis=1)
    max_distance = distances.max()
    bins = np.linspace(0, max_distance, num_bins)
    density, bin_edges = np.histogram(distances, bins=bins, density=True)
    return density, bin_edges

def compute_sphericity(positions):
    """
    Compute the sphericity of a set of positions.

    Parameters:
    - positions (numpy.ndarray): Positions of atoms.

    Returns:
    - sphericity (float): Sphericity value between 0 and 1.
    """
    if len(positions) < 4:
        return 0
    try:
        hull = ConvexHull(positions)
        surface_area = hull.area
        volume = hull.volume
        sphericity = (np.pi**(1/3)) * (6 * volume)**(2/3) / surface_area
        return sphericity
    except Exception as e:
        print(f"ConvexHull computation failed: {e}")
        return 0

def compute_hollowness_ratio(positions, voxel_size=2.0):
    """
    Quantify hollowness using voxelization and flood fill algorithm.

    Parameters:
    - positions (numpy.ndarray): Positions of atoms.
    - voxel_size (float): Size of each voxel.

    Returns:
    - hollowness_ratio (float): Ratio of void volume to total volume.
    """
    # Define voxel grid dimensions
    min_coords = positions.min(axis=0) - voxel_size
    max_coords = positions.max(axis=0) + voxel_size
    grid_shape = np.ceil((max_coords - min_coords) / voxel_size).astype(int)
    grid = np.zeros(grid_shape, dtype=bool)

    # Map positions to grid indices
    indices = np.floor((positions - min_coords) / voxel_size).astype(int)
    grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True

    # Try to fill internal voids
    try:
        filled_grid = binary_fill_holes(grid)
        if filled_grid is None:
            print("Warning: binary_fill_holes returned None. Defaulting hollowness_ratio to 0.")
            return 0  # Default hollowness ratio if filling fails

        # Compute volumes
        aggregate_volume = int(grid.sum())
        total_volume = int(filled_grid.sum())
        void_volume = total_volume - aggregate_volume

        hollowness_ratio = void_volume / total_volume if total_volume > 0 else 0
        return hollowness_ratio
    except MemoryError:
        print("Memory error during hollowness calculation. Consider adjusting voxel size or analyzing fewer frames.")
        return 0  # Return 0 as the default if memory error occurs

def compute_shape_descriptors(positions):
    """
    Compute asphericity and acylindricity of a set of positions.

    Parameters:
    - positions (numpy.ndarray): Positions of atoms.

    Returns:
    - asphericity (float): Asphericity value.
    - acylindricity (float): Acylindricity value.
    """
    com = positions.mean(axis=0)
    relative_positions = positions - com
    gyration_tensor = np.dot(relative_positions.T, relative_positions) / len(relative_positions)
    eigenvalues, _ = np.linalg.eigh(gyration_tensor)
    lambda_avg = eigenvalues.mean()
    asphericity = ((eigenvalues - lambda_avg)**2).sum() / (2 * lambda_avg**2)
    acylindricity = ((eigenvalues[1] - eigenvalues[0])**2 + (eigenvalues[2] - eigenvalues[1])**2 +
                     (eigenvalues[0] - eigenvalues[2])**2) / (2 * lambda_avg**2)
    return asphericity, acylindricity

def perform_cylindrical_analysis(positions):
    """
    Perform cylindrical harmonic analysis on a set of positions.

    Parameters:
    - positions (numpy.ndarray): Positions of atoms.

    Returns:
    - radial_std (float): Standard deviation of radial distances.
    - angular_uniformity (float): Measure of angular uniformity.
    - r (numpy.ndarray): Radial distances.
    - theta (numpy.ndarray): Angular coordinates.
    - z (numpy.ndarray): Positions along the principal axis.
    - principal_axis (numpy.ndarray): Principal axis vector.
    """
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
    """
    Compute angular uniformity of angular coordinates.

    Parameters:
    - theta (numpy.ndarray): Angular coordinates in radians.

    Returns:
    - angular_uniformity (float): Angular uniformity metric.
    """
    histogram, _ = np.histogram(theta, bins=36, range=(-np.pi, np.pi))
    histogram_normalized = histogram / np.sum(histogram)
    with np.errstate(divide='ignore', invalid='ignore'):
        entropy = -np.sum(histogram_normalized * np.log(histogram_normalized + 1e-8))
    max_entropy = np.log(len(histogram))
    angular_uniformity = 1 - (entropy / max_entropy)
    return angular_uniformity

def compute_shape_anisotropy(positions):
    """
    Compute shape anisotropy of a set of positions using the gyration tensor.

    Parameters:
    - positions (numpy.ndarray): Positions of atoms.

    Returns:
    - asphericity (float): Asphericity value.
    - ratio (float): Ratio of smallest to largest eigenvalue.
    """
    relative_positions = positions - positions.mean(axis=0)
    gyration_tensor = np.dot(relative_positions.T, relative_positions) / len(relative_positions)
    eigenvalues, _ = np.linalg.eigh(gyration_tensor)
    eigenvalues = np.sort(eigenvalues)
    asphericity = 1 - (2 * (eigenvalues[0] + eigenvalues[1]) / (2 * eigenvalues[2]))
    ratio = eigenvalues[0] / eigenvalues[2]
    return asphericity, ratio

def compute_moments_of_inertia(positions):
    """
    Compute the moments of inertia and shape ratios of an aggregate.

    Parameters:
    - positions (numpy.ndarray): Positions of atoms.

    Returns:
    - shape_ratio1 (float): Ratio of largest to second largest eigenvalue.
    - shape_ratio2 (float): Ratio of second largest to smallest eigenvalue.
    - principal_axis (numpy.ndarray): Principal axis vector.
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

def compute_fop(orientations, principal_axis):
    """
    Compute the Fibrillar Order Parameter (FOP).

    Parameters:
    - orientations (numpy.ndarray): Orientation vectors of peptides.
    - principal_axis (numpy.ndarray): Principal axis vector.

    Returns:
    - fop (float): Fibrillar Order Parameter value.
    """
    cos_angles = np.dot(orientations, principal_axis)
    cos2_angles = (3 * cos_angles**2 - 1) / 2  # Standard P2(cosθ)
    fop = np.mean(cos2_angles)
    # FOP = 1: Perfect alignment.
    # FOP = -0.5: Perfect anti-alignment.
    # FOP = 0: Random orientation.
    return fop

def analyze_orientation_distribution(orientations, principal_axis):
    """
    Analyze the distribution of peptide orientations relative to the principal axis.

    Parameters:
    - orientations (numpy.ndarray): Orientation vectors of peptides.
    - principal_axis (numpy.ndarray): Principal axis vector.

    Returns:
    - mean_angle (float): Mean angle in degrees.
    - std_angle (float): Standard deviation of angles in degrees.
    - angles (numpy.ndarray): Array of angles in degrees.
    """
    cos_angles = np.dot(orientations, principal_axis)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)  # Numerical stability
    angles = np.arccos(cos_angles) * (180 / np.pi)  # Convert to degrees
    mean_angle = np.mean(angles)
    std_angle = np.std(angles)
    return mean_angle, std_angle, angles

def get_peptide_orientations(aggregate_atoms):
    """
    Compute orientation vectors for peptides in an aggregate.

    Args:
        aggregate_atoms (MDAnalysis.AtomGroup): AtomGroup of the aggregate.

    Returns:
        numpy.ndarray: Array of orientation vectors for each peptide.
    """
    import MDAnalysis as mda

    orientations = []
    residues = aggregate_atoms.residues

    for residue in residues:
        backbone_atoms = residue.atoms.select_atoms('name BB')
        if len(backbone_atoms) == 1:
            position = backbone_atoms.positions[0]
            # Use the position difference between adjacent residues
            # to define the orientation vector
            next_residue = residue.residue_group.next(residue)
            if next_residue:
                next_bb_atoms = next_residue.atoms.select_atoms('name BB')
                if len(next_bb_atoms) == 1:
                    next_position = next_bb_atoms.positions[0]
                    vector = next_position - position
                    norm = np.linalg.norm(vector)
                    if norm > 0:
                        orientations.append(vector / norm)
                    else:
                        orientations.append(np.zeros(3))
                else:
                    orientations.append(np.zeros(3))
            else:
                orientations.append(np.zeros(3))
        else:
            orientations.append(np.zeros(3))

    return np.array(orientations)

def cross_sectional_profiling(relative_positions, principal_axis, num_sections=10, thickness=5.0):
    """
    Perform cross-sectional profiling along the principal axis.

    Args:
        relative_positions (numpy.ndarray): Positions relative to the aggregate's center of mass.
        principal_axis (numpy.ndarray): Principal axis vector.
        num_sections (int): Number of cross-sectional slices.
        thickness (float): Thickness of each cross-sectional slice in Å.

    Returns:
        list: List of cross-sectional areas for each section.
    """
    z = np.dot(relative_positions, principal_axis)
    z_min, z_max = z.min(), z.max()
    cross_section_areas = []
    section_length = (z_max - z_min) / num_sections

    for i in range(num_sections):
        z_center = z_min + (i + 0.5) * section_length
        indices = np.where((z >= z_center - thickness / 2) & (z < z_center + thickness / 2))[0]
        cross_section_positions = relative_positions[indices]

        if len(cross_section_positions) >= 3:
            # Project onto plane perpendicular to principal axis
            projections = cross_section_positions - np.outer(np.dot(cross_section_positions, principal_axis), principal_axis)
            # Compute Convex Hull area
            try:
                hull = ConvexHull(projections[:, :2])  # Use first two coordinates
                area = hull.area
                cross_section_areas.append(area)
            except Exception as e:
                print(f"ConvexHull computation failed in section {i}: {e}")
                cross_section_areas.append(0)
        else:
            cross_section_areas.append(0)

    return cross_section_areas