"""Module for analyzing fiber characteristics in peptide clusters."""

import warnings
warnings.filterwarnings("ignore", message=".*Bio.Application modules and modules relying on it have been deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning)

import os
import numpy as np
import MDAnalysis as mda
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
import logging
logger = logging.getLogger('ffi')

# Constants
SHAPE_RATIO_THRESHOLD = 2.5  # Decreased to catch more potential fibers
ALIGNMENT_STD_THRESHOLD = 25.0  # More strict alignment requirement
CROSS_SECTION_THICKNESS = 5.0
NUM_CROSS_SECTIONS = 10
FOP_THRESHOLD_POSITIVE = 0.05  # Lowered to be less strict
FOP_THRESHOLD_NEGATIVE = -0.1
MIN_LENGTH_RATIO = 4.0  # More elongated
RADIUS_VARIATION_THRESHOLD = 0.3  # Rolled back from 0.2
MIN_CYLINDRICAL_SCORE = 0.6  # Keep relatively strict
CROSS_SECTION_VAR_THRESHOLD = 0.5  # Consistency of cross-section

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

def get_peptide_orientations(cluster_atoms):
    """Calculate the orientation vectors for each dipeptide."""
    orientations = []
    peptide_groups = cluster_atoms.residues

    for i in range(0, len(peptide_groups) - 1, 2):
        residue1 = peptide_groups[i]
        residue2 = peptide_groups[i + 1]

        backbone1 = residue1.atoms.select_atoms('name BB')
        backbone2 = residue2.atoms.select_atoms('name BB')

        if len(backbone1.positions) == 1 and len(backbone2.positions) == 1:
            vector = backbone2.positions[0] - backbone1.positions[0]
            norm = np.linalg.norm(vector)
            orientations.append(vector / norm if norm > 0 else np.zeros(3))
        else:
            orientations.append(np.zeros(3))

    return np.array(orientations)

def analyze_orientation_distribution(orientations, principal_axis):
    """
    Analyze the distribution of peptide orientations relative to the principal axis.
    """
    # Filter out zero vectors
    valid_orientations = orientations[np.linalg.norm(orientations, axis=1) > 0.1]
    if len(valid_orientations) < 3:
        return 90, 90, np.array([90])  # Default values indicating random orientation

    # Calculate absolute angles (0-90 degrees) to handle bidirectional fibers
    cos_angles = np.abs(np.dot(valid_orientations, principal_axis))
    cos_angles = np.clip(cos_angles, 0.0, 1.0)
    angles = np.arccos(cos_angles) * (180 / np.pi)

    # Consider both parallel and antiparallel alignments
    angles = np.minimum(angles, 180 - angles)

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

def check_cylindrical_geometry(positions, principal_axis):
    """Enhanced check for fiber-like cylindrical geometry."""
    proj_matrix = np.eye(3) - np.outer(principal_axis, principal_axis)
    projections = np.dot(positions, proj_matrix)

    z = np.dot(positions, principal_axis)
    z_range = np.ptp(z)

    # More sections for better analysis
    sections = np.linspace(z.min(), z.max(), NUM_CROSS_SECTIONS * 2)

    radii = []
    areas = []

    for z_mid in sections[1:-1]:
        mask = np.abs(z - z_mid) < CROSS_SECTION_THICKNESS
        if np.sum(mask) > 6:  # Increased minimum points
            section = projections[mask]
            center = np.mean(section, axis=0)
            radii_section = np.linalg.norm(section - center, axis=1)
            radii.append(np.mean(radii_section))

            # Calculate cross-sectional area
            if len(section) > 3:
                hull = ConvexHull(section[:,:2])
                areas.append(hull.area)

    if not radii or not areas:
        return False, 0.0, 0.0

    radius_variation = np.std(radii) / np.mean(radii)
    area_variation = np.std(areas) / np.mean(areas)
    avg_radius = np.mean(radii)

    return (radius_variation < RADIUS_VARIATION_THRESHOLD,
            radius_variation,
            z_range / avg_radius)

def compute_cylindrical_score(positions, principal_axis):
    """Rolled back to original cylindrical score calculation"""
    proj_matrix = np.eye(3) - np.outer(principal_axis, principal_axis)
    projections = np.dot(positions, proj_matrix)

    z = np.dot(positions, principal_axis)
    z_range = np.ptp(z)
    sections = np.linspace(z.min(), z.max(), 20)

    radii = []
    circularities = []

    for z_mid in sections[1:-1]:
        mask = np.abs(z - z_mid) < z_range/20
        if np.sum(mask) > 10:
            section = projections[mask]
            center = np.mean(section, axis=0)
            radii_section = np.linalg.norm(section - center, axis=1)
            radii.append(np.mean(radii_section))

            if len(section) > 3:
                try:
                    hull = ConvexHull(section[:,:2])
                    area = hull.area
                    perimeter = hull.area
                    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                    circularities.append(circularity)
                except Exception:
                    continue

    if not radii or not circularities:
        return 0.0

    # Original scoring system
    radius_consistency = 1 - min(1.0, (np.std(radii) / np.mean(radii)))
    avg_circularity = np.mean(circularities)
    length_ratio = min(3.0, z_range / np.mean(radii)) / 3.0

    # Original weights
    return (radius_consistency * 0.4 +
            avg_circularity * 0.3 +
            length_ratio * 0.3)

def compute_radial_density(positions, nbins=20):
            """
            Compute the radial density profile from the center.

            Args:
                positions: Nx3 array of coordinates
                nbins: Number of radial bins

            Returns:
                density: Array of density values
                bins: Array of bin edges
            """
            # Calculate distances from center
            distances = np.linalg.norm(positions, axis=1)

            # Create histogram of counts
            max_dist = np.max(distances)
            bins = np.linspace(0, max_dist, nbins)
            counts, bin_edges = np.histogram(distances, bins=bins)

            # Calculate volumes of shells
            shell_volumes = 4/3 * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)

            # Calculate density (counts per volume)
            density = counts / shell_volumes

            return density, bin_edges

def is_hollow_tube(density, threshold=None):
            """
            Determine if structure has a hollow core based on radial density.

            Args:
                density: Array of radial density values
                threshold: Optional density threshold

            Returns:
                bool: True if structure appears hollow
            """
            if threshold is None:
                threshold = np.max(density) * 0.3

            # Check if inner density is significantly lower than outer
            inner_density = np.mean(density[:len(density)//3])
            outer_density = np.mean(density[len(density)//3:2*len(density)//3])

            return inner_density < threshold and outer_density > inner_density * 2

def compute_sphericity(positions):
    """
    Compute the sphericity of a set of positions.
    """
    hull = ConvexHull(positions)
    volume = hull.volume
    area = hull.area
    sphericity = (np.pi ** (1 / 3) * (6 * volume) ** (2 / 3)) / area
    return sphericity

def analyze_clusters(cluster_files, min_peptides):
    """Enhanced cluster analysis for fiber detection with better sheet distinction"""
    results = []
    for cluster_file in cluster_files:
        try:
            u = mda.Universe(cluster_file)
            filename = os.path.basename(cluster_file)
            cluster_num = int(''.join(filter(str.isdigit, filename.split('_')[0])))

            if u.atoms is None or len(u.atoms) < min_peptides * 8:
                continue

            positions = u.atoms.positions
            shape_ratio1, shape_ratio2, principal_axis = compute_moments_of_inertia(positions)
            is_cylindrical, radius_var, length_ratio = check_cylindrical_geometry(positions, principal_axis)
            cylindrical_score = compute_cylindrical_score(positions, principal_axis)

            orientations = get_peptide_orientations(u.atoms)
            mean_angle, std_angle, angles = analyze_orientation_distribution(orientations, principal_axis)
            fop = compute_fop(orientations, principal_axis)

            # Calculate additional metrics for sheet vs fiber distinction
            # Check if structure has consistent cross-sectional area (cylinder-like)
            cross_sections = cross_sectional_profiling(positions - positions.mean(axis=0), principal_axis)
            if len(cross_sections) > 2:
                cross_section_variation = np.std(cross_sections) / np.mean(cross_sections)
            else:
                cross_section_variation = float('inf')

            # Determine if it's a fiber based on updated criteria
            is_fiber = bool(
                cylindrical_score > MIN_CYLINDRICAL_SCORE and
                shape_ratio1 >= SHAPE_RATIO_THRESHOLD and
                std_angle < ALIGNMENT_STD_THRESHOLD and
                length_ratio > MIN_LENGTH_RATIO and
                cross_section_variation < CROSS_SECTION_VAR_THRESHOLD and  # Must have consistent cross-section
                not is_hollow_tube(compute_radial_density(positions)[0])  # Should not be hollow inside
            )

            # Update metrics
            metrics = {
                'shape_ratio': round(float(shape_ratio1), 1),
                'alignment': round(float(std_angle), 1),
                'cylindrical_score': round(float(cylindrical_score), 1),
                'length_ratio': round(float(length_ratio), 1),
                'cross_section_var': round(float(cross_section_variation), 2),
                'total_beads': len(positions)
            }

            results.append({
                'size': len(positions),
                'is_fiber': is_fiber,
                'cluster_num': cluster_num,
                'metrics': metrics
            })

        except Exception as e:
            logger.error(f"Error analyzing cluster {cluster_file}: {str(e)}")
            continue

    return results
