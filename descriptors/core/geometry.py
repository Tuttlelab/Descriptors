"""
geometry.py: Shared geometry and shape analysis utilities for descriptors.
"""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.ndimage import binary_fill_holes, binary_dilation
from skimage.morphology import ball

def compute_sphericity(positions):
    if len(positions) < 4:
        return 0.0
    hull = ConvexHull(positions)
    volume = hull.volume
    area = hull.area
    if area == 0:
        return 0.0
    sphericity = (np.pi ** (1/3)) * (6 * volume) ** (2/3) / area
    return sphericity

def compute_gyration_tensor(positions):
    com = np.mean(positions, axis=0)
    rel = positions - com
    gyration_tensor = np.dot(rel.T, rel) / len(rel)
    eigenvalues, _ = np.linalg.eigh(gyration_tensor)
    eigenvalues = np.sort(eigenvalues)
    asphericity = 1 - (2 * (eigenvalues[0] + eigenvalues[1]) / (2 * eigenvalues[2] + 1e-8))
    acylindricity = (eigenvalues[1] - eigenvalues[0]) / (eigenvalues[2] + 1e-8)
    return gyration_tensor, asphericity, acylindricity

def voxelize_positions(positions, voxel_size=0.5):
    pos = positions - positions.min(axis=0)
    grid_size = np.ceil(np.ptp(pos, axis=0) / voxel_size).astype(int) + 4
    grid = np.zeros(grid_size, dtype=bool)
    indices = (pos / voxel_size).astype(int)
    grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    return grid

def compute_voxel_hollowness(positions, voxel_size=0.5):
    grid = voxelize_positions(positions, voxel_size)
    grid = binary_dilation(grid, structure=ball(1))
    filled = binary_fill_holes(grid)
    shell_volume = np.sum(grid)
    if filled is None:
        return 0.0
    total_volume = np.sum(filled.astype(int))
    if total_volume > shell_volume and total_volume > 0:
        voxel_hollowness = (total_volume - shell_volume) / total_volume
    else:
        voxel_hollowness = 0.0
    return voxel_hollowness
