

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
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


def compute_radial_density(positions, com, num_bins=50):
    r = np.linalg.norm(positions - com, axis=1)
    density, bin_edges = np.histogram(r, bins=num_bins, density=True)
    return density, bin_edges


def is_hollow(density, bin_edges, window_size=7, hollow_ratio=0.05):
	# Advanced: look for a dip in the center of the density profile (RDP)
	if len(density) < window_size:
		return False
	kernel = np.ones(window_size)/window_size
	density_smooth = np.convolve(density, kernel, mode='same')
	center_bin = len(density_smooth) // 2
	window = density_smooth[center_bin-window_size//2:center_bin+window_size//2+1]
	if len(window) == 0:
		return False
	min_density = np.min(window)
	max_density = np.max(density_smooth)
	return min_density < hollow_ratio * max_density


def calculate_vfi(
	positions,
	min_vesicle_size=30,
	sphericity_threshold=0.5,
	hollowness_threshold=0.05,
	voxel_size=0.5,
	num_bins=50
):
	"""
	Stateless VFI calculation for a single frame or cluster.
	Args:
		positions: np.ndarray (N, 3) of atom positions
		min_vesicle_size: int, minimum atoms to consider a vesicle
		sphericity_threshold: float, minimum sphericity for vesicle
		hollowness_threshold: float, threshold for hollowness
		voxel_size: float, voxel size for voxelization
		num_bins: int, number of bins for radial density
	Returns:
		dict with VFI metrics (vesicle-like: bool, sphericity, hollowness, asphericity, acylindricity, etc.)
	"""
	if len(positions) < min_vesicle_size:
		return {
			'is_vesicle': False,
			'sphericity': 0.0,
			'hollow': False,
			'voxel_hollowness': 0.0,
			'asphericity': 0.0,
			'acylindricity': 0.0,
			'density_profile': np.zeros(num_bins),
			'bin_edges': np.zeros(num_bins+1)
		}
	com = np.mean(positions, axis=0)
	sphericity = compute_sphericity(positions)
	density, bin_edges = compute_radial_density(positions, com, num_bins=num_bins)
	hollow = is_hollow(density, bin_edges, hollow_ratio=hollowness_threshold)
	voxel_hollowness = compute_voxel_hollowness(positions, voxel_size=voxel_size)
	_, asphericity, acylindricity = compute_gyration_tensor(positions)
	is_vesicle = sphericity > sphericity_threshold and (hollow or voxel_hollowness > hollowness_threshold)
	return {
		'is_vesicle': is_vesicle,
		'sphericity': sphericity,
		'hollow': hollow,
		'voxel_hollowness': voxel_hollowness,
		'asphericity': asphericity,
		'acylindricity': acylindricity,
		'density_profile': density,
		'bin_edges': bin_edges,
	}
