

import numpy as np
from scipy.spatial.distance import cdist
from scipy.signal import argrelextrema


# --- Advanced TFI helpers ---
def compute_radial_std(positions):
	com = np.mean(positions, axis=0)
	rel_pos = positions - com
	radial_dist = np.linalg.norm(rel_pos, axis=1)
	return np.std(radial_dist)

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

def segment_based_analysis(positions, segment_length=40, step_size=20, radial_threshold=12.0, angular_uniformity_threshold=0.04):
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
		if radial_std < radial_threshold and angular_uniformity > angular_uniformity_threshold:
			tube_like_segments += 1
		num_segments += 1
	if num_segments == 0:
		return 0.0
	return tube_like_segments / num_segments

def compute_radial_density(r, num_bins=50):
	max_radius = r.max() if len(r) > 0 else 1.0
	bins = np.linspace(0, max_radius, num_bins)
	density, bin_edges = np.histogram(r, bins=bins, density=True)
	return density, bin_edges

def is_hollow_tube(density, bin_edges):
	if len(density) < 5:
		return False
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
	asphericity = 1 - (2 * (eigenvalues[0] + eigenvalues[1]) / (2 * eigenvalues[2] + 1e-8))
	ratio = eigenvalues[0] / (eigenvalues[2] + 1e-8)
	return asphericity, ratio


# --- Main modular TFI function ---
def calculate_tfi(
	positions,
	min_tube_size=50,
	radial_threshold=12.0,
	angular_uniformity_threshold=0.04,
	asphericity_threshold=0.5,
	ratio_threshold=0.3,
	segment_length=40,
	step_size=20
):
	"""
	Stateless TFI calculation for a single cluster (per-frame, per-cluster).
	Args:
		positions: np.ndarray (N, 3) of atom positions
		min_tube_size: int, minimum atoms to consider a tube
		radial_threshold: float, threshold for radial std
		angular_uniformity_threshold: float, threshold for angular uniformity
		asphericity_threshold: float, threshold for asphericity
		ratio_threshold: float, threshold for eigenvalue ratio
		segment_length: int, segment size for segment-based analysis
		step_size: int, step size for segment-based analysis
	Returns:
		dict with all TFI metrics (tube-like: bool, and all raw features)
	"""
	n_atoms = len(positions)
	if n_atoms < min_tube_size:
		return {
			'is_tube': False,
			'radial_std': 0.0,
			'angular_uniformity': 0.0,
			'tube_segment_ratio': 0.0,
			'hollow': False,
			'asphericity': 0.0,
			'eigenvalue_ratio': 0.0
		}
	tube_segment_ratio = segment_based_analysis(
		positions,
		segment_length=segment_length,
		step_size=step_size,
		radial_threshold=radial_threshold,
		angular_uniformity_threshold=angular_uniformity_threshold
	)
	radial_std, angular_uniformity, r, theta, z, principal_axis = perform_cylindrical_analysis(positions)
	density, bin_edges = compute_radial_density(r)
	hollow = is_hollow_tube(density, bin_edges)
	asphericity, ratio = compute_shape_anisotropy(positions)
	is_tube = (
		tube_segment_ratio >= 0.5 and
		radial_std < radial_threshold and
		angular_uniformity > angular_uniformity_threshold and
		hollow and
		asphericity > asphericity_threshold and
		ratio < ratio_threshold
	)
	return {
		'is_tube': is_tube,
		'radial_std': radial_std,
		'angular_uniformity': angular_uniformity,
		'tube_segment_ratio': tube_segment_ratio,
		'hollow': hollow,
		'asphericity': asphericity,
		'eigenvalue_ratio': ratio
	}
