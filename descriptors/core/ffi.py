
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull


def compute_moments_of_inertia(positions):
	com = np.mean(positions, axis=0)
	rel_pos = positions - com
	inertia_tensor = np.zeros((3, 3))
	for r in rel_pos:
		inertia_tensor += np.outer(r, r)
	inertia_tensor /= len(rel_pos)
	eigvals, eigvecs = np.linalg.eigh(inertia_tensor)
	eigvals = np.sort(eigvals)
	principal_axis = eigvecs[:, -1]
	return eigvals, principal_axis

def analyze_orientation_distribution(orientations, principal_axis):
	cos_angles = np.dot(orientations, principal_axis)
	cos_angles = np.clip(cos_angles, -1.0, 1.0)
	angles = np.arccos(cos_angles) * (180 / np.pi)
	mean_angle = np.mean(angles)
	std_angle = np.std(angles)
	return mean_angle, std_angle, angles

def compute_fop(orientations, principal_axis):
	cos_angles = np.dot(orientations, principal_axis)
	cos2_angles = (3 * cos_angles**2 - 1) / 2
	return np.mean(cos2_angles)

def cross_sectional_profiling(positions, principal_axis, thickness=5.0, num_sections=10):
	rel_pos = positions - np.mean(positions, axis=0)
	z = np.dot(rel_pos, principal_axis)
	z_min, z_max = z.min(), z.max()
	cross_section_areas = []
	for i in range(num_sections):
		z_i = z_min + i * (z_max - z_min) / num_sections
		indices = np.where((z >= z_i - thickness / 2) & (z < z_i + thickness / 2))[0]
		section_pos = rel_pos[indices]
		if len(section_pos) >= 3:
			try:
				hull = ConvexHull(section_pos)
				cross_section_areas.append(hull.area)
			except Exception:
				cross_section_areas.append(0.0)
		else:
			cross_section_areas.append(0.0)
	return cross_section_areas

def calculate_ffi(positions, min_fiber_size=1000, shape_ratio_threshold=1.5):
	"""
	Stateless FFI calculation for a single frame or cluster.
	Args:
		positions: np.ndarray (N, 3) of atom positions
		min_fiber_size: int, minimum atoms to consider a fiber
		shape_ratio_threshold: float, threshold for shape ratios
	Returns:
		dict with FFI metrics (fiber-like: bool, shape ratios, etc.)
	"""
	if len(positions) < min_fiber_size:
		return {'is_fiber': False, 'shape_ratios': (0.0, 0.0, 0.0)}
	eigvals, principal_axis = compute_moments_of_inertia(positions)
	if eigvals[1] == 0 or eigvals[2] == 0:
		return {'is_fiber': False, 'shape_ratios': tuple(eigvals)}
	ratio1 = eigvals[2] / eigvals[1]
	ratio2 = eigvals[1] / eigvals[0]
	# Orientation analysis (if orientations provided)
	mean_angle = std_angle = fop = None
	if orientations is not None:
		mean_angle, std_angle, _ = analyze_orientation_distribution(orientations, principal_axis)
		fop = compute_fop(orientations, principal_axis)
	# Cross-sectional profiling
	cross_section_areas = cross_sectional_profiling(positions, principal_axis)
	# Fiber classification
	is_fiber = (ratio1 > shape_ratio_threshold and ratio2 > shape_ratio_threshold)
	if std_angle is not None and fop is not None:
		is_fiber = is_fiber and (std_angle < alignment_std_threshold) and (abs(fop) > fop_threshold)
	return {
		'is_fiber': is_fiber,
		'shape_ratios': (ratio1, ratio2, eigvals[0]),
		'eigvals': eigvals,
		'principal_axis': principal_axis,
		'mean_angle': mean_angle,
		'std_angle': std_angle,
		'fop': fop,
		'cross_section_areas': cross_section_areas,
	}
