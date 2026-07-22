
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from descriptors.core.rdf import compute_rdf
from scipy.optimize import curve_fit


def compute_angle_matrix(orientations):
	dot_products = np.dot(orientations, orientations.T)
	norms = np.linalg.norm(orientations, axis=1)
	norms_matrix = np.outer(norms, norms)
	norms_matrix[norms_matrix == 0] = 1
	cos_angles = dot_products / norms_matrix
	cos_angles = np.clip(cos_angles, -1.0, 1.0)
	angles = np.degrees(np.arccos(cos_angles))
	return angles

# Quadratic surface fitting for curvature/planarity
def fit_quadratic_surface(positions):
	if len(positions) < 6:
		return np.inf, None
	def quad(X, a, b, c, d, e, f):
		x, y = X
		return a * x**2 + b * y**2 + c * x * y + d * x + e * y + f
	x = positions[:, 0]
	y = positions[:, 1]
	z = positions[:, 2]
	X = np.vstack((x, y))
	try:
		params, _ = curve_fit(quad, X, z)
		z_fit = quad(X, *params)
		residuals = z - z_fit
		rmsd = np.sqrt(np.mean(residuals**2))
		return rmsd, params
	except Exception:
		return np.inf, None

# Euler characteristic (simple version)
def euler_characteristic(cluster_indices, positions, cutoff=10.0):
	V = len(cluster_indices)
	if V < 2:
		return 1
	pos = positions[cluster_indices]
	dists = squareform(pdist(pos))
	E = np.sum((dists < cutoff) & (dists > 0)) // 2
	F = 0
	return V - E + F

# Bilayer detection stub
def detect_bilayer(positions, orientations):
	# Placeholder: real bilayer detection would analyze orientation distribution and spatial separation
	return False, 0.0

def calculate_sfi(positions, orientations, spatial_cutoff=15, angle_cutoff=45, min_sheet_size=5, rdf_range=(0.0, 30.0), nbins=100):
	"""
	Extended SFI calculation for a single frame or cluster.
	Args:
		positions: np.ndarray (N, 3) of peptide positions
		orientations: np.ndarray (N, 3) of peptide orientation vectors
		spatial_cutoff: float, max distance for sheet clustering
		angle_cutoff: float, max angle (deg) for orientation similarity
		min_sheet_size: int, minimum peptides to consider a sheet
		rdf_range: tuple, RDF calculation range
		nbins: int, RDF bins
	Returns:
		dict with SFI metrics (sheet count, sizes, clusters, RDF, curvature, Euler, bilayer)
	"""
	clusters = cluster_peptides(positions, orientations, spatial_cutoff, angle_cutoff, min_sheet_size)
	sheet_sizes = [len(c) for c in clusters]
	# Global RDF
	r, g_r = compute_rdf(positions, rdf_range=rdf_range, nbins=nbins)
	# Per-sheet metrics
	sheet_curvature = []
	sheet_euler = []
	sheet_bilayer = []
	for c in clusters:
		rmsd, params = fit_quadratic_surface(positions[c])
		sheet_curvature.append({'rmsd': rmsd, 'params': params})
		euler = euler_characteristic(c, positions)
		sheet_euler.append(euler)
		is_bilayer, separation = detect_bilayer(positions[c], orientations[c])
		sheet_bilayer.append({'is_bilayer': is_bilayer, 'separation': separation})
	result = {
		'n_sheets': len(clusters),
		'sheet_sizes': sheet_sizes,
		'clusters': clusters,
		'rdf_r': r,
		'rdf_g_r': g_r,
		'sheet_curvature': sheet_curvature,
		'sheet_euler': sheet_euler,
		'sheet_bilayer': sheet_bilayer,
	}
	return result
