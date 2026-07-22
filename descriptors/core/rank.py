
import numpy as np

def rank_clusters(clusters, positions, masses=None, top_n=3):
	# Rank by size, then mass, then COM radius
	def cluster_score(indices):
		size = len(indices)
		mass = np.sum(masses[indices]) if masses is not None else size
		com = np.mean(positions[indices], axis=0)
		radius = np.mean(np.linalg.norm(positions[indices] - com, axis=1))
		return (-size, -mass, radius)
	scores = [cluster_score(c) for c in clusters]
	order = np.argsort(scores)
	return [clusters[i] for i in order[:top_n]]
