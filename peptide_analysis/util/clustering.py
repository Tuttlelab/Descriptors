# util/clustering.py

"""
Clustering Utilities for Peptide Analysis

This module provides utility functions related to clustering algorithms,
including identifying aggregates based on distance cutoffs, connected components
analysis, and peptide clustering based on positions and orientations.

Functions:
- identify_aggregates: Identify aggregates in the system using a distance cutoff.
- connected_components: Find connected components in an adjacency matrix.
- cluster_peptides: Cluster peptides based on spatial and orientation similarities.
- compute_distance_matrix: Compute pairwise distance matrix for positions.
- compute_adjacency_matrix: Compute adjacency matrix based on distance cutoff.
"""

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering

def identify_aggregates(positions, distance_cutoff):
    """
    Identify aggregates (clusters) in the system using a distance cutoff.

    Args:
        positions (numpy.ndarray): Array of positions (N x 3).
        distance_cutoff (float): Distance cutoff for clustering in Å.

    Returns:
        list: List of aggregates, each containing indices of atoms in that aggregate.
    """
    distance_matrix = cdist(positions, positions)
    adjacency_matrix = distance_matrix < distance_cutoff
    np.fill_diagonal(adjacency_matrix, False)
    labels, num_labels = connected_components(adjacency_matrix)
    aggregates = defaultdict(list)
    for idx, label_id in enumerate(labels):
        aggregates[label_id].append(idx)
    return list(aggregates.values())

def connected_components(adjacency_matrix):
    """
    Find connected components in an adjacency matrix.

    Args:
        adjacency_matrix (numpy.ndarray): Boolean adjacency matrix (N x N).

    Returns:
        tuple:
            - labels (numpy.ndarray): Array of component labels for each node.
            - num_components (int): Number of connected components found.
    """
    n_nodes = adjacency_matrix.shape[0]
    visited = np.zeros(n_nodes, dtype=bool)
    labels = np.full(n_nodes, -1, dtype=int)
    label = 0
    for node in range(n_nodes):
        if not visited[node]:
            stack = [node]
            while stack:
                current = stack.pop()
                if not visited[current]:
                    visited[current] = True
                    labels[current] = label
                    neighbors = np.where(adjacency_matrix[current])[0]
                    stack.extend(neighbors)
            label += 1
    return labels, label

def cluster_peptides(positions, orientations, spatial_weight, orientation_weight, clustering_threshold):
    """
    Perform clustering of peptides based on spatial proximity and orientation similarity.

    Args:
        positions (numpy.ndarray): Positions of peptides (N x 3).
        orientations (numpy.ndarray): Orientation vectors of peptides (N x 3).
        spatial_weight (float): Weight for spatial distance in the clustering metric.
        orientation_weight (float): Weight for orientation similarity in the clustering metric.
        clustering_threshold (float): Threshold for clustering algorithm.

    Returns:
        numpy.ndarray: Array of cluster labels for each peptide.
    """
    # Compute spatial distance matrix
    spatial_dist = squareform(pdist(positions))

    # Compute orientation angle matrix in degrees
    angle_matrix = compute_orientation_angle_matrix(orientations)

    # Normalize spatial and angle matrices
    if np.max(spatial_dist) > 0:
        spatial_dist /= np.max(spatial_dist)
    if np.max(angle_matrix) > 0:
        angle_matrix /= np.max(angle_matrix)

    # Combine spatial and orientation distances with respective weights
    distance_matrix = spatial_weight * spatial_dist + orientation_weight * angle_matrix

    # Perform agglomerative clustering
    clustering = AgglomerativeClustering(n_clusters=None, linkage='average', distance_threshold=clustering_threshold, affinity='precomputed')
    labels = clustering.fit_predict(distance_matrix)

    return labels

def compute_orientation_angle_matrix(orientations):
    """
    Compute a pairwise angle matrix between orientation vectors.

    Args:
        orientations (numpy.ndarray): Orientation vectors (N x 3).

    Returns:
        numpy.ndarray: Pairwise angle matrix in degrees (N x N).
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

def compute_distance_matrix(positions):
    """
    Compute the pairwise distance matrix for a set of positions.

    Args:
        positions (numpy.ndarray): Positions of atoms (N x 3).

    Returns:
        numpy.ndarray: Pairwise distance matrix (N x N).
    """
    return cdist(positions, positions)

def compute_adjacency_matrix(distance_matrix, distance_cutoff):
    """
    Compute an adjacency matrix based on a distance cutoff.

    Args:
        distance_matrix (numpy.ndarray): Pairwise distance matrix (N x N).
        distance_cutoff (float): Distance cutoff for adjacency.

    Returns:
        numpy.ndarray: Boolean adjacency matrix (N x N).
    """
    adjacency_matrix = distance_matrix < distance_cutoff
    np.fill_diagonal(adjacency_matrix, False)
    return adjacency_matrix
