
import numpy as np
import logging
from MDAnalysis.analysis import rdf
from descriptors.core.rdf import compute_rdf, find_first_minimum
from scipy.spatial.distance import cdist
import networkx as nx

def determine_r_cut(peptides, dynamic_cutoff, static_cutoff, rdf_range, nbins, logger):
    if dynamic_cutoff:
        logger.info(f"Computing RDF with range {rdf_range} and {nbins} bins.")
        r, g_r = compute_rdf(peptides, rdf_range=rdf_range, nbins=nbins)
        r_cut = find_first_minimum(r, g_r)
        logger.info(f"Dynamic cutoff determined: r_cut = {r_cut:.2f} Å")
    else:
        r_cut = static_cutoff
        logger.info(f"Using static cutoff: r_cut = {r_cut:.2f} Å")
    return r_cut

def build_neighbor_graph(positions, r_cut):
    dist_matrix = cdist(positions, positions, metric='euclidean')
    neighbors = dist_matrix < r_cut
    np.fill_diagonal(neighbors, False)
    return neighbors

def find_clusters(neighbors):
    G = nx.from_numpy_array(neighbors)
    clusters = [list(comp) for comp in nx.connected_components(G)]
    return clusters

logging.basicConfig(level=logging.DEBUG)

def calculate_adi(
    peptides,
    box,
    min_persistence=None,
    dynamic_cutoff=True,
    static_cutoff=6.3,
    rdf_range=(0.0, 12.0),
    nbins=100,
    logger=None
):
    """
    Stateless ADI calculation for a single frame or selection.
    Args:
        peptides: MDAnalysis AtomGroup (selection for peptides)
        box: simulation box dimensions (array-like)
        min_persistence: minimum frames for a contact to be considered stable (optional)
        dynamic_cutoff: if True, use RDF to determine cutoff; else use static_cutoff
        static_cutoff: fallback cutoff if not using RDF
        rdf_range: tuple, range for RDF calculation
        nbins: number of bins for RDF
        logger: optional logger for detailed output
    Returns:
        dict with ADI metrics (e.g., cluster sizes, cutoff used, etc.)
    """
    if logger is None:
        logger = logging.getLogger("adi")
    logger.info(f"Starting ADI calculation with {len(peptides)} atoms.")

    # --- Cutoff determination ---
    r_cut = determine_r_cut(peptides, dynamic_cutoff, static_cutoff, rdf_range, nbins, logger)

    # --- Neighbor graph ---
    positions = peptides.positions
    logger.info("Computing distance matrix and neighbor graph.")
    neighbors = build_neighbor_graph(positions, r_cut)

    # --- Connected components using NetworkX ---
    logger.info("Building NetworkX graph and finding connected components.")
    clusters = find_clusters(neighbors)
    logger.info(f"Found {len(clusters)} clusters.")

    # --- Persistence filtering (optional) ---
    if min_persistence and min_persistence > 1:
        logger.info(f"Filtering clusters by min_persistence={min_persistence} (not implemented in this frame-only function).")
        # Placeholder: Persistence requires multi-frame logic.
        # Here, just log and skip.
        pass

    # --- Output ---
    cluster_sizes = [len(c) for c in clusters]
    result = {
        'r_cut': r_cut,
        'n_clusters': len(clusters),
        'cluster_sizes': cluster_sizes,
        'clusters': clusters,
    }
    logger.info(f"ADI calculation complete. Largest cluster: {max(cluster_sizes) if cluster_sizes else 0} atoms.")
    return result