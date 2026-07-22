"""
descriptors/adi.py

Aggregate Dynamics Index (ADI) Analysis for peptide self-assembly simulations.
Calculates cluster formation, size distribution, and aggregate dynamics over time.
"""

import os
import argparse
import logging
import gc
from datetime import datetime
from collections import defaultdict
import numpy as np
import MDAnalysis as mda
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from descriptors.utils import suppress_warnings, ensure_output_directory, setup_logger

suppress_warnings()

DEFAULT_MIN_PERSISTENCE = 5
STATIC_CUTOFF = 4.5  # Å

def parse_arguments():
    parser = argparse.ArgumentParser(description='Aggregate Dynamics Index (ADI) Analysis')
    parser.add_argument('-t', '--topology', required=True, help='Topology file (e.g., .gro, .pdb)')
    parser.add_argument('-x', '--trajectory', required=True, help='Trajectory file (e.g., .xtc, .trr)')
    parser.add_argument('-o', '--output', default='adi_results', help='Output directory for results')
    parser.add_argument('-p', '--persistence', type=int, default=DEFAULT_MIN_PERSISTENCE,
                        help='Minimum persistence (in frames) for a contact to be considered stable')
    parser.add_argument('--skip', type=int, default=1, help='Process every nth frame')
    parser.add_argument('--first', type=int, default=0, help='First frame index')
    parser.add_argument('--last', type=int, default=None, help='Last frame index')
    return parser.parse_args()

def identify_clusters(peptides, cutoff_distance, min_peptides=50):
    positions = peptides.positions
    dist_matrix = cdist(positions, positions)
    adjacency_matrix = dist_matrix < cutoff_distance
    np.fill_diagonal(adjacency_matrix, 0)
    G = nx.from_numpy_array(adjacency_matrix)
    all_clusters = list(nx.connected_components(G))
    significant_clusters = [cluster for cluster in all_clusters if len(cluster) >= min_peptides]
    return significant_clusters

def process_cluster_contacts(peptides, cluster, cutoff_distance, max_pairs=1000000):
    cluster = list(cluster)
    positions = peptides.positions[cluster]
    tree = cKDTree(positions)
    pairs = tree.query_pairs(cutoff_distance, output_type='set')
    contact_pairs = set()
    for i, j in pairs:
        a, b = cluster[i], cluster[j]
        if a < b:
            contact_pairs.add(frozenset([a, b]))
        if len(contact_pairs) > max_pairs:
            break
    return contact_pairs

def calculate_adi(topology, trajectory, output_dir='adi_results', persistence=DEFAULT_MIN_PERSISTENCE,
                  first=0, last=None, skip=1):
    """Programmatic API for ADI analysis."""
    ensure_output_directory(output_dir)
    logger = setup_logger("adi", output_dir)
    logger.info(f"Starting ADI analysis on {trajectory} with topology {topology}")

    u = mda.Universe(topology, trajectory)
    peptides = u.select_atoms('all')

    cluster_size_distribution = []
    frame_data = []
    frames = range(first, last or len(u.trajectory), skip)

    for frame_number in frames:
        u.trajectory[frame_number]
        clusters = identify_clusters(peptides, STATIC_CUTOFF, min_peptides=50)
        cluster_sizes = [len(c) for c in clusters]

        total_peptides = len(peptides)
        aggregate_count = len(clusters)
        total_in_aggregates = sum(cluster_sizes)
        avg_size = total_in_aggregates / aggregate_count if aggregate_count > 0 else 0

        cluster_size_distribution.append({'frame': frame_number, 'cluster_sizes': cluster_sizes})
        frame_data.append({
            'frame': frame_number,
            'peptides': total_peptides,
            'aggregate_count': aggregate_count,
            'total_peptides_in_aggregate': total_in_aggregates,
            'avg_aggregate_size': avg_size
        })

    save_frame_results(frame_data, output_dir)
    save_cluster_size_distribution(cluster_size_distribution, output_dir)
    plot_cluster_size_distribution(cluster_size_distribution, output_dir)

    logger.info("ADI analysis completed successfully.")
    return frame_data

def save_frame_results(frame_data, output_dir):
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_file = os.path.join(output_dir, f'adi_frame_results_{timestamp}.csv')
    with open(output_file, 'w') as f:
        f.write('Frame,Peptides,aggregate_count,total_peptides_in_aggregate,avg_aggregate_size\n')
        for data in frame_data:
            f.write(f"{data['frame']},{data['peptides']},{data['aggregate_count']},"
                   f"{data['total_peptides_in_aggregate']},{data['avg_aggregate_size']:.2f}\n")

def save_cluster_size_distribution(cluster_size_distribution, output_dir):
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_file = os.path.join(output_dir, f'cluster_size_distribution_{timestamp}.csv')
    with open(output_file, 'w') as f:
        f.write('Frame,ClusterSizes\n')
        for entry in cluster_size_distribution:
            sizes = ';'.join(map(str, entry['cluster_sizes']))
            f.write(f"{entry['frame']},{sizes}\n")

def plot_cluster_size_distribution(cluster_size_distribution, output_dir):
    frames = [entry['frame'] for entry in cluster_size_distribution]
    max_sizes = [max(entry['cluster_sizes']) if entry['cluster_sizes'] else 0 for entry in cluster_size_distribution]
    plt.figure()
    plt.plot(frames, max_sizes, label='Max Cluster Size')
    plt.xlabel('Frame')
    plt.ylabel('Cluster Size')
    plt.title('Cluster Size Distribution Over Time')
    plt.legend()
    timestamp = datetime.now().strftime("%m%d_%H%M")
    plt.savefig(os.path.join(output_dir, f'cluster_size_distribution_{timestamp}.png'))
    plt.close()

def main():
    args = parse_arguments()
    calculate_adi(args.topology, args.trajectory, args.output, args.persistence,
                  args.first, args.last, args.skip)

if __name__ == '__main__':
    main()
