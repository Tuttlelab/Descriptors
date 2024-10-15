#!/usr/bin/env python3
"""
adi_analysis.py

This script calculates the Aggregate Detection Index (ADI) for peptide simulations.
It incorporates advanced features such as adaptive cutoff distances based on the
Radial Distribution Function (RDF), persistence criteria for aggregates, cluster
size distribution analysis, and spatial distribution insights.

"""

import os
import argparse
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import rdf
from scipy.spatial.distance import cdist
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

# Constants
DEFAULT_MIN_PERSISTENCE = 5  # Minimum number of frames a contact must persist
DEFAULT_RDF_RANGE = (4.0, 15.0)  # Range for RDF calculation in Angstroms
DEFAULT_NBINS = 50  # Number of bins for RDF

def parse_arguments():
    parser = argparse.ArgumentParser(description='Aggregate Detection Index (ADI) Analysis')
    parser.add_argument('-t', '--topology', required=True, help='Topology file (e.g., .gro, .pdb)')
    parser.add_argument('-x', '--trajectory', required=True, help='Trajectory file (e.g., .xtc, .trr)')
    parser.add_argument('-s', '--selection', default='all', help='Atom selection string for peptides')
    parser.add_argument('-p', '--persistence', type=int, default=DEFAULT_MIN_PERSISTENCE,
                        help='Minimum persistence (in frames) for a contact to be considered stable')
    parser.add_argument('-o', '--output', default='adi_results', help='Output directory for results')
    parser.add_argument('--rdf_range', type=float, nargs=2, default=DEFAULT_RDF_RANGE,
                        help='Range for RDF calculation (start, end)')
    parser.add_argument('--nbins', type=int, default=DEFAULT_NBINS, help='Number of bins for RDF')
    args = parser.parse_args()
    return args

def ensure_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def calculate_adaptive_cutoff(universe, selection_string, rdf_range, nbins, output_dir):
    """
    Calculate an adaptive cutoff distance based on the first minimum after the first peak in the RDF.
    """
    print("Calculating adaptive cutoff distance based on RDF...")
    peptides = universe.select_atoms(selection_string)
    
    rdf_analysis = rdf.InterRDF(peptides, peptides, nbins=nbins, range=rdf_range)
    rdf_analysis.run()
    
    # Save RDF plot
    plt.figure()
    plt.plot(rdf_analysis.bins, rdf_analysis.rdf)
    plt.xlabel('Distance (Å)')
    plt.ylabel('g(r)')
    plt.title('Radial Distribution Function')
    plt.savefig(os.path.join(output_dir, 'rdf_plot.png'))
    plt.show()
    plt.close()
    
    # Identify the first minimum after the first peak
    rdf_values = rdf_analysis.rdf
    bins = rdf_analysis.bins
    peaks = (np.diff(np.sign(np.diff(rdf_values))) < 0).nonzero()[0] + 1
    if peaks.size > 0:
        first_peak = peaks[0]
        minima = (np.diff(np.sign(np.diff(rdf_values[first_peak:]))) > 0).nonzero()[0] + first_peak + 1
        if minima.size > 0:
            cutoff_distance = bins[minima[0]]
            print(f"Adaptive cutoff distance determined: {cutoff_distance:.2f} Å")
        else:
            raise ValueError("No minimum found in RDF after the first peak. Please check your RDF or specify a manual cutoff.")
    else:
        raise ValueError("No peaks found in RDF. Please check your RDF or specify a manual cutoff.")
    return cutoff_distance

def identify_clusters(peptides, cutoff_distance):
    """
    Identify clusters of peptides based on the cutoff distance.
    """
    positions = peptides.positions
    dist_matrix = cdist(positions, positions)
    adjacency_matrix = dist_matrix < cutoff_distance
    np.fill_diagonal(adjacency_matrix, 0)
    G = nx.from_numpy_array(adjacency_matrix)
    clusters = list(nx.connected_components(G))
    return clusters

def analyze_aggregate_persistence(cluster_records, min_persistence):
    """
    Apply persistence criteria to filter out transient aggregates.
    """
    print("Applying persistence criteria to aggregates...")
    persistent_aggregates = []
    for cluster_id, frames in cluster_records.items():
        if len(frames) >= min_persistence:
            persistent_aggregates.append((cluster_id, frames))
    return persistent_aggregates

def main():
    args = parse_arguments()
    ensure_output_directory(args.output)

    # Load the trajectory
    print("Loading trajectory data...")
    u = mda.Universe(args.topology, args.trajectory)
    selection_string = args.selection
    peptides = u.select_atoms(selection_string)
    n_frames = len(u.trajectory)
    print(f"Total frames in trajectory: {n_frames}")

    # Calculate adaptive cutoff distance
    cutoff_distance = calculate_adaptive_cutoff(u, selection_string, args.rdf_range, args.nbins, args.output)

    # Initialize variables for analysis
    cluster_records = defaultdict(list)  # {cluster_id: [frame_numbers]}
    cluster_size_distribution = []  # List of dicts with frame number and cluster sizes
    contact_persistence = defaultdict(int)

    # Analyze each frame
    print("Analyzing frames for aggregation...")
    for frame_number, ts in enumerate(u.trajectory):
        current_clusters = identify_clusters(peptides, cutoff_distance)
        cluster_sizes = [len(cluster) for cluster in current_clusters]
        cluster_size_distribution.append({'frame': frame_number, 'cluster_sizes': cluster_sizes})

        # Record clusters for persistence analysis
        for cluster in current_clusters:
            cluster_id = frozenset(cluster)
            cluster_records[cluster_id].append(frame_number)

    # Apply persistence criteria
    persistent_aggregates = analyze_aggregate_persistence(cluster_records, args.persistence)

    # Save results
    save_cluster_size_distribution(cluster_size_distribution, args.output)
    save_persistent_aggregates(persistent_aggregates, args.output)

    # Generate plots
    plot_cluster_size_distribution(cluster_size_distribution, args.output)
    plot_persistent_aggregates(persistent_aggregates, args.output)

    print("ADI analysis completed successfully.")

def save_cluster_size_distribution(cluster_size_distribution, output_dir):
    """
    Save cluster size distribution data to a file.
    """
    output_file = os.path.join(output_dir, 'cluster_size_distribution.csv')
    with open(output_file, 'w') as f:
        f.write('Frame,ClusterSizes\n')
        for entry in cluster_size_distribution:
            sizes = ';'.join(map(str, entry['cluster_sizes']))
            f.write(f"{entry['frame']},{sizes}\n")
    print(f"Cluster size distribution data saved to {output_file}")

def save_persistent_aggregates(persistent_aggregates, output_dir):
    """
    Save information about persistent aggregates.
    """
    output_file = os.path.join(output_dir, 'persistent_aggregates.csv')
    with open(output_file, 'w') as f:
        f.write('AggregateID,Frames\n')
        for idx, (cluster_id, frames) in enumerate(persistent_aggregates):
            frames_str = ';'.join(map(str, frames))
            f.write(f"{idx},{frames_str}\n")
    print(f"Persistent aggregates data saved to {output_file}")

def plot_cluster_size_distribution(cluster_size_distribution, output_dir):
    """
    Plot the cluster size distribution over time.
    """
    frames = [entry['frame'] for entry in cluster_size_distribution]
    max_cluster_sizes = [max(entry['cluster_sizes']) if entry['cluster_sizes'] else 0 for entry in cluster_size_distribution]
    plt.figure()
    plt.plot(frames, max_cluster_sizes, label='Max Cluster Size')
    plt.xlabel('Frame')
    plt.ylabel('Cluster Size')
    plt.title('Cluster Size Distribution Over Time')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'cluster_size_distribution.png'))
    plt.close()
    print("Cluster size distribution plot saved.")

def plot_persistent_aggregates(persistent_aggregates, output_dir):
    """
    Plot the number of persistent aggregates over time.
    """
    aggregate_counts = defaultdict(int)
    for _, frames in persistent_aggregates:
        for frame in frames:
            aggregate_counts[frame] += 1
    frames = sorted(aggregate_counts.keys())
    counts = [aggregate_counts[frame] for frame in frames]
    plt.figure()
    plt.plot(frames, counts, label='Number of Persistent Aggregates')
    plt.xlabel('Frame')
    plt.ylabel('Count')
    plt.title('Persistent Aggregates Over Time')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'persistent_aggregates.png'))
    plt.close()
    print("Persistent aggregates plot saved.")

if __name__ == '__main__':
    main()
