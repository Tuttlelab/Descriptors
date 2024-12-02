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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
import psutil  # Add at top of file with other imports
from scipy.spatial import cKDTree
import gc
import logging
from logging.handlers import RotatingFileHandler

import warnings
# Remove the import that causes the deprecation warning
# from Bio import BiopythonDeprecationWarning
# Modify the warnings filter to ignore the BiopythonDeprecationWarning
warnings.filterwarnings("ignore", ".*BiopythonDeprecationWarning.*")
warnings.filterwarnings("ignore", category=UserWarning)

# Constants
DEFAULT_MIN_PERSISTENCE = 5  # Minimum number of frames a contact must persist
DEFAULT_RDF_RANGE = (4.0, 15.0)  # Range for RDF calculation in Angstroms
DEFAULT_NBINS = 50  # Number of bins for RDF

def parse_arguments():
    parser = argparse.ArgumentParser(description='Aggregate Detection Index (ADI) Analysis')
    parser.add_argument('-t', '--topology', required=True, help='Topology file (e.g., .gro, .pdb)')
    parser.add_argument('-x', '--trajectory', required=True, help='Trajectory file (e.g., .xtc, .trr)')
    parser.add_argument('-o', '--output', default='adi_results', help='Output directory for results')
    parser.add_argument('-p', '--persistence', type=int, default=DEFAULT_MIN_PERSISTENCE,
                        help='Minimum persistence (in frames) for a contact to be considered stable')
    parser.add_argument('--rdf_range', type=float, nargs=2, default=DEFAULT_RDF_RANGE,
                        help='Range for RDF calculation (start, end)')
    parser.add_argument('--nbins', type=int, default=DEFAULT_NBINS, help='Number of bins for RDF')
    parser.add_argument('--skip', type=int, default=1, help='Process every nth frame (default is every frame)')
    parser.add_argument('--first', type=int, default=0, help='Only analyze the first N frames (default is all frames)')
    parser.add_argument('--last', type=int, default=None, help='Only analyze the last N frames (default is all frames)')
    args = parser.parse_args()
    return args

def ensure_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def setup_logging(output_dir):
    """Configure logging to both file and console."""
    timestamp = datetime.now().strftime("%m%d_%H%M")
    log_file = os.path.join(output_dir, f'adi_analysis_{timestamp}.log')

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Setup file handler
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setFormatter(formatter)

    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Setup logger
    logger = logging.getLogger('ADI_Analysis')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

'''
def load_and_crop_trajectory(topology, trajectory, first, last, skip, selection="protein"):
    u = mda.Universe(topology, trajectory)

    # Set last frame if not specified
    if last is None or last > len(u.trajectory):
        last = len(u.trajectory)
    if first < 0 or first >= len(u.trajectory):
        raise ValueError(f"Invalid first frame: {first}.")

    protein = u.select_atoms(selection)
    if len(protein) == 0:
        raise ValueError(f"Selection '{selection}' returned no atoms.")

    indices = list(range(first, last, skip))

    # Create fixed temporary filenames
    temp_gro = "centered_protein_slice.gro"
    temp_xtc = "centered_protein_slice.xtc"

    with mda.Writer(temp_gro, protein.n_atoms) as W:
        W.write(protein)

    # Apply centering transformation
    print("Centering and wrapping protein in the box...")
    transformations = [
        center_in_box(protein, wrap=True)    # Center selected protein group and wrap the box
    ]
    u.trajectory.add_transformations(*transformations)

    with mda.Writer(temp_xtc, protein.n_atoms) as W:
        for ts in u.trajectory[indices]:
            W.write(protein)

    # Reload the cropped trajectory
    cropped_u = mda.Universe(temp_gro, temp_xtc)
    return cropped_u
'''

def calculate_adaptive_cutoff(universe, selection_string, rdf_range, nbins, output_dir, first, last, skip):
    """
    Calculate an adaptive cutoff distance based on the first minimum after the first peak in the RDF.
    Uses the same frame range as specified in command line arguments.
    """
    print("Calculating adaptive cutoff distance based on RDF...")
    print()
    peptides = universe.select_atoms(selection_string)
    print(f'{len(peptides)} peptide beads selected for RDF analysis.')
    print()

    # Use the same frame range as main analysis
    if last is None or last > len(universe.trajectory):
        last = len(universe.trajectory)
    print(f"Using frames {first} to {last} with step {skip} for RDF calculation...")
    print()

    start_time = datetime.now()
    rdf_analysis = rdf.InterRDF(peptides, peptides, nbins=nbins, range=rdf_range)

    # Run RDF analysis over specified frames
    rdf_analysis.run(start=first, stop=last, step=skip)

    end_time = datetime.now()
    print(f"RDF calculation completed in {end_time - start_time}.")
    print()

    # Save RDF plot
    plt.figure()
    plt.plot(rdf_analysis.results.bins, rdf_analysis.results.rdf)
    plt.xlabel('Distance (Å)')
    plt.ylabel('g(r)')
    plt.title('Radial Distribution Function')
    timestamp = datetime.now().strftime("%m%d_%H%M")
    plt.savefig(os.path.join(output_dir, f'rdf_plot_{timestamp}.png'))
    print("RDF plot saved.")
    print()
    # plt.show()
    plt.close()

    # Identify the first minimum after the first peak
    rdf_values = rdf_analysis.results.rdf
    bins = rdf_analysis.results.bins
    peaks = (np.diff(np.sign(np.diff(rdf_values))) < 0).nonzero()[0] + 1
    if peaks.size > 0:
        first_peak = peaks[0]
        minima = (np.diff(np.sign(np.diff(rdf_values[first_peak:]))) > 0).nonzero()[0] + first_peak + 1
        if minima.size > 0:
            cutoff_distance = bins[minima[0]]
            print(f"Adaptive cutoff distance determined: {cutoff_distance:.2f} Å")
            print()
        else:
            raise ValueError("No minimum found in RDF after the first peak. Please check your RDF or specify a manual cutoff.")
    else:
        raise ValueError("No peaks found in RDF. Please check your RDF or specify a manual cutoff.")
    return cutoff_distance

def identify_clusters(peptides, cutoff_distance, min_peptides=50):
    """
    Identify clusters of peptides based on the cutoff distance.
    Only returns clusters with at least min_peptides members.
    """
    positions = peptides.positions
    dist_matrix = cdist(positions, positions)
    adjacency_matrix = dist_matrix < cutoff_distance
    np.fill_diagonal(adjacency_matrix, 0)
    G = nx.from_numpy_array(adjacency_matrix)
    all_clusters = list(nx.connected_components(G))
    # Filter clusters to include only those with min_peptides or more
    significant_clusters = [cluster for cluster in all_clusters if len(cluster) >= min_peptides]
    # Add debug print before returning
    print(f"Identified {len(significant_clusters)} significant clusters")
    for i, cluster in enumerate(significant_clusters):
        print(f"Cluster {i} size: {len(cluster)}")
    return significant_clusters

def analyze_aggregate_persistence(cluster_records, contact_persistence, min_persistence):
    """
    Apply persistence criteria to filter out transient aggregates.
    """
    # print("Applying persistence criteria to aggregates...")
    # print()
    persistent_aggregates = []
    for cluster_id, frames in cluster_records.items():
        if len(frames) >= min_persistence:
             # Check each contact within the cluster to ensure it meets persistence criteria
            all_contacts_persistent = True
            for a in cluster_id:
                for b in cluster_id:
                    if a < b:  # Avoid duplicates since cluster_id is unordered
                        contact = frozenset([a, b])
                        # Check if this contact's persistence count meets the threshold
                        if contact_persistence.get(contact, 0) < min_persistence:
                            all_contacts_persistent = False
                            break
                if not all_contacts_persistent:
                    break
            # Only add this aggregate if all contacts are persistent
            if all_contacts_persistent:
                persistent_aggregates.append((cluster_id, frames))
    return persistent_aggregates

def save_frame_results(frame_data, output_dir):
    """
    Save frame-by-frame ADI analysis results.
    Format: Frame,Peptides,aggregate_count,total_peptides_in_aggregate,avg_aggregate_size
    """
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_file = os.path.join(output_dir, f'adi_frame_results_{timestamp}.csv')
    with open(output_file, 'w') as f:
        f.write('Frame,Peptides,aggregate_count,total_peptides_in_aggregate,avg_aggregate_size\n')
        for data in frame_data:
            f.write(f"{data['frame']},{data['peptides']},{data['aggregate_count']},"
                   f"{data['total_peptides_in_aggregate']},{data['avg_aggregate_size']:.2f}\n")
    print(f"Frame-by-frame ADI results saved to {output_file}")
    print()

def process_cluster_contacts(peptides, cluster, cutoff_distance, max_pairs=1000000):
    """
    Process contacts for a cluster using KDTree for efficiency.
    Limits maximum number of pairs to avoid memory explosion.
    """
    cluster = list(cluster)
    positions = peptides.positions[cluster]

    # Use KDTree for efficient neighbor search
    tree = cKDTree(positions)
    pairs = tree.query_pairs(cutoff_distance, output_type='set')

    # Convert indices back to original atom indices
    contact_pairs = set()
    for i, j in pairs:
        a, b = cluster[i], cluster[j]
        if a < b:
            contact_pairs.add(frozenset([a, b]))
        if len(contact_pairs) > max_pairs:
            print(f"WARNING: Reached maximum pairs limit ({max_pairs})")
            break

    return contact_pairs

def main():
    args = parse_arguments()
    ensure_output_directory(args.output)

    # Setup logging
    logger = setup_logging(args.output)
    logger.info("Starting ADI analysis")
    logger.info(f"Input topology: {args.topology}")
    logger.info(f"Input trajectory: {args.trajectory}")

    # Load trajectory
    logger.info("Loading trajectory...")
    u = mda.Universe(args.topology, args.trajectory)
    peptides = u.select_atoms('all')
    logger.info(f"Successfully loaded trajectory:")
    logger.info(f"  - Total atoms: {len(peptides)}")
    logger.info(f"  - Total frames: {len(u.trajectory)}")
    logger.info(f"  - Time step: {u.trajectory.dt} ps")

    # Calculate adaptive cutoff distance based on RDF using specified frame range
    cutoff_distance = calculate_adaptive_cutoff(u, 'all', args.rdf_range, args.nbins,
                                              args.output, args.first, args.last, args.skip)
    logger.info(f"Determined adaptive cutoff distance: {cutoff_distance:.2f} Å")

    # Initialize variables for analysis
    cluster_records = defaultdict(list)  # {cluster_id: [frame_numbers]}
    cluster_size_distribution = []  # List of dicts with frame number and cluster sizes
    contact_persistence = defaultdict(int)  # {contact_pair: persistence_count}
    frame_data = []

    try:
        # Analyze each frame
        logger.info("Starting frame analysis")
        print("Analyzing frames for aggregation...")
        print()

        frames = range(args.first, args.last or len(u.trajectory), args.skip)
        for frame_number in frames:
            try:
                u.trajectory[frame_number]
                logger.info(f"Processing frame {frame_number}")
                print(f"Processing frame {frame_number}...")

                # Identify clusters
                current_clusters = identify_clusters(peptides, cutoff_distance, min_peptides=50)
                cluster_sizes = [len(cluster) for cluster in current_clusters]
                logger.info(f"Frame {frame_number}: Found clusters with sizes: {cluster_sizes}")
                print(f"Found clusters with sizes: {cluster_sizes}")

                if cluster_sizes:
                    logger.info("Processing cluster data...")
                    print("Processing cluster data...")
                    cluster_size_distribution.append({'frame': frame_number, 'cluster_sizes': cluster_sizes})

                    current_contacts = set()  # Initialize the set for current frame's contacts
                    total_peptides = len(peptides)
                    aggregate_count = len(current_clusters)
                    logger.info(f"Found {aggregate_count} aggregates in frame {frame_number}.")
                    print(f"Found {aggregate_count} aggregates in frame {frame_number}.")
                    total_in_aggregates = sum(cluster_sizes)
                    avg_size = total_in_aggregates / aggregate_count if aggregate_count > 0 else 0

                    logger.info("Updating frame data...")
                    print("Updating frame data...")
                    frame_data.append({
                        'frame': frame_number,
                        'peptides': total_peptides,
                        'aggregate_count': aggregate_count,
                        'total_peptides_in_aggregate': total_in_aggregates,
                        'avg_aggregate_size': avg_size
                    })

                    logger.info("Processing contacts...")
                    print("Processing contacts...")

                    # Add memory usage monitoring
                    process = psutil.Process()
                    mem_usage = process.memory_info().rss / 1024 / 1024
                    logger.info(f"Memory usage before contact processing: {mem_usage:.2f} MB")
                    print(f"Memory usage before contact processing: {mem_usage:.2f} MB")

                    for cluster_idx, cluster in enumerate(current_clusters):
                        try:
                            logger.info(f"Processing cluster {cluster_idx} with size {len(cluster)}")
                            print(f"Processing cluster {cluster_idx} with size {len(cluster)}")
                            cluster_pairs = process_cluster_contacts(peptides, cluster, cutoff_distance)
                            current_contacts.update(cluster_pairs)
                            logger.info(f"Cluster {cluster_idx} generated {len(cluster_pairs)} contact pairs")
                            print(f"Cluster {cluster_idx} generated {len(cluster_pairs)} contact pairs")

                            # Force garbage collection after each cluster
                            gc.collect()

                        except Exception as e:
                            logger.error(f"Error processing cluster {cluster_idx}: {e}")
                            print(f"Error processing cluster {cluster_idx}: {e}")
                            continue

                    mem_usage = process.memory_info().rss / 1024 / 1024
                    logger.info(f"Memory usage after contact processing: {mem_usage:.2f} MB")
                    print(f"Memory usage after contact processing: {mem_usage:.2f} MB")
                    logger.info(f"Total contacts generated: {len(current_contacts)}")
                    print(f"Total contacts generated: {len(current_contacts)}")
                    logger.info(f"Completed frame {frame_number} processing")
                    print(f"Completed frame {frame_number} processing")
                    print("-" * 50)

            except Exception as e:
                logger.error(f"ERROR in frame {frame_number}: {str(e)}")
                print(f"ERROR in frame {frame_number}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                traceback.print_exc()
                continue

    except Exception as e:
        logger.critical(f"FATAL ERROR: {str(e)}")
        print(f"FATAL ERROR: {str(e)}")
        import traceback
        logger.critical(traceback.format_exc())
        traceback.print_exc()

    # Apply persistence criteria
    persistent_aggregates = analyze_aggregate_persistence(cluster_records, contact_persistence, args.persistence)

    # Save results
    save_frame_results(frame_data, args.output)
    save_cluster_size_distribution(cluster_size_distribution, args.output)
    # save_persistent_aggregates(persistent_aggregates, args.output)

    # Generate plots
    plot_cluster_size_distribution(cluster_size_distribution, args.output)
    # plot_persistent_aggregates(persistent_aggregates, args.output)

    logger.info("ADI analysis completed successfully.")
    print("ADI analysis completed successfully.")

def save_cluster_size_distribution(cluster_size_distribution, output_dir):
    """
    Save cluster size distribution data to a file.
    """
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_file = os.path.join(output_dir, f'cluster_size_distribution_{timestamp}.csv')
    with open(output_file, 'w') as f:
        f.write('Frame,ClusterSizes\n')
        for entry in cluster_size_distribution:
            sizes = ';'.join(map(str, entry['cluster_sizes']))
            f.write(f"{entry['frame']},{sizes}\n")
    print(f"Cluster size distribution data saved to {output_file}")
    print()

def save_persistent_aggregates(persistent_aggregates, output_dir):
    """
    Save information about persistent aggregates.
    """
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_file = os.path.join(output_dir, f'persistent_aggregates_{timestamp}.csv')
    with open(output_file, 'w') as f:
        f.write('AggregateID,Frames\n')
        for idx, (cluster_id, frames) in enumerate(persistent_aggregates):
            frames_str = ';'.join(map(str, frames))
            f.write(f"{idx},{frames_str}\n")
    print(f"Persistent aggregates data saved to {output_file}")
    print()

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
    timestamp = datetime.now().strftime("%m%d_%H%M")
    plt.savefig(os.path.join(output_dir, f'cluster_size_distribution_{timestamp}.png'))
    plt.close()
    print("Cluster size distribution plot saved.")
    print()

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
    timestamp = datetime.now().strftime("%m%d_%H%M")
    plt.savefig(os.path.join(output_dir, f'persistent_aggregates_{timestamp}.png'))
    plt.close()
    print("Persistent aggregates plot saved.")
    print()

if __name__ == '__main__':
    main()
    print()