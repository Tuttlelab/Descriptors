#!/usr/bin/env python3
"""
adi_analysis.py

This script calculates the Aggregate Detection Index (ADI) for peptide simulations.
It incorporates advanced features such as adaptive cutoff distances based on the
Radial Distribution Function (RDF), persistence criteria for aggregates, cluster
size distribution analysis, and spatial distribution insights.

"""

import os
import csv
import argparse
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import rdf
from MDAnalysis.transformations import unwrap, center_in_box
from scipy.spatial.distance import cdist
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
import logging

import warnings
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Constants
DEFAULT_MIN_PERSISTENCE = 5  # Minimum number of frames a contact must persist
DEFAULT_RDF_RANGE = (4.0, 15.0)  # Range for RDF calculation in Angstroms
DEFAULT_NBINS = 50  # Number of bins for RDF

def parse_arguments():
    parser = argparse.ArgumentParser(description='Aggregate Detection Index (ADI) Analysis')
    parser.add_argument('-t', '--topology', required=True, help='Topology file (e.g., .gro, .pdb)')
    parser.add_argument('-x', '--trajectory', required=True, help='Trajectory file (e.g., .xtc, .trr)')
    parser.add_argument('-s', '--selection', default='protein', help='Atom selection string for peptides')
    parser.add_argument('-p', '--persistence', type=int, default=DEFAULT_MIN_PERSISTENCE,
                        help='Minimum persistence (in frames) for a contact to be considered stable')
    parser.add_argument('-o', '--output', default='adi_results', help='Output directory for results')
    parser.add_argument('--rdf_range', type=float, nargs=2, default=DEFAULT_RDF_RANGE,
                        help='Range for RDF calculation (start, end)')
    parser.add_argument('--nbins', type=int, default=DEFAULT_NBINS, help='Number of bins for RDF')
    parser.add_argument('--skip', type=int, default=1, help='Process every nth frame (default is every frame)')
    parser.add_argument('--first', type=int, default=0, help='Only analyze the first N frames (default is all frames)')
    parser.add_argument('--last', type=int, default=None, help='Only analyze the last N frames (default is all frames)')
    args = parser.parse_args()
    logging.debug(f"Parsed arguments: {args}")
    return args

def ensure_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")
        print(f"Created output directory: {output_dir}")  # Added print statement
    else:
        logging.debug(f"Output directory already exists: {output_dir}")

def load_and_crop_trajectory(topology, trajectory, first, last, skip, selection="protein"):
    logging.info("Loading universe with topology and trajectory.")
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
    logging.info("Applying centering and wrapping transformations.")
    transformations = [
        center_in_box(protein, wrap=True)    # Center selected protein group and wrap the box
    ]
    u.trajectory.add_transformations(*transformations)

    with mda.Writer(temp_xtc, protein.n_atoms) as W:
        for ts in u.trajectory[indices]:
            W.write(protein)

    # Reload the cropped trajectory
    logging.info("Reloading the cropped trajectory.")
    cropped_u = mda.Universe(temp_gro, temp_xtc)
    return cropped_u

def calculate_adaptive_cutoff(universe, selection_string, rdf_range, nbins, output_dir):
    """
    Calculate an adaptive cutoff distance based on the first minimum after the first peak in the RDF.
    """
    logging.info("Calculating adaptive cutoff distance based on RDF.")
    peptides = universe.select_atoms(selection_string)
    logging.debug(f"{len(peptides)} peptide beads selected for RDF analysis.")

    rdf_analysis = rdf.InterRDF(peptides, peptides, nbins=nbins, range=rdf_range)
    rdf_analysis.run()
    logging.info("RDF calculation completed.")

    # Save RDF plot
    plt.figure()
    plt.plot(rdf_analysis.results.bins, rdf_analysis.results.rdf)
    plt.xlabel('Distance (Å)')
    plt.ylabel('g(r)')
    plt.title('Radial Distribution Function')
    timestamp = datetime.now().strftime("%m%d_%H%M")
    plt.savefig(os.path.join(output_dir, f'rdf_plot_{timestamp}.png'))
    logging.info("RDF plot saved.")
    # plt.show()
    plt.close()

    # Identify the first minimum after the first peak
    rdf_values = rdf_analysis.results.rdf
    bins = rdf_analysis.results.bins
    peaks = (np.diff(np.sign(np.diff(rdf_values))) < 0).nonzero()[0] + 1
    if peaks.size > 0:
        logging.debug(f"First peak found at bin index {peaks[0]}.")
        first_peak = peaks[0]
        minima = (np.diff(np.sign(np.diff(rdf_values[first_peak:]))) > 0).nonzero()[0] + first_peak + 1
        if minima.size > 0:
            cutoff_distance = bins[minima[0]]
            logging.info(f"Adaptive cutoff distance determined: {cutoff_distance:.2f} Å")
        else:
            logging.error("No minimum found in RDF after the first peak.")
            raise ValueError("No minimum found in RDF after the first peak. Please check your RDF or specify a manual cutoff.")
    else:
        logging.error("No peaks found in RDF.")
        raise ValueError("No peaks found in RDF. Please check your RDF or specify a manual cutoff.")
    return cutoff_distance

def identify_clusters(peptides, cutoff_distance):
    """
    Identify clusters of peptides based on center of mass and cutoff distance.
    """
    logging.debug(f"Identifying clusters with cutoff distance: {cutoff_distance:.2f} Å.")

    # Get unique residues and their centers of mass
    residues = peptides.residues
    positions = np.array([residue.atoms.center_of_mass() for residue in residues])
    resids = np.array([residue.resid for residue in residues])

    # Create distance matrix based on centers of mass
    dist_matrix = cdist(positions, positions)
    adjacency_matrix = dist_matrix < cutoff_distance
    np.fill_diagonal(adjacency_matrix, 0)

    # Create graph and find connected components (clusters)
    G = nx.from_numpy_array(adjacency_matrix)
    components = list(nx.connected_components(G))

    # Convert node indices to residue IDs
    clusters = [set(resids[list(component)]) for component in components]

    logging.debug(f"Found {len(clusters)} clusters.")
    return clusters

def analyze_aggregate_persistence(cluster_records, contact_persistence, min_persistence):
    """
    Apply persistence criteria to filter out transient aggregates.
    """
    logging.info("Applying persistence criteria to aggregates.")
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
    logging.info(f"Identified {len(persistent_aggregates)} persistent aggregates.")
    return persistent_aggregates

def main():
    args = parse_arguments()
    logging.info("Starting ADI analysis.")
    print("Starting ADI analysis.")
    ensure_output_directory(args.output)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output, f'adi_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
            logging.StreamHandler()
        ]
    )

    # logging.info(f"Output directory: {args.output}")
    # print(f"Output directory: {args.output}")

    # Initialize a list to store frame records
    frame_records = []

    try:
        # Load and crop trajectory
        logging.info("Loading and processing trajectory...")
        u = load_and_crop_trajectory(args.topology, args.trajectory, args.first, args.last, args.skip, args.selection)
        logging.info(f"Total frames in cropped trajectory: {len(u.trajectory)}")

        selection_string = args.selection
        logging.debug(f"Atom selection string: {selection_string}")
        peptides = u.select_atoms(selection_string)

        # Calculate adaptive cutoff distance based on RDF
        cutoff_distance = calculate_adaptive_cutoff(u, selection_string, args.rdf_range, args.nbins, args.output)

        # Initialize variables for analysis
        logging.info("Initializing variables for analysis.")
        cluster_records = defaultdict(list)  # {cluster_id: [frame_numbers]}
        cluster_size_distribution = []  # List of dicts with frame number and cluster sizes
        contact_persistence = defaultdict(int)  # {contact_pair: persistence_count}

        # Analyze each frame
        logging.info("Analyzing frames for aggregation.")
        for frame_number, ts in enumerate(u.trajectory):
            logging.info(f"Processing frame {frame_number}...")

            # Get peptide residues instead of atoms
            peptide_residues = peptides.residues

            # Identify clusters using residue-based approach
            current_clusters = identify_clusters(peptides, cutoff_distance)

            # Convert residue IDs to peptide identifiers
            aggregated_peptides = set()
            for cluster in current_clusters:
                aggregated_peptides.update(f'PEP{resid}' for resid in cluster)

            # Calculate ADI metrics
            aggregate_size = len(aggregated_peptides)
            aggregation_rate = aggregate_size / (frame_number + 1)
            stability_index = aggregate_size / len(peptide_residues)

            # Create frame record with proper peptide identification
            frame_record = {
                'Frame': frame_number,
                'Peptides': sorted(list(aggregated_peptides)),  # Sort for consistency
                'aggregate_size': aggregate_size,
                'aggregation_rate': aggregation_rate,
                'stability_index': stability_index
            }

            frame_records.append(frame_record)

            # Identify clusters in the current frame
            current_clusters = identify_clusters(peptides, cutoff_distance)
            logging.debug(f"Found {len(current_clusters)} clusters in frame {frame_number}.")
            cluster_sizes = [len(cluster) for cluster in current_clusters]
            cluster_size_distribution.append({'frame': frame_number, 'cluster_sizes': cluster_sizes})

            #ADI-specific metrics (replace with actual calculations)
            aggregate_size = sum(cluster_sizes)
            aggregation_rate = aggregate_size / (frame_number + 1)  # Example metric
            stability_index = aggregate_size / (len(peptides) + 1)  # Example metric

            # Extract peptides involved in aggregates
            aggregated_peptides = []
            for cluster in current_clusters:
                aggregated_peptides.extend(cluster)
            aggregated_peptides = list(set(aggregated_peptides))  # Remove duplicates

            # Create a record for the current frame
            frame_record = {
                'Frame': frame_number,
                'Peptides': str(aggregated_peptides),
                'aggregate_size': aggregate_size,
                'aggregation_rate': aggregation_rate,
                'stability_index': stability_index
                # Add other ADI-specific metrics here
            }

            frame_records.append(frame_record)

            # Track contact persistence across frames
            current_contacts = set()
            for cluster in current_clusters:
                cluster_pairs = {frozenset([a, b]) for i, a in enumerate(cluster) for b in cluster if a != b}
                current_contacts.update(cluster_pairs)

            # Update persistence counts for ongoing contacts
            for contact in current_contacts:
                contact_persistence[contact] += 1  # Increase count if contact is present in this frame

            # Reset counts for contacts no longer present
            for contact in list(contact_persistence.keys()):
                if contact not in current_contacts:
                    contact_persistence[contact] = 0  # Reset persistence count for broken contact

            # Record clusters for persistence analysis
            for cluster in current_clusters:
                cluster_id = frozenset(cluster)
                cluster_records[cluster_id].append(frame_number)

        # Apply persistence criteria
        persistent_aggregates = analyze_aggregate_persistence(cluster_records, contact_persistence, args.persistence)

        # Save results
        logging.info("Saving analysis results.")
        print("Saving analysis results.")
        save_cluster_size_distribution(cluster_size_distribution, args.output)
        save_persistent_aggregates(persistent_aggregates, args.output)
        # Save ADI frame results
        save_frame_results(frame_records, args.output)

        # Generate plots
        logging.info("Generating plots.")
        print("Generating plots.")
        plot_cluster_size_distribution(cluster_size_distribution, args.output)  # Updated function call
        plot_persistent_aggregates(persistent_aggregates, args.output)

        logging.info("ADI analysis completed successfully.")
        print("ADI analysis completed successfully.")

    finally:
        # Clean up the fixed temporary files after the analysis
        os.remove("centered_protein_slice.gro")
        os.remove("centered_protein_slice.xtc")

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
    logging.info("Cluster size distribution data saved.")

def save_frame_results(frame_records, output_dir):
    """
    Save ADI frame results to a CSV file with proper peptide identification.
    """
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_file = os.path.join(output_dir, f'adi_frame_results_{timestamp}.csv')

    fieldnames = ['Frame', 'Peptides', 'aggregate_size', 'aggregation_rate', 'stability_index']

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in frame_records:
            # Ensure Peptides is a string representation of a list
            if isinstance(record['Peptides'], (list, set)):
                record['Peptides'] = str(sorted(record['Peptides']))
            writer.writerow(record)

    logging.info(f"ADI frame results saved to {output_file}")

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
    logging.info("Persistent aggregates data saved.")

def plot_cluster_size_distribution(cluster_size_distribution, output_dir):
    """
    Plot the number of clusters per frame with circle sizes proportional to the average cluster size.
    """
    logging.info("Plotting cluster size distribution.")
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from datetime import datetime

    # Extract frame numbers, number of clusters, and average cluster sizes
    frames = [entry['frame'] for entry in cluster_size_distribution]
    num_clusters = [len(entry['cluster_sizes']) for entry in cluster_size_distribution]
    avg_cluster_sizes = [np.mean(entry['cluster_sizes']) if entry['cluster_sizes'] else 0 for entry in cluster_size_distribution]

    # Scale the sizes of the scatter points for better visibility
    sizes_scaled = np.array(avg_cluster_sizes) * 100  # Adjust scaling factor as needed

    # Create the bubble plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(frames, num_clusters, s=sizes_scaled, alpha=0.6, c=num_clusters, cmap='viridis', edgecolors='w', linewidth=0.5)

    # Add color bar to indicate the number of clusters
    plt.colorbar(scatter, label='Number of Clusters')

    # Enhance plot aesthetics
    plt.xlabel("Frame Number")
    plt.ylabel("Number of Clusters")
    plt.title("Number of Clusters per Frame with Average Cluster Size")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Save the plot with a timestamped filename
    timestamp = datetime.now().strftime("%m%d_%H%M")
    plot_filename = os.path.join(output_dir, f'number_of_clusters_per_frame_{timestamp}.png')
    plt.savefig(plot_filename)
    plt.close()
    logging.info("Cluster size distribution plot saved.")

    # Save cluster size distribution data to a CSV file
    csv_filename = os.path.join(output_dir, f'cluster_size_distribution_{timestamp}.csv')
    with open(csv_filename, 'w') as f:
        f.write('Frame,ClusterSizes\n')
        for entry in cluster_size_distribution:
            sizes = ';'.join(map(str, entry['cluster_sizes']))
            f.write(f"{entry['frame']},{sizes}\n")
    logging.info(f"Cluster size distribution data saved to {csv_filename}")

def plot_persistent_aggregates(persistent_aggregates, output_dir):
    """
    Plot the number of persistent aggregates over time.
    """
    logging.info("Plotting persistent aggregates over time.")
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
    logging.info("Persistent aggregates plot saved.")

if __name__ == '__main__':
    main()
