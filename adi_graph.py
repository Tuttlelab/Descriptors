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
DEFAULT_MIN_PERSISTENCE = 3  # Minimum number of frames a contact must persist
DEFAULT_RDF_RANGE = (0.0, 30.0)  # Start from 0 for CG beads
DEFAULT_NBINS = 100  # Increased bins for better resolution with many peptides
MARTINI_CUTOFF = 11.0  # Default MARTINI cutoff
CG_BEAD_MIN_DIST = 4.7  # Minimum distance between MARTINI CG beads
CHUNK_SIZE = 100  # For memory management

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

def load_and_crop_trajectory(topology, trajectory, first, last, skip, selection="name BB SC1 SC2 SC3"):
    u = mda.Universe(topology, trajectory)

    # Define end frame if not specified
    total_frames = len(u.trajectory)
    if last is None or last > total_frames:
        last = total_frames
    if first < 0 or first >= total_frames:
        raise ValueError(f"Invalid first frame: {first}.")

    # Select the specified atoms
    peptides = u.select_atoms(selection)
    if len(peptides) == 0:
        raise ValueError(f"Selection '{selection}' returned no atoms.")

    indices = list(range(first, last, skip))

    # Add timestamp for unique temp file names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_gro = f"temp_protein_slice_{timestamp}.gro"
    temp_xtc = f"temp_protein_slice_{timestamp}.xtc"

    # Write the selected atoms to a temporary trajectory
    with mda.Writer(temp_gro, peptides.n_atoms) as W:
        W.write(peptides)
    with mda.Writer(temp_xtc, peptides.n_atoms) as W:
        for ts in u.trajectory[indices]:
            W.write(peptides)

    # Reload the cropped trajectory
    cropped_u = mda.Universe(temp_gro, temp_xtc)
    return cropped_u, temp_gro, temp_xtc  # Modify return statement

def calculate_adaptive_cutoff(universe, selection_string, rdf_range, nbins, output_dir):
    """
    Calculate adaptive cutoff distance based on RDF, optimized for MARTINI CG beads.
    """
    logging.info("Calculating adaptive cutoff distance based on RDF for CG beads.")
    box_size = universe.dimensions[:3].min()
    max_cutoff = box_size / 4  # Prevent artifacts from periodic boundaries

    # Select all CG beads instead of peptide centers
    beads = universe.select_atoms(selection_string)

    # Adjust RDF calculation for CG beads
    rdf_analysis = rdf.InterRDF(
        beads,
        beads,
        nbins=nbins,
        range=rdf_range,
        exclusion_block=(1, 1)  # Exclude self-interactions within same bead
    )
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

    # Ensure cutoff is not smaller than minimum CG bead distance
    cutoff_distance = min(cutoff_distance, max_cutoff)

    return cutoff_distance

def identify_clusters(peptides, cutoff_distance):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    positions = peptides.positions
    residue_ids = np.array([atom.residue.resid for atom in peptides])

    # Process in chunks
    chunk_size = CHUNK_SIZE
    n_chunks = len(positions) // chunk_size + 1
    adjacency = csr_matrix((len(positions), len(positions)), dtype=bool)

    for i in range(n_chunks):
        start_i = i * chunk_size
        end_i = min((i + 1) * chunk_size, len(positions))
        chunk_i = positions[start_i:end_i]

        for j in range(i, n_chunks):
            start_j = j * chunk_size
            end_j = min((j + 1) * chunk_size, len(positions))
            chunk_j = positions[start_j:end_j]

            dist = cdist(chunk_i, chunk_j)
            connections = dist < cutoff_distance
            adjacency[start_i:end_i, start_j:end_j] = connections
            if i != j:
                adjacency[start_j:end_j, start_i:end_i] = connections.T

    n_components, labels = connected_components(adjacency, directed=False)

    # Convert to residue-based clusters
    clusters = []
    for i in range(n_components):
        cluster_residues = set(residue_ids[labels == i])
        if len(cluster_residues) > 0:
            clusters.append(cluster_residues)

    return clusters

def analyze_aggregate_persistence(cluster_records, contact_persistence, min_persistence):
    """
    Apply persistence criteria to filter out transient aggregates.
    """
    logging.info("Applying persistence criteria to aggregates.")

    # Adjust persistence threshold based on CG nature
    effective_persistence = max(min_persistence, 3)  # CG systems might need lower persistence

    persistent_aggregates = []
    for cluster_id, frames in cluster_records.items():
        if len(frames) >= effective_persistence:
             # Check each contact within the cluster to ensure it meets persistence criteria
            all_contacts_persistent = True
            for a in cluster_id:
                for b in cluster_id:
                    if a < b:  # Avoid duplicates since cluster_id is unordered
                        contact = frozenset([a, b])
                        # Check if this contact's persistence count meets the threshold
                        if contact_persistence.get(contact, 0) < effective_persistence:
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
    # Initialize logging before any logging calls
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output, f'adi_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
            logging.StreamHandler()
        ]
    )
    logging.info("Starting ADI analysis.")
    print("Starting ADI analysis.")
    ensure_output_directory(args.output)

    # logging.info(f"Output directory: {args.output}")
    # print(f"Output directory: {args.output}")

    # Initialize a list to store frame records
    frame_records = []

    temp_gro, temp_xtc = None, None
    try:
        # Load and crop trajectory
        logging.info("Loading and processing trajectory...")
        u, temp_gro, temp_xtc = load_and_crop_trajectory(args.topology, args.trajectory, args.first, args.last, args.skip, args.selection)
        logging.info(f"Total frames in cropped trajectory: {len(u.trajectory)}")

        # Modify peptide selection to handle all peptides
        selection_string = args.selection
        # if selection_string == "protein":
        #     selection_string = "name BB SC1 SC2 SC3 SC4 SC5"  # Add more bead names if needed

        logging.debug(f"Atom selection string: {selection_string}")
        peptides = u.select_atoms(selection_string)
        logging.info(f"Selected {len(peptides)} atoms from {len(peptides.residues)} peptides")

        # Get actual peptide range
        residue_ids = [residue.resid for residue in peptides.residues]
        min_peptide_id = min(residue_ids)
        max_peptide_id = max(residue_ids)
        peptide_indices = [residue.resid for residue in peptides.residues]
        logging.info(f"Analyzing peptides from PEP{min_peptide_id} to PEP{max_peptide_id}")
        print(f"Analyzing peptides from PEP{min_peptide_id} to PEP{max_peptide_id}")

        # Calculate adaptive cutoff distance based on RDF
        cutoff_distance = calculate_adaptive_cutoff(
            u,
            selection_string,
            args.rdf_range,
            args.nbins,
            args.output)
        #TODO: why is it averaged for all agg. in a frame?

        # Initialize variables for analysis
        logging.info("Initializing variables for analysis.")
        cluster_records = defaultdict(list)  # {cluster_id: [frame_numbers]}
        cluster_size_distribution = []  # List of dicts with frame number and cluster sizes
        contact_persistence = defaultdict(int)  # {contact_pair: persistence_count}

        # Modify peptide ID extraction
        peptides = u.select_atoms(selection_string)
        residues = peptides.residues
        peptide_ids = [residue.resid for residue in residues]
        min_peptide_id = min(peptide_ids)
        max_peptide_id = max(peptide_ids)
        print(f"Peptide ID range: PEP{min_peptide_id} to PEP{max_peptide_id}")

        # Analyze each frame
        logging.info("Analyzing frames for aggregation.")
        frame_records = []
        for frame_number, ts in enumerate(u.trajectory):
            logging.info(f"Processing frame {frame_number}...")

            # Identify clusters in the current frame
            current_clusters = identify_clusters(peptides, cutoff_distance)
            logging.debug(f"Found {len(current_clusters)} clusters in frame {frame_number}.")
            cluster_sizes = [len(cluster) for cluster in current_clusters]
            cluster_size_distribution.append({'frame': frame_number, 'cluster_sizes': cluster_sizes})

            # Collect peptides involved in aggregates with actual IDs
            frame_peptides = []
            for cluster in current_clusters:
                peptides_in_cluster = [f'PEP{resid}' for resid in cluster]
                frame_peptides.extend(peptides_in_cluster)

                # Capture peptide indices and print min and max range
                cluster_indices = list(cluster)
                min_index = min(cluster_indices)
                max_index = max(cluster_indices)
                print(f"Aggregate peptide indices range from {min_index} to {max_index}")

            frame_peptides = sorted(set(frame_peptides))  # Remove duplicates and sort

            # Calculate required indices
            aggregate_count = len(current_clusters)
            total_peptides_in_aggregate = sum(cluster_sizes)
            avg_aggregate_size = round(total_peptides_in_aggregate / aggregate_count, 1) if aggregate_count > 0 else 0

            # Create a record for the current frame
            frame_record = {
                'Frame': frame_number,
                'Peptides': str(frame_peptides),
                'aggregate_count': aggregate_count,
                'total_peptides_in_aggregate': total_peptides_in_aggregate,
                'avg_aggregate_size': avg_aggregate_size
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
        if temp_gro:
            os.remove(temp_gro)
        if temp_xtc:
            os.remove(temp_xtc)

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
    Save frame results to a CSV file.
    """
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_file = os.path.join(output_dir, f'adi_frame_results_{timestamp}.csv')

    fieldnames = ['Frame', 'Peptides', 'aggregate_count', 'total_peptides_in_aggregate', 'avg_aggregate_size']

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in frame_records:
            writer.writerow(record)

    logging.info(f"Frame results saved to {output_file}")

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
