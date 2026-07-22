"""
descriptors/sfi.py

Sheet Formation Index (SFI) Analysis for peptide self-assembly simulations.
Identifies planar beta-sheet assemblies and quantifies sheet counts and average sizes.
"""

import os
import csv
import argparse
import logging
from datetime import datetime
import numpy as np
import MDAnalysis as mda
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from descriptors.utils import suppress_warnings, ensure_output_directory, setup_logger

suppress_warnings()

PLANARITY_THRESHOLD = 0.9   # Å
CURVATURE_THRESHOLD = 2.0   # Å
SPATIAL_CUTOFF = 15         # nm
ANGLE_CUTOFF = 45           # degrees
MIN_SHEET_SIZE = 5

def parse_arguments():
    parser = argparse.ArgumentParser(description='Sheet Formation Index (SFI) Analysis')
    parser.add_argument('-t', '--topology', required=True, help='Topology file (e.g., .gro, .pdb)')
    parser.add_argument('-x', '--trajectory', required=True, help='Trajectory file (e.g., .xtc, .trr)')
    parser.add_argument('-o', '--output', default='sfi_results', help='Output directory for results')
    parser.add_argument('-pl', '--peptide_length', type=int, default=8, help='Length of each peptide in residues')
    parser.add_argument('--first', type=int, default=0, help='First frame index')
    parser.add_argument('--last', type=int, default=None, help='Last frame index')
    parser.add_argument('--skip', type=int, default=1, help='Process every nth frame')
    return parser.parse_args()

def calculate_sfi(topology, trajectory, output_dir='sfi_results', peptide_length=8,
                  first=0, last=None, skip=1):
    """Programmatic API for SFI analysis."""
    ensure_output_directory(output_dir)
    logger = setup_logger("sfi", output_dir)
    logger.info(f"Starting SFI analysis on {trajectory}")

    u = mda.Universe(topology, trajectory)
    peptides = u.select_atoms('all')
    n_atoms = len(peptides)
    n_peptides = n_atoms // peptide_length

    frame_data = []
    frames = range(first, last or len(u.trajectory), skip)

    for frame_number in frames:
        u.trajectory[frame_number]
        # Calculate centroids per peptide
        coords = peptides.positions.reshape((n_peptides, peptide_length, 3))
        centroids = coords.mean(axis=1)

        # Distance-based sheet clustering
        dist_matrix = squareform(pdist(centroids))
        adj = (dist_matrix < (SPATIAL_CUTOFF / 10.0)).astype(int)
        n_components, labels = connected_components(csr_matrix(adj))

        sheet_sizes = [np.sum(labels == i) for i in range(n_components) if np.sum(labels == i) >= MIN_SHEET_SIZE]
        sheet_count = len(sheet_sizes)
        total_peptides_in_sheets = sum(sheet_sizes)
        avg_sheet_size = np.mean(sheet_sizes) if sheet_count > 0 else 0

        frame_data.append({
            'frame': frame_number,
            'peptides': n_peptides,
            'sheet_count': sheet_count,
            'total_peptides_in_sheets': total_peptides_in_sheets,
            'avg_sheet_size': avg_sheet_size
        })

    save_sfi_results(frame_data, output_dir)
    logger.info("SFI analysis completed successfully.")
    return frame_data

def save_sfi_results(frame_data, output_dir):
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_file = os.path.join(output_dir, f'sfi_frame_results_{timestamp}.csv')
    headers = ['Frame', 'Peptides', 'sheet_count', 'total_peptides_in_sheets', 'avg_sheet_size']
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for data in frame_data:
            writer.writerow(data)

def main():
    args = parse_arguments()
    calculate_sfi(args.topology, args.trajectory, args.output, args.peptide_length,
                  args.first, args.last, args.skip)

if __name__ == '__main__':
    main()
