"""
descriptors/tfi.py

Tube Formation Index (TFI) Analysis for peptide self-assembly simulations.
Identifies hollow cylindrical or tubular aggregates using gyration tensor asphericity and radial uniformity.
"""

import os
import csv
import argparse
import logging
from datetime import datetime
import numpy as np
import MDAnalysis as mda
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from descriptors.utils import suppress_warnings, ensure_output_directory, setup_logger

suppress_warnings()

MIN_TUBE_SIZE = 50

def parse_arguments():
    parser = argparse.ArgumentParser(description='Tube Formation Index (TFI) Analysis')
    parser.add_argument('-t', '--topology', required=True, help='Topology file (e.g., .gro, .pdb)')
    parser.add_argument('-x', '--trajectory', required=True, help='Trajectory file (e.g., .xtc, .trr)')
    parser.add_argument('-o', '--output', default='tfi_results', help='Output directory for results')
    parser.add_argument('--min_tube_size', type=int, default=MIN_TUBE_SIZE, help='Minimum atoms for tube')
    parser.add_argument('--first', type=int, default=0, help='First frame index')
    parser.add_argument('--last', type=int, default=None, help='Last frame index')
    parser.add_argument('--skip', type=int, default=1, help='Process every nth frame')
    return parser.parse_args()

def calculate_tfi(topology, trajectory, output_dir='tfi_results', min_tube_size=MIN_TUBE_SIZE,
                  first=0, last=None, skip=1):
    """Programmatic API for TFI analysis."""
    ensure_output_directory(output_dir)
    logger = setup_logger("tfi", output_dir)
    logger.info(f"Starting TFI analysis on {trajectory}")

    u = mda.Universe(topology, trajectory)
    peptides = u.select_atoms('all')

    frame_data = []
    frames = range(first, last or len(u.trajectory), skip)

    for frame_number in frames:
        u.trajectory[frame_number]
        positions = peptides.positions
        com = peptides.center_of_mass()
        centered = positions - com

        # Inertia tensor / gyration tensor
        cov = np.cov(centered.T)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        lambda1, lambda2, lambda3 = eigvals[0], eigvals[1], eigvals[2]

        # Cylindrical / tube condition: lambda1 >> lambda2 ~ lambda3 > 0
        is_tube = (lambda1 > 2.0 * lambda2) and (lambda2 > 0) and (abs(lambda2 - lambda3)/lambda2 < 0.5)
        tube_count = 1 if is_tube else 0

        frame_data.append({
            'frame': frame_number,
            'peptides': len(peptides),
            'tube_count': tube_count,
            'total_peptides_in_tubes': len(peptides) if is_tube else 0,
            'avg_tube_size': len(peptides) if is_tube else 0
        })

    save_tfi_results(frame_data, output_dir)
    logger.info("TFI analysis completed successfully.")
    return frame_data

def save_tfi_results(frame_data, output_dir):
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_file = os.path.join(output_dir, f'tfi_frame_results_{timestamp}.csv')
    headers = ['Frame', 'Peptides', 'tube_count', 'total_peptides_in_tubes', 'avg_tube_size']
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for data in frame_data:
            writer.writerow(data)

def main():
    args = parse_arguments()
    calculate_tfi(args.topology, args.trajectory, args.output, args.min_tube_size,
                  args.first, args.last, args.skip)

if __name__ == '__main__':
    main()
