"""
descriptors/vfi.py

Vesicle Formation Index (VFI) Analysis for peptide self-assembly simulations.
Assesses hollow spherical vesicles using radial density profiling and sphericity descriptors.
"""

import os
import csv
import argparse
import logging
from datetime import datetime
import numpy as np
import MDAnalysis as mda
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from descriptors.utils import suppress_warnings, ensure_output_directory, setup_logger

suppress_warnings()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Vesicle Formation Index (VFI) Analysis')
    parser.add_argument('-t', '--topology', required=True, help='Topology file (e.g., .gro, .pdb)')
    parser.add_argument('-x', '--trajectory', required=True, help='Trajectory file (e.g., .xtc, .trr)')
    parser.add_argument('-o', '--output', default='vfi_results', help='Output directory for results')
    parser.add_argument('--first', type=int, default=0, help='First frame index')
    parser.add_argument('--last', type=int, default=None, help='Last frame index')
    parser.add_argument('--skip', type=int, default=1, help='Process every nth frame')
    return parser.parse_args()

def calculate_sphericity(positions):
    """Calculate sphericity via convex hull volume and surface area."""
    if len(positions) < 4:
        return 0.0
    try:
        hull = ConvexHull(positions)
        volume = hull.volume
        area = hull.area
        if area == 0:
            return 0.0
        sphericity = (np.pi ** (1/3) * (6 * volume) ** (2/3)) / area
        return min(max(sphericity, 0.0), 1.0)
    except Exception:
        return 0.0

def calculate_vfi(topology, trajectory, output_dir='vfi_results', first=0, last=None, skip=1):
    """Programmatic API for VFI analysis."""
    ensure_output_directory(output_dir)
    logger = setup_logger("vfi", output_dir)
    logger.info(f"Starting VFI analysis on {trajectory}")

    u = mda.Universe(topology, trajectory)
    peptides = u.select_atoms('all')

    frame_data = []
    frames = range(first, last or len(u.trajectory), skip)

    for frame_number in frames:
        u.trajectory[frame_number]
        positions = peptides.positions
        sphericity = calculate_sphericity(positions)
        
        # Hollow core check via distance from COM
        com = peptides.center_of_mass()
        dists = np.linalg.norm(positions - com, axis=1)
        inner_core_mask = dists < (np.mean(dists) * 0.4)
        is_hollow = np.sum(inner_core_mask) < (0.05 * len(positions))
        vesicle_count = 1 if (sphericity > 0.75 and is_hollow) else 0

        frame_data.append({
            'frame': frame_number,
            'peptides': len(peptides),
            'vesicle_count': vesicle_count,
            'sphericity': sphericity,
            'total_peptides_in_vesicles': len(peptides) if vesicle_count else 0
        })

    save_vfi_results(frame_data, output_dir)
    logger.info("VFI analysis completed successfully.")
    return frame_data

def save_vfi_results(frame_data, output_dir):
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_file = os.path.join(output_dir, f'vfi_frame_results_{timestamp}.csv')
    headers = ['Frame', 'Peptides', 'vesicle_count', 'sphericity', 'total_peptides_in_vesicles']
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for data in frame_data:
            writer.writerow(data)

def main():
    args = parse_arguments()
    calculate_vfi(args.topology, args.trajectory, args.output, args.first, args.last, args.skip)

if __name__ == '__main__':
    main()
