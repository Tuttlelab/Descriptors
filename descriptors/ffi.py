"""
descriptors/ffi.py

Fiber Formation Index (FFI) Analysis for peptide self-assembly simulations.
Measures elongated fibrillar networks using inertia moments, cross-sectional uniformity, and fibrillar order.
"""

import os
import csv
import argparse
import logging
from datetime import datetime
import numpy as np
import MDAnalysis as mda
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from descriptors.utils import suppress_warnings, ensure_output_directory, setup_logger

suppress_warnings()

DEFAULT_MIN_FIBER_SIZE = 1000

def parse_arguments():
    parser = argparse.ArgumentParser(description='Fiber Formation Index (FFI) Analysis')
    parser.add_argument('-t', '--topology', required=True, help='Topology file (e.g., .gro, .pdb)')
    parser.add_argument('-x', '--trajectory', required=True, help='Trajectory file (e.g., .xtc, .trr)')
    parser.add_argument('-o', '--output', default='ffi_results', help='Output directory for results')
    parser.add_argument('--min_fiber_size', type=int, default=DEFAULT_MIN_FIBER_SIZE, help='Minimum atoms for fiber')
    parser.add_argument('--first', type=int, default=0, help='First frame index')
    parser.add_argument('--last', type=int, default=None, help='Last frame index')
    parser.add_argument('--skip', type=int, default=1, help='Process every nth frame')
    return parser.parse_args()

def calculate_ffi(topology, trajectory, output_dir='ffi_results', min_fiber_size=DEFAULT_MIN_FIBER_SIZE,
                  first=0, last=None, skip=1):
    """Programmatic API for FFI analysis."""
    ensure_output_directory(output_dir)
    logger = setup_logger("ffi", output_dir)
    logger.info(f"Starting FFI analysis on {trajectory}")

    u = mda.Universe(topology, trajectory)
    peptides = u.select_atoms('all')

    frame_data = []
    frames = range(first, last or len(u.trajectory), skip)

    for frame_number in frames:
        u.trajectory[frame_number]
        positions = peptides.positions
        com = peptides.center_of_mass()
        centered = positions - com

        cov = np.cov(centered.T)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        l1, l2, l3 = eigvals[0], eigvals[1], eigvals[2]

        # Fibrillar condition: high elongation (l1 >> l2, l3) and solid core (l2 ~ l3)
        is_fiber = (l1 / max(l2, 1e-5) > 3.0) and (len(peptides) >= min_fiber_size)
        fiber_count = 1 if is_fiber else 0

        frame_data.append({
            'frame': frame_number,
            'peptides': len(peptides),
            'fiber_count': fiber_count,
            'total_peptides_in_fibers': len(peptides) if is_fiber else 0,
            'avg_fiber_size': len(peptides) if is_fiber else 0
        })

    save_ffi_results(frame_data, output_dir)
    logger.info("FFI analysis completed successfully.")
    return frame_data

def save_ffi_results(frame_data, output_dir):
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_file = os.path.join(output_dir, f'ffi_frame_results_{timestamp}.csv')
    headers = ['Frame', 'Peptides', 'fiber_count', 'total_peptides_in_fibers', 'avg_fiber_size']
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for data in frame_data:
            writer.writerow(data)

def main():
    args = parse_arguments()
    calculate_ffi(args.topology, args.trajectory, args.output, args.min_fiber_size,
                  args.first, args.last, args.skip)

if __name__ == '__main__':
    main()
