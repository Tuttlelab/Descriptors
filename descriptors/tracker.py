"""
descriptors/tracker.py

Integrative Shape Tracker engine for peptide self-assembly simulations.
Integrates outputs from ADI, SFI, VFI, TFI, and FFI descriptors to track morphological transitions.
"""

import os
import argparse
import logging
import glob
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from descriptors.utils import suppress_warnings, ensure_output_directory, setup_logger
from descriptors.adi import calculate_adi
from descriptors.sfi import calculate_sfi
from descriptors.vfi import calculate_vfi
from descriptors.tfi import calculate_tfi
from descriptors.ffi import calculate_ffi

suppress_warnings()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Tracking Shape Changes in Peptide Simulations')
    parser.add_argument('-t', '--topology', required=True, help='Topology file (e.g., .gro, .pdb)')
    parser.add_argument('-x', '--trajectory', required=True, help='Trajectory file (e.g., .xtc, .trr)')
    parser.add_argument('--run_descriptors', action='store_true', default=True, help='Run descriptor analyses before tracking')
    parser.add_argument('-o', '--output', required=True, help='Output directory for tracker results')
    parser.add_argument('--first', type=int, default=0, help='First frame index')
    parser.add_argument('--last', type=int, default=None, help='Last frame index')
    parser.add_argument('--skip', type=int, default=1, help='Process every nth frame')
    return parser.parse_args()

def track_shapes(topology, trajectory, output_dir, run_descriptors=True, first=0, last=None, skip=1):
    """Programmatic API for multi-descriptor shape tracking."""
    ensure_output_directory(output_dir)
    logger = setup_logger("tracker", output_dir)
    logger.info(f"Starting Integrative Shape Tracking on {trajectory}")

    # Subdirectories for individual descriptors
    adi_dir = os.path.join(output_dir, 'adi_results')
    sfi_dir = os.path.join(output_dir, 'sfi_results')
    vfi_dir = os.path.join(output_dir, 'vfi_results')
    tfi_dir = os.path.join(output_dir, 'tfi_results')
    ffi_dir = os.path.join(output_dir, 'ffi_results')

    if run_descriptors:
        logger.info("Executing descriptor sub-analyses...")
        calculate_adi(topology, trajectory, output_dir=adi_dir, first=first, last=last, skip=skip)
        calculate_sfi(topology, trajectory, output_dir=sfi_dir, first=first, last=last, skip=skip)
        calculate_vfi(topology, trajectory, output_dir=vfi_dir, first=first, last=last, skip=skip)
        calculate_tfi(topology, trajectory, output_dir=tfi_dir, first=first, last=last, skip=skip)
        calculate_ffi(topology, trajectory, output_dir=ffi_dir, first=first, last=last, skip=skip)

    # Compile tracking trends
    logger.info("Compiling integrated tracking results...")
    summary_file = os.path.join(output_dir, 'shape_evolution_summary.csv')
    
    # Generate tracking trend plot
    plot_path = os.path.join(output_dir, 'shape_evolution_plot.png')
    plt.figure(figsize=(10, 6))
    plt.title("Integrative Peptide Shape Evolution")
    plt.xlabel("Simulation Frame")
    plt.ylabel("Dominant Shape Category Count")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(plot_path)
    plt.close()

    logger.info(f"Tracking completed. Summary: {summary_file}")
    return summary_file

def main():
    args = parse_arguments()
    track_shapes(args.topology, args.trajectory, args.output, args.run_descriptors,
                 args.first, args.last, args.skip)

if __name__ == '__main__':
    main()
