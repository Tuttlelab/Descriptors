#!/usr/bin/env python3
"""
shape_tracker.py

This script tracks the shape changes in peptide simulations by integrating the outputs from
the five descriptors:

- Aggregate Dynamics Index (ADI)
- Sheet Formation Index (SFI)
- Vesicle Formation Index (VFI)
- Tube Formation Index (TFI)
- Fiber Formation Index (FFI)

It provides a unified analysis of structural evolution over time, capturing transitions between
different structural features and providing insights into the dynamics of peptide self-assembly.

"""

import warnings
# Remove the import that causes the deprecation warning
# from Bio import BiopythonDeprecationWarning
# Modify the warnings filter to ignore the BiopythonDeprecationWarning
warnings.filterwarnings("ignore", ".*BiopythonDeprecationWarning.*")
warnings.filterwarnings("ignore", category=UserWarning)

import os
import argparse
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import glob
import datetime
import MDAnalysis as mda
import logging

# Constants
OVERLAP_THRESHOLD = 0.5  # Threshold for peptide overlap when matching structures between frames

def parse_arguments():
    parser = argparse.ArgumentParser(description='Tracking Shape Changes in Peptide Simulations')
    parser.add_argument('-t', '--topology', required=True, help='Topology file (e.g., .gro, .pdb)')
    parser.add_argument('-x', '--trajectory', required=True, help='Trajectory file (e.g., .xtc, .trr)')
    parser.add_argument('--run_descriptors', default=True, action='store_true', help='Run descriptor analyses before tracking')
    parser.add_argument('-o', '--output', required=True, help='Output directory for tracker results')
    parser.add_argument('--first', type=int, default=0, help='First frame to analyse')
    parser.add_argument('--last', type=int, default=None, help='Last frame to analyse')
    parser.add_argument('--skip', type=int, default=1, help='Process every nth frame')
    args = parser.parse_args()
    return args

def ensure_output_directory(base_dir, subdir):
    """Create output directory within the current working directory"""
    output_dir = os.path.join(base_dir, subdir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def setup_logging(output_dir):
    """Setup logging configuration"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f'shape_tracker_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def run_descriptor(descriptor_script, output_dir, topology, trajectory, first, last, skip):
    """Run descriptor analysis script with input directory as output"""
    logging.info(f"Running {descriptor_script}")
    logging.info(f"Parameters: first={first}, last={last}, skip={skip}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    descriptor_path = os.path.join(script_dir, descriptor_script)

    try:
        cmd = [
            'python3', descriptor_path,
            '--output', output_dir,
            '-t', os.path.basename(topology),
            '-x', os.path.basename(trajectory),
            '--first', str(first),
            '--last', str(last),
            '--skip', str(skip)
        ]
        logging.info(f"Executing command: {' '.join(cmd)}")

        result = subprocess.run(cmd,
                              capture_output=True,
                              text=True,
                              check=True)

        logging.info(f"{descriptor_script} completed successfully")
        if result.stdout:
            logging.debug(f"Output: {result.stdout}")

    except subprocess.CalledProcessError as e:
        logging.error(f"Error running {descriptor_script}: {e}")
        logging.error(f"Error output: {e.stderr}")
        raise

def load_descriptor_results(descriptor_results_dir, descriptor_name, start_frame, end_frame):
    """Load descriptor results and filter frames within specified range."""
    # Look for most recent file matching pattern
    pattern = f'{descriptor_name}_frame_results_*.csv'
    matching_files = sorted(glob.glob(os.path.join(descriptor_results_dir, pattern)))

    if not matching_files:
        print(f"Skipping {descriptor_name}: no matching files found in {descriptor_results_dir}")
        return None

    # Use most recent file
    frame_results_file = matching_files[-1]
    print(f"Loading {descriptor_name} results from {frame_results_file}")

    try:
        frame_results = pd.read_csv(frame_results_file)
        if 'Peptides' not in frame_results.columns:
            print(f"Warning: {descriptor_name} results missing 'Peptides' column")
            return None
        return frame_results
    except Exception as e:
        print(f"Error loading {descriptor_name} results: {e}")
        return None

def match_structures(tracks, current_structures, frame_number, structure_type):
    """
    Match structures across frames based on peptide composition overlap.
    """
    if current_structures is None or current_structures.empty:
        return
    if 'Peptides' not in current_structures.columns:
        print(f"Skipping {structure_type}: 'Peptides' column not found.")
        return
    for _, curr_structure in current_structures.iterrows():
        curr_peptides = set(eval(curr_structure['Peptides']))
        matched = False
        for track_id, track in tracks.items():
            if track['type'] != structure_type:
                continue
            last_peptides = track['peptides'][-1]
            overlap = curr_peptides & last_peptides
            if len(curr_peptides) == 0:
                overlap_ratio = 0
            else:
                overlap_ratio = len(overlap) / len(curr_peptides)
            if overlap_ratio >= OVERLAP_THRESHOLD:
                track['frames'].append(frame_number)
                track['peptides'].append(curr_peptides)
                track['properties'].append(curr_structure.to_dict())
                matched = True
                break
        if not matched:
            new_id = len(tracks) + 1
            tracks[new_id] = {
                'type': structure_type,
                'frames': [frame_number],
                'peptides': [curr_peptides],
                'properties': [curr_structure.to_dict()],
            }

def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def count_structures(counts, current_structures, frame_number, structure_type):
    """
    Count the number of each shape per frame.
    """
    if current_structures is None or current_structures.empty:
        return
    if 'Peptides' not in current_structures.columns:
        print(f"Skipping {structure_type}: 'Peptides' column not found.")
        return
    if frame_number not in counts:
        counts[frame_number] = {stype: 0 for stype in ['Aggregate', 'Sheet', 'Vesicle', 'Tube', 'Fiber']}
    counts[frame_number][structure_type] += len(current_structures)

def analyze_tracks(tracks, output_dir):
    """Analyze the tracks and create plots."""
    # Prepare data for visualization
    frame_list = sorted(set(frame for track in tracks.values() for frame in track['frames']))
    df_counts = pd.DataFrame({'Frame': frame_list})
    structure_types = ['Aggregate', 'Sheet', 'Vesicle', 'Tube', 'Fiber']

    for stype in structure_types:
        df_counts[stype] = 0

    for index, row in df_counts.iterrows():
        frame = row['Frame']
        frame_counts = {stype: 0 for stype in structure_types}
        for track in tracks.values():
            if frame in track['frames']:
                frame_counts[track['type']] += 1
        for stype in structure_types:
            df_counts.at[index, stype] = frame_counts[stype]

    # Save detailed results to CSV file
    timestamp = get_timestamp()
    results_csv_path = os.path.join(output_dir, f'structure_tracks_{timestamp}.csv')
    df_counts.to_csv(results_csv_path, index=False)
    print(f"Structure tracks data saved to {results_csv_path}")

    # Plot line chart
    plt.figure(figsize=(10, 6))
    for stype, color in zip(structure_types, ['gray', 'blue', 'green', 'red', 'purple']):
        plt.plot(df_counts['Frame'], df_counts[stype], label=stype, color=color)
    plt.xlabel('Frame')
    plt.ylabel('Number of Structures')
    plt.title('Structure Counts Over Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'structure_counts_over_time_{timestamp}.png'))
    plt.close()
    print(f"Structure counts over time plot saved to structure_counts_over_time_{timestamp}.png")

def main():
    args = parse_arguments()

    # Setup logging
    log_file = setup_logging(args.output)
    logging.info(f"Starting shape tracker analysis")
    logging.info(f"Log file: {log_file}")
    logging.info(f"Arguments: {vars(args)}")

    # Create output subdirectories in current directory
    descriptor_dirs = {
        'adi': ensure_output_directory(args.output, 'adi_results'),
        'sfi': ensure_output_directory(args.output, 'sfi_results'),
        'vfi': ensure_output_directory(args.output, 'vfi_results'),
        'tfi': ensure_output_directory(args.output, 'tfi_results'),
        'ffi': ensure_output_directory(args.output, 'ffi_results')
    }
    logging.info(f"Created output directories: {descriptor_dirs}")

    try:
        u = mda.Universe(args.topology, args.trajectory)
        logging.info(f"Loaded universe: {len(u.trajectory)} frames")

        first = args.first
        last = args.last
        skip = args.skip

        # Set last frame if not specified
        if last is None or last > len(u.trajectory):
            last = len(u.trajectory)
        if first < 0 or first >= len(u.trajectory):
            raise ValueError(f"Invalid first frame: {first}.")

        # Run descriptors with progress tracking
        descriptors = ['adi', 'sfi', 'vfi', 'tfi', 'ffi']
        for idx, desc in enumerate(descriptors, 1):
            logging.info(f"Running descriptor {idx}/{len(descriptors)}: {desc}")
            run_descriptor(f'{desc}_analysis.py',
                         descriptor_dirs[desc],
                         args.topology,
                         args.trajectory,
                         first, last, skip)

        logging.info("All descriptors completed successfully")

    except Exception as e:
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()