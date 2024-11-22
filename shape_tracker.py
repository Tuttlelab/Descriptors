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
import os
import argparse
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import glob

# Constants
OVERLAP_THRESHOLD = 0.5  # Threshold for peptide overlap when matching structures between frames

import warnings
from Bio import Application
from Bio import BiopythonDeprecationWarning

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Tracking Shape Changes in Peptide Simulations')
    parser.add_argument('--run_descriptors', action='store_true', help='Run descriptor analyses before tracking')
    parser.add_argument('-o', '--output', default='tracker_results', help='Output directory for tracker results')
    parser.add_argument('--first', type=int, default=None, help='Only analyze the first N frames (default is all frames)')
    parser.add_argument('--last', type=int, default=None, help='Only analyze the last N frames (default is all frames)')
    parser.add_argument('--skip', type=int, default=1, help='Process every nth frame (default is every frame)')
    args = parser.parse_args()
    return args

def ensure_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def run_descriptor(descriptor_script, output_dir, first, last, skip):
    """
    Run a descriptor analysis script.
    """
    try:
        subprocess.run(['python3', descriptor_script,
                        '--output', output_dir,
                        '-t', "eq_FF1200.gro",
                        '-x', "eq_FF1200.xtc",
                        '--first', str(first) if first is not None else 'None',
                        '--last', str(last) if last is not None else 'None',
                        '--skip', str(skip)],
                       check=True)
        print(f"{descriptor_script} executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running {descriptor_script}: {e}")
        exit(1)

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
    results_csv_path = os.path.join(output_dir, 'structure_tracks.csv')
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
    plt.savefig(os.path.join(output_dir, 'structure_counts_over_time.png'))
    plt.close()
    print("Structure counts over time plot saved.")

def main():
    args = parse_arguments()
    ensure_output_directory(args.output)

    descriptor_dirs = {
        'adi': 'adi_results',
        'sfi': 'sfi_results',
        'vfi': 'vfi_results',
        'tfi': 'tfi_results',
        'ffi': 'ffi_results'
    }

    if args.run_descriptors:
        print("Running descriptor analyses...")
        run_descriptor('adi_analysis.py', descriptor_dirs['adi'], args.first, args.last, args.skip)
        run_descriptor('sfi_analysis.py', descriptor_dirs['sfi'], args.first, args.last, args.skip)
        run_descriptor('vfi_analysis.py', descriptor_dirs['vfi'], args.first, args.last, args.skip)
        run_descriptor('tfi_analysis.py', descriptor_dirs['tfi'], args.first, args.last, args.skip)
        run_descriptor('ffi_analysis.py', descriptor_dirs['ffi'], args.first, args.last, args.skip)

    # Load descriptor results without frame filtering
    print("Loading descriptor results...")
    adi_results = load_descriptor_results(descriptor_dirs['adi'], 'adi', args.first, args.last)
    sfi_results = load_descriptor_results(descriptor_dirs['sfi'], 'sfi', args.first, args.last)
    vfi_results = load_descriptor_results(descriptor_dirs['vfi'], 'vfi', args.first, args.last)
    tfi_results = load_descriptor_results(descriptor_dirs['tfi'], 'tfi', args.first, args.last)
    ffi_results = load_descriptor_results(descriptor_dirs['ffi'], 'ffi', args.first, args.last)

    # Combine all frames from the descriptors
    all_frames = set()
    for results in [adi_results, sfi_results, vfi_results, tfi_results, ffi_results]:
        if results is not None:
            all_frames.update(results['Frame'].unique())

    # Apply frame range and skip
    start_frame = args.first if args.first is not None else min(all_frames)
    end_frame = args.last if args.last is not None else max(all_frames)
    frame_numbers = range(start_frame, end_frame + 1, args.skip)

    print(f"Analyzing frames from {start_frame} to {end_frame}, skipping every {args.skip} frames")

    # Initialize tracks dictionary
    structure_tracks = {}

    # Process the selected frames
    print("Processing frames for structure tracking...")
    for idx, frame_number in enumerate(frame_numbers):
        print(f"Processing frame {frame_number} ({idx + 1}/{len(frame_numbers)})...")

        # Get structures from each descriptor in the current frame
        adi_structures = adi_results[adi_results['Frame'] == frame_number] if adi_results is not None else None
        sfi_structures = sfi_results[sfi_results['Frame'] == frame_number] if sfi_results is not None else None
        vfi_structures = vfi_results[vfi_results['Frame'] == frame_number] if vfi_results is not None else None
        tfi_structures = tfi_results[tfi_results['Frame'] == frame_number] if tfi_results is not None else None
        ffi_structures = ffi_results[ffi_results['Frame'] == frame_number] if ffi_results is not None else None

        # Match and track structures
        match_structures(structure_tracks, adi_structures, frame_number, 'Aggregate')
        match_structures(structure_tracks, sfi_structures, frame_number, 'Sheet')
        match_structures(structure_tracks, vfi_structures, frame_number, 'Vesicle')
        match_structures(structure_tracks, tfi_structures, frame_number, 'Tube')
        match_structures(structure_tracks, ffi_structures, frame_number, 'Fiber')

    # Analyze tracks
    analyze_tracks(structure_tracks, args.output)

if __name__ == '__main__':
    main()