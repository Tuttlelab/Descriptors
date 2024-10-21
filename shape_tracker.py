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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Constants
OVERLAP_THRESHOLD = 0.5  # Threshold for peptide overlap when matching structures between frames

def parse_arguments():
    parser = argparse.ArgumentParser(description='Tracking Shape Changes in Peptide Simulations')
    parser.add_argument('-adi', '--adi_results', help='Path to ADI results directory')
    parser.add_argument('-sfi', '--sfi_results', help='Path to SFI results directory')
    parser.add_argument('-vfi', '--vfi_results', help='Path to VFI results directory')
    parser.add_argument('-tfi', '--tfi_results', help='Path to TFI results directory')
    parser.add_argument('-ffi', '--ffi_results', help='Path to FFI results directory')
    parser.add_argument('-o', '--output', default='tracker_results', help='Output directory for tracker results')
    args = parser.parse_args()
    return args

def ensure_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def load_descriptor_results(descriptor_results_dir, descriptor_name):
    """
    Load results from a descriptor's results directory.
    """
    if descriptor_results_dir is None or not os.path.isdir(descriptor_results_dir):
        print(f"Skipping {descriptor_name}: directory not provided or doesn't exist.")
        return None
    frame_results_file = os.path.join(descriptor_results_dir, f'{descriptor_name}_frame_results.csv')
    if not os.path.isfile(frame_results_file):
        print(f"Skipping {descriptor_name}: file {frame_results_file} not found.")
        return None
    frame_results = pd.read_csv(frame_results_file)
    return frame_results

def match_structures(tracks, current_structures, frame_number, structure_type):
    """
    Match structures across frames based on peptide composition overlap.
    """
    if current_structures is None or current_structures.empty:
        return
    # For each current structure
    for _, curr_structure in current_structures.iterrows():
        curr_peptides = set(eval(curr_structure['Peptides']))
        matched = False
        # Attempt to match with existing tracks
        for track_id, track in tracks.items():
            if track['type'] != structure_type:
                continue
            last_peptides = track['peptides'][-1]
            overlap = curr_peptides & last_peptides
            overlap_ratio = len(overlap) / len(curr_peptides)
            if overlap_ratio >= OVERLAP_THRESHOLD:
                # Update existing track
                track['frames'].append(frame_number)
                track['peptides'].append(curr_peptides)
                track['properties'].append(curr_structure.to_dict())
                matched = True
                break
        if not matched:
            # Create new track
            new_id = len(tracks) + 1
            tracks[new_id] = {
                'type': structure_type,
                'frames': [frame_number],
                'peptides': [curr_peptides],
                'properties': [curr_structure.to_dict()],
            }

def analyze_tracks(tracks, output_dir):
    """
    Analyze the tracks of structures over time.
    """
    # Prepare data for visualization
    structure_counts = defaultdict(list)
    for frame_number in sorted({frame for track in tracks.values() for frame in track['frames']}):
        frame_structures = [track for track in tracks.values() if frame_number in track['frames']]
        structure_types = [track['type'] for track in frame_structures]
        for structure_type in ['Aggregate', 'Sheet', 'Vesicle', 'Tube', 'Fiber']:
            count = structure_types.count(structure_type)
            structure_counts[structure_type].append((frame_number, count))

    # Plot structure counts over time
    plt.figure()
    for structure_type, counts in structure_counts.items():
        frames, counts = zip(*sorted(counts))
        plt.plot(frames, counts, label=structure_type)
    plt.xlabel('Frame')
    plt.ylabel('Number of Structures')
    plt.title('Structure Counts Over Time')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'structure_counts_over_time.png'))
    plt.close()
    print("Structure counts over time plot saved.")

    # Save tracks to file
    tracks_file = os.path.join(output_dir, 'structure_tracks.csv')
    with open(tracks_file, 'w') as f:
        headers = ['TrackID', 'Type', 'Lifetime', 'StartFrame', 'EndFrame']
        f.write(','.join(headers) + '\n')
        for track_id, track in tracks.items():
            lifetime = len(track['frames'])
            start_frame = track['frames'][0]
            end_frame = track['frames'][-1]
            f.write(f"{track_id},{track['type']},{lifetime},{start_frame},{end_frame}\n")
    print(f"Structure tracks data saved to {tracks_file}")

def main():
    args = parse_arguments()
    ensure_output_directory(args.output)

    # Load descriptor results
    print("Loading descriptor results...")
    adi_results = load_descriptor_results(args.adi_results, 'adi')
    sfi_results = load_descriptor_results(args.sfi_results, 'sfi')
    vfi_results = load_descriptor_results(args.vfi_results, 'vfi')
    tfi_results = load_descriptor_results(args.tfi_results, 'tfi')
    ffi_results = load_descriptor_results(args.ffi_results, 'ffi')

    # Initialize tracks dictionary
    structure_tracks = {}

    # Get total number of frames
    all_frames = set()
    for results in [adi_results, sfi_results, vfi_results, tfi_results, ffi_results]:
        if results is not None:
            all_frames.update(results['Frame'])
    
    if not all_frames:
        print("No frames found. Exiting.")
        return
    
    total_frames = max(all_frames) + 1
    print(f"Total frames in simulation: {total_frames}")

    # Process each frame
    print("Processing frames for structure tracking...")
    for frame_number in range(total_frames):
        print(f"Processing frame {frame_number+1}/{total_frames}...")
        # Get structures from each descriptor in the current frame
        adi_structures = adi_results[adi_results['Frame'] == frame_number]
        sfi_structures = sfi_results[sfi_results['Frame'] == frame_number]
        vfi_structures = vfi_results[vfi_results['Frame'] == frame_number]
        tfi_structures = tfi_results[tfi_results['Frame'] == frame_number]
        ffi_structures = ffi_results[ffi_results['Frame'] == frame_number]

        # Match and track structures
        match_structures(structure_tracks, adi_structures, frame_number, 'Aggregate')
        match_structures(structure_tracks, sfi_structures, frame_number, 'Sheet')
        match_structures(structure_tracks, vfi_structures, frame_number, 'Vesicle')
        match_structures(structure_tracks, tfi_structures, frame_number, 'Tube')
        match_structures(structure_tracks, ffi_structures, frame_number, 'Fiber')

    # Analyze tracks
    analyze_tracks(structure_tracks, args.output)

    print("Tracking analysis completed successfully.")

if __name__ == '__main__':
    main()
