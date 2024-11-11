# util/data.py

"""
Data Utilities for Peptide Analysis

This module provides utility functions for data manipulation and processing,
including saving analysis results to files, calculating lifetimes, and
handling data structures.

Functions:
- save_lifetimes: Save lifetimes of structures (e.g., sheets, fibers) to a CSV file.
- save_frame_results: Save per-frame analysis results to a CSV file.
- analyze_lifetimes: Analyze the lifetimes of structures over time.
- save_cluster_size_distribution: Save cluster size distribution data to a CSV file.
- save_persistent_aggregates: Save information about persistent aggregates to a CSV file.
- read_results_from_csv: Read analysis results from a CSV file.
- filter_results: Filter analysis results based on criteria.
"""

import os
import csv
import logging
from collections import defaultdict
from datetime import datetime

def save_lifetimes(lifetimes, output_dir, filename_prefix):
    """
    Save lifetimes data of structures to a CSV file.

    Args:
        lifetimes (dict): Dictionary with structure IDs as keys and dictionaries containing
                          'start_frame', 'end_frame', and 'lifetime' as values.
        output_dir (str): Path to the output directory.
        filename_prefix (str): Prefix for the output filename (e.g., 'fiber', 'sheet').
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'{filename_prefix}_lifetimes_{timestamp}.csv')
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['StructureID', 'StartFrame', 'EndFrame', 'Lifetime'])
        for structure_id, data in lifetimes.items():
            writer.writerow([structure_id, data['start_frame'], data['end_frame'], data['lifetime']])
    print(f"{filename_prefix.capitalize()} lifetimes data saved to {output_file}")

def save_frame_results(frame_results, output_dir, filename_prefix, headers):
    """
    Save per-frame analysis results to a CSV file.

    Args:
        frame_results (list): List of dictionaries containing analysis results per frame.
        output_dir (str): Path to the output directory.
        filename_prefix (str): Prefix for the output filename (e.g., 'ffi', 'sfi').
        headers (list): List of column headers for the CSV file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'{filename_prefix}_frame_results_{timestamp}.csv')
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for result in frame_results:
            row = [result.get(header, '') for header in headers]
            writer.writerow(row)
    print(f"Per-frame results saved to {output_file}")

def analyze_lifetimes(structure_records):
    """
    Analyze the lifetimes of structures over time.

    Args:
        structure_records (dict): Dictionary where keys are structure IDs and values are lists of frame numbers.

    Returns:
        dict: Dictionary with structure IDs as keys and dictionaries containing 'start_frame',
              'end_frame', and 'lifetime' as values.
    """
    lifetimes = {}
    for structure_id, frames in structure_records.items():
        start_frame = min(frames)
        end_frame = max(frames)
        lifetime = end_frame - start_frame + 1  # Inclusive
        lifetimes[structure_id] = {
            'start_frame': start_frame,
            'end_frame': end_frame,
            'lifetime': lifetime
        }
    return lifetimes

def save_cluster_size_distribution(cluster_size_distribution, output_dir):
    """
    Save cluster size distribution data to a CSV file.

    Args:
        cluster_size_distribution (list): List of dictionaries with 'frame' and 'cluster_sizes'.
        output_dir (str): Path to the output directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'cluster_size_distribution_{timestamp}.csv')
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame', 'ClusterSizes'])
        for entry in cluster_size_distribution:
            sizes = ';'.join(map(str, entry['cluster_sizes']))
            writer.writerow([entry['frame'], sizes])
    print(f"Cluster size distribution data saved to {output_file}")

def save_persistent_aggregates(persistent_aggregates, output_dir):
    """
    Save information about persistent aggregates to a CSV file.

    Args:
        persistent_aggregates (list): List of tuples with (cluster_id, frames).
        output_dir (str): Path to the output directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'persistent_aggregates_{timestamp}.csv')
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['AggregateID', 'Frames'])
        for idx, (cluster_id, frames) in enumerate(persistent_aggregates):
            frames_str = ';'.join(map(str, frames))
            writer.writerow([idx, frames_str])
    print(f"Persistent aggregates data saved to {output_file}")

def read_results_from_csv(filepath):
    """
    Read analysis results from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        list: List of dictionaries representing each row in the CSV file.
    """
    results = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results

def filter_results(results, criteria):
    """
    Filter analysis results based on specified criteria.

    Args:
        results (list): List of dictionaries containing analysis results.
        criteria (dict): Dictionary of criteria to filter results (e.g., {'is_fiber': True}).

    Returns:
        list: Filtered list of results.
    """
    filtered_results = []
    for result in results:
        match = all(str(result.get(key, '')).lower() == str(value).lower() for key, value in criteria.items())
        if match:
            filtered_results.append(result)
    return filtered_results
