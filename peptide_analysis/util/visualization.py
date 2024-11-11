# util/visualization.py

"""
Visualization Utilities for Peptide Analysis

This module provides utility functions for creating plots and visualizations
used in the analysis of peptide simulations.

Functions:
- plot_cluster_size_distribution: Plot cluster size distribution over time.
- plot_persistent_aggregates: Plot number of persistent aggregates over time.
- plot_sheet_lifetimes: Plot distribution of sheet lifetimes.
- plot_vesicle_lifetimes: Plot distribution of vesicle lifetimes.
- plot_tube_lifetimes: Plot distribution of tube lifetimes.
- plot_fiber_lifetimes: Plot distribution of fiber lifetimes.
- plot_number_of_structures_per_frame: Plot number of structures (e.g., fibers) per frame.
- save_plot: Save a plot to the specified output directory.
- set_plot_style: Set a consistent style for all plots.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from datetime import datetime

def set_plot_style():
    """
    Set a consistent style for all plots.
    """
    plt.style.use('seaborn-darkgrid')
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })

def save_plot(plt, output_dir, filename):
    """
    Save the current plot to the specified output directory with a timestamp.

    Args:
        plt (matplotlib.pyplot): The pyplot module.
        output_dir (str): Path to the output directory.
        filename (str): Base filename for the plot.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"{filename}_{timestamp}.png")
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {filepath}")

def plot_cluster_size_distribution(cluster_size_distribution, output_dir):
    """
    Plot the cluster size distribution over time.

    Args:
        cluster_size_distribution (list): List of dicts with 'frame' and 'cluster_sizes'.
        output_dir (str): Path to the output directory.
    """
    set_plot_style()
    frames = [entry['frame'] for entry in cluster_size_distribution]
    max_cluster_sizes = [max(entry['cluster_sizes']) if entry['cluster_sizes'] else 0 for entry in cluster_size_distribution]
    plt.figure()
    plt.plot(frames, max_cluster_sizes, label='Max Cluster Size')
    plt.xlabel('Frame')
    plt.ylabel('Cluster Size')
    plt.title('Cluster Size Distribution Over Time')
    plt.legend()
    save_plot(plt, output_dir, 'cluster_size_distribution')

def plot_persistent_aggregates(persistent_aggregates, output_dir):
    """
    Plot the number of persistent aggregates over time.

    Args:
        persistent_aggregates (list): List of tuples with (cluster_id, frames).
        output_dir (str): Path to the output directory.
    """
    set_plot_style()
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
    save_plot(plt, output_dir, 'persistent_aggregates')

def plot_sheet_lifetimes(sheet_lifetimes, output_dir):
    """
    Plot the distribution of sheet lifetimes.

    Args:
        sheet_lifetimes (dict): Dictionary with sheet IDs and their lifetimes.
        output_dir (str): Path to the output directory.
    """
    set_plot_style()
    lifetimes = [data['lifetime'] for data in sheet_lifetimes.values()]
    if not lifetimes:
        print("No sheets met the minimum lifetime criteria; skipping plot.")
        return
    plt.figure()
    plt.hist(lifetimes, bins=range(1, max(lifetimes) + 2), align='left', edgecolor='black')
    plt.xlabel('Lifetime (frames)')
    plt.ylabel('Number of Sheets')
    plt.title('Distribution of Sheet Lifetimes')
    save_plot(plt, output_dir, 'sheet_lifetimes_distribution')

def plot_vesicle_lifetimes(vesicle_lifetimes, output_dir):
    """
    Plot the distribution of vesicle lifetimes.

    Args:
        vesicle_lifetimes (dict): Dictionary with vesicle IDs and their lifetimes.
        output_dir (str): Path to the output directory.
    """
    set_plot_style()
    lifetimes = list(vesicle_lifetimes.values())
    if not lifetimes:
        print("No vesicles detected; skipping plot.")
        return
    plt.figure()
    plt.hist(lifetimes, bins=range(1, max(lifetimes)+2), align='left', edgecolor='black')
    plt.xlabel('Lifetime (frames)')
    plt.ylabel('Number of Vesicles')
    plt.title('Distribution of Vesicle Lifetimes')
    save_plot(plt, output_dir, 'vesicle_lifetimes_distribution')

def plot_tube_lifetimes(tube_lifetimes, output_dir):
    """
    Plot the distribution of tube lifetimes.

    Args:
        tube_lifetimes (dict): Dictionary with tube IDs and their lifetimes.
        output_dir (str): Path to the output directory.
    """
    set_plot_style()
    lifetimes = list(tube_lifetimes.values())
    if not lifetimes:
        print("No tubes detected; skipping plot.")
        return
    plt.figure()
    plt.hist(lifetimes, bins=range(1, max(lifetimes)+2), align='left', edgecolor='black')
    plt.xlabel('Lifetime (frames)')
    plt.ylabel('Number of Tubes')
    plt.title('Distribution of Tube Lifetimes')
    save_plot(plt, output_dir, 'tube_lifetimes_distribution')

def plot_fiber_lifetimes(fiber_lifetimes, output_dir):
    """
    Plot the distribution of fiber lifetimes.

    Args:
        fiber_lifetimes (dict): Dictionary with fiber IDs and their lifetimes.
        output_dir (str): Path to the output directory.
    """
    set_plot_style()
    lifetimes = [data['lifetime'] for data in fiber_lifetimes.values()]
    if not lifetimes:
        print("No fibers detected; skipping plot.")
        return
    plt.figure()
    plt.hist(lifetimes, bins=range(1, max(lifetimes)+2), align='left', edgecolor='black')
    plt.xlabel('Lifetime (frames)')
    plt.ylabel('Number of Fibers')
    plt.title('Distribution of Fiber Lifetimes')
    save_plot(plt, output_dir, 'fiber_lifetimes_distribution')

def plot_number_of_structures_per_frame(frame_results, structure_type, output_dir):
    """
    Plot the number of structures (e.g., fibers, sheets) per frame.

    Args:
        frame_results (list): List of dictionaries with analysis results per frame.
        structure_type (str): Type of structure to count ('is_fiber', 'is_sheet', etc.).
        output_dir (str): Path to the output directory.
    """
    set_plot_style()
    structure_counts = defaultdict(int)
    for result in frame_results:
        if result.get(structure_type, False):
            frame = result.get('frame')
            structure_counts[frame] += 1

    frames = sorted(structure_counts.keys())
    counts = [structure_counts[frame] for frame in frames]

    plt.figure()
    plt.plot(frames, counts, marker='o', linestyle='-')
    plt.xlabel('Frame')
    plt.ylabel(f'Number of {structure_type.replace("is_", "").capitalize()}s')
    plt.title(f'Number of {structure_type.replace("is_", "").capitalize()}s per Frame')
    plt.grid(True)
    save_plot(plt, output_dir, f'number_of_{structure_type.replace("is_", "")}s_per_frame')

def plot_radial_density_profile(density, bin_edges, output_dir, aggregate_id=''):
    """
    Plot the radial density profile for an aggregate.

    Args:
        density (numpy.ndarray): Radial density values.
        bin_edges (numpy.ndarray): Edges of the bins.
        output_dir (str): Path to the output directory.
        aggregate_id (str): Identifier for the aggregate (optional).
    """
    set_plot_style()
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.figure()
    plt.plot(bin_centers, density, label=f'Aggregate {aggregate_id}')
    plt.xlabel('Radius (Å)')
    plt.ylabel('Density')
    plt.title('Radial Density Profile')
    plt.legend()
    save_plot(plt, output_dir, f'radial_density_profile_{aggregate_id}')

def plot_orientation_distribution(angles, output_dir, aggregate_id=''):
    """
    Plot the distribution of orientation angles.

    Args:
        angles (numpy.ndarray): Array of angles in degrees.
        output_dir (str): Path to the output directory.
        aggregate_id (str): Identifier for the aggregate (optional).
    """
    set_plot_style()
    plt.figure()
    plt.hist(angles, bins=36, range=(0, 180), edgecolor='black')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Frequency')
    plt.title(f'Orientation Distribution {aggregate_id}')
    save_plot(plt, output_dir, f'orientation_distribution_{aggregate_id}')

def plot_cross_sectional_areas(cross_section_areas, output_dir, aggregate_id=''):
    """
    Plot the cross-sectional areas along the principal axis.

    Args:
        cross_section_areas (list): List of cross-sectional areas.
        output_dir (str): Path to the output directory.
        aggregate_id (str): Identifier for the aggregate (optional).
    """
    set_plot_style()
    sections = range(1, len(cross_section_areas) + 1)
    plt.figure()
    plt.plot(sections, cross_section_areas, marker='o')
    plt.xlabel('Section Number')
    plt.ylabel('Cross-Sectional Area (Å²)')
    plt.title(f'Cross-Sectional Profiling {aggregate_id}')
    plt.grid(True)
    save_plot(plt, output_dir, f'cross_sectional_profiling_{aggregate_id}')

def plot_radial_distribution_function(bins, rdf_values, output_dir):
    """
    Plot the Radial Distribution Function (RDF).

    Args:
        bins (numpy.ndarray): Bin centers.
        rdf_values (numpy.ndarray): RDF values.
        output_dir (str): Path to the output directory.
    """
    set_plot_style()
    plt.figure()
    plt.plot(bins, rdf_values)
    plt.xlabel('Distance (Å)')
    plt.ylabel('g(r)')
    plt.title('Radial Distribution Function')
    save_plot(plt, output_dir, 'rdf_plot')
