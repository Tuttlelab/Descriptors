"""
descriptors/centering.py

PBC Centering and Cluster Extraction Utilities for MD trajectories.
Handles periodic boundary conditions, cluster centering via DBSCAN, and GRO/XTC export.
"""

import os
import argparse
import numpy as np
import MDAnalysis as mda
from MDAnalysis.transformations import center_in_box, wrap
from sklearn.cluster import DBSCAN

from descriptors.utils import suppress_warnings, ensure_output_directory

suppress_warnings()

def find_largest_cluster(positions, eps=10.0, min_samples=5):
    """Identify the largest spatial cluster using DBSCAN."""
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(positions)
    labels = clustering.labels_

    if len(set(labels)) <= 1:
        return None

    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique_labels) == 0:
        return None
    largest_label = unique_labels[np.argmax(counts)]
    return labels == largest_label

def center_trajectory(topology_path, trajectory_path, output_dir='centered_files',
                      dipep_name='pep', first=0, last=None, skip=1):
    """Center protein aggregates in simulation box and wrap PBC."""
    ensure_output_directory(output_dir)
    u = mda.Universe(topology_path, trajectory_path)
    protein = u.select_atoms("protein")
    if len(protein) == 0:
        protein = u.select_atoms("all")

    output_gro = os.path.join(output_dir, f"centered_{dipep_name}.gro")
    output_xtc = os.path.join(output_dir, f"centered_{dipep_name}.xtc")

    with mda.Writer(output_gro, protein.n_atoms) as W:
        W.write(protein)

    total_frames = len(u.trajectory)
    last = total_frames if last is None else min(last, total_frames)
    frame_indices = range(first, last, skip)

    transformations = [center_in_box(protein, wrap=True)]
    u.trajectory.add_transformations(*transformations)

    with mda.Writer(output_xtc, protein.n_atoms) as W:
        for ts in u.trajectory[frame_indices]:
            W.write(protein)

    return output_gro, output_xtc

def main():
    parser = argparse.ArgumentParser(description='PBC Centering and Trajectory Processing')
    parser.add_argument('-t', '--topology', required=True, help='Topology file (.gro, .pdb)')
    parser.add_argument('-x', '--trajectory', required=True, help='Trajectory file (.xtc, .trr)')
    parser.add_argument('-o', '--output', default='centered_files', help='Output directory')
    parser.add_argument('-n', '--name', default='pep', help='Dipeptide / system name')
    parser.add_argument('--first', type=int, default=0, help='First frame index')
    parser.add_argument('--last', type=int, default=None, help='Last frame index')
    parser.add_argument('--skip', type=int, default=1, help='Frame step size')
    args = parser.parse_args()

    center_trajectory(args.topology, args.trajectory, args.output, args.name,
                      args.first, args.last, args.skip)

if __name__ == '__main__':
    main()
