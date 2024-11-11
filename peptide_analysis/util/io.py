# util/io.py

"""
Input/Output Utilities for Peptide Analysis

This module provides utility functions for loading and processing trajectories,
handling command-line arguments, and ensuring output directories exist.

Functions:
- ensure_output_directory: Check and create output directories.
- load_trajectory: Load topology and trajectory files using MDAnalysis.
- load_and_crop_trajectory: Load and crop trajectory based on frame range.
- center_and_wrap_trajectory: Center and wrap trajectory to handle periodic boundary conditions.
- parse_common_arguments: Parse common command-line arguments.
- parse_arguments: Parse command-line arguments including additional script-specific arguments.
"""

import os
import argparse
import MDAnalysis as mda
from MDAnalysis.transformations import center_in_box, unwrap

def ensure_output_directory(output_dir):
    """
    Ensure that the specified output directory exists. If it does not exist, create it.

    Args:
        output_dir (str): Path to the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

def load_trajectory(topology, trajectory, selection):
    """
    Load the topology and trajectory files using MDAnalysis, and select atoms based on the selection string.

    Args:
        topology (str): Path to the topology file (e.g., .gro, .pdb).
        trajectory (str): Path to the trajectory file (e.g., .xtc, .trr).
        selection (str): Atom selection string.

    Returns:
        tuple: A tuple containing the MDAnalysis Universe and the selected AtomGroup.
    """
    u = mda.Universe(topology, trajectory)
    atoms = u.select_atoms(selection)
    if len(atoms) == 0:
        raise ValueError(f"Selection '{selection}' returned no atoms.")
    return u, atoms

def load_and_crop_trajectory(topology, trajectory, first, last, skip, selection="protein", transformations=None):
    """
    Load and crop the trajectory based on specified frame range and selection.

    Args:
        topology (str): Path to the topology file (e.g., .gro, .pdb).
        trajectory (str): Path to the trajectory file (e.g., .xtc, .trr).
        first (int): First frame to include in the analysis.
        last (int): Last frame to include in the analysis.
        skip (int): Frame skipping interval.
        selection (str): Atom selection string.
        transformations (list): List of transformations to apply to the trajectory.

    Returns:
        MDAnalysis.Universe: The Universe containing the cropped trajectory and selected atoms.
    """
    u = mda.Universe(topology, trajectory)

    # Define total number of frames
    total_frames = len(u.trajectory)
    if last is None or last > total_frames:
        last = total_frames
    if first < 0 or first >= total_frames:
        raise ValueError(f"Invalid first frame: {first}.")

    # Ensure that 'last' is greater than 'first'
    if last <= first:
        raise ValueError(f"'last' frame must be greater than 'first' frame. Got first={first}, last={last}.")

    # Select the specified atoms
    atoms = u.select_atoms(selection)
    if len(atoms) == 0:
        raise ValueError(f"Selection '{selection}' returned no atoms.")

    indices = list(range(first, last, skip))

    # Create temporary filenames
    temp_top = "temp_topology.gro"
    temp_traj = "temp_trajectory.xtc"

    # Apply transformations if any
    if transformations:
        u.trajectory.add_transformations(*transformations)

    # Write the selected atoms to a temporary trajectory
    with mda.Writer(temp_top, atoms.n_atoms) as W:
        W.write(atoms)

    with mda.Writer(temp_traj, atoms.n_atoms) as W:
        for ts in u.trajectory[indices]:
            W.write(atoms)

    # Reload the cropped trajectory
    cropped_u = mda.Universe(temp_top, temp_traj)
    return cropped_u

def center_and_wrap_trajectory(universe, selection_string):
    """
    Center the selected group in the simulation box and wrap all atoms to handle PBC issues.

    Args:
        universe (MDAnalysis.Universe): The MDAnalysis Universe object.
        selection_string (str): Atom selection string for centering.

    Returns:
        None
    """
    selection = universe.select_atoms(selection_string)

    # Define transformations
    transformations = [
        center_in_box(selection, wrap=True),  # Center selected group and wrap the box
        unwrap(universe.atoms)  # Unwrap all atoms
    ]

    universe.trajectory.add_transformations(*transformations)

def parse_common_arguments():
    """
    Parse common command-line arguments for analysis scripts.

    Returns:
        argparse.ArgumentParser: An ArgumentParser object with common arguments added.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-t', '--topology', required=True, help='Topology file (e.g., .gro, .pdb)')
    parser.add_argument('-x', '--trajectory', required=True, help='Trajectory file (e.g., .xtc, .trr)')
    parser.add_argument('-s', '--selection', default='protein', help='Atom selection string')
    parser.add_argument('-o', '--output', default='results', help='Output directory for results')
    parser.add_argument('--first', type=int, default=0, help='First frame to analyze (default is 0)')
    parser.add_argument('--last', type=int, default=None, help='Last frame to analyze (default is all frames)')
    parser.add_argument('--skip', type=int, default=1, help='Process every nth frame (default is every frame)')
    return parser

def parse_arguments(description, additional_args=None):
    """
    Parse command-line arguments, including common arguments and any additional arguments provided.

    Args:
        description (str): Description of the script for the ArgumentParser.
        additional_args (function): Function that adds additional arguments to the parser.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=description)
    common_parser = parse_common_arguments()
    parser._actions.extend(common_parser._actions)
    parser._option_string_actions.update(common_parser._option_string_actions)

    if additional_args:
        additional_args(parser)

    args = parser.parse_args()
    return args
