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
import tempfile
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
    Load the topology and trajectory files using MDAnalysis, and select beads based on the selection string.

    Args:
        topology (str): Path to the topology file (e.g., .gro, .pdb).
        trajectory (str): Path to the trajectory file (e.g., .xtc, .trr).
        selection (str): Bead selection string.

    Returns:
        tuple: A tuple containing the MDAnalysis Universe and the selected BeadGroup.
    """
    u = mda.Universe(topology, trajectory)
    atoms = u.select_atoms(selection)
    if len(atoms) == 0:
        raise ValueError(f"Selection '{selection}' returned no atoms.")
    return u, atoms

def load_and_crop_trajectory(topology, trajectory, first, last, skip, selection):
    """
    Load and optionally crop the trajectory.
    Returns a Universe object.
    """
    # Load the trajectory
    u = mda.Universe(topology, trajectory)
    total_frames = len(u.trajectory)

    # Apply frame selection if 'first', 'last', or 'skip' is specified
    if first is not None or last is not None or skip is not None:
        # Determine frames to include
        start = first if first is not None else 0
        stop = last if last is not None else total_frames
        step = skip if skip is not None else 1

        # Create temporary files for the trajectory and topology
        temp_topol_file = tempfile.NamedTemporaryFile(suffix='.gro', delete=False)
        temp_traj_file = tempfile.NamedTemporaryFile(suffix='.xtc', delete=False)

        # Copy the original topology file to the temporary file
        with open(topology, 'rb') as src, open(temp_topol_file.name, 'wb') as dst:
            dst.write(src.read())

        # Save the selected frames to the temporary trajectory file
        if u.atoms is None:
            raise ValueError("No atoms found in the trajectory.")
        with mda.Writer(temp_traj_file.name, n_atoms=u.atoms.n_atoms) as W:
            for ts in u.trajectory[start:stop:step]:
                W.write(u.atoms)

        # Create a new Universe with the cropped trajectory
        u_cropped = mda.Universe(temp_topol_file.name, temp_traj_file.name)

        # Store temp file names to delete them later
        temp_files = [temp_topol_file.name, temp_traj_file.name]

        return u_cropped, temp_files
    else:
        return u, []

def center_and_wrap_trajectory(universe, selection_string):
    """
    Center the selected group in the simulation box and wrap all beads to handle PBC issues.

    Args:
        universe (MDAnalysis.Universe): The MDAnalysis Universe object.
        selection_string (str): Bead selection string for centering.

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
    parser.add_argument('-s', '--selection', default='protein', help='Bead selection string')
    parser.add_argument('-o', '--output', default='results', help='Output directory for results') # TODO: Review this line later for the output directory
    parser.add_argument('--first', type=int, default=0, help='First frame to analyze (default is 0)')
    parser.add_argument('--last', type=int, default=None, help='Last frame to analyze (default is all frames)')
    parser.add_argument('--skip', type=int, default=1, help='Process every nth frame (default is every frame)') # TODO: Change all skip to step / stride
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
