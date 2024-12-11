import os
import sys
import argparse
from datetime import datetime
import MDAnalysis as mda
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import logging

def ensure_output_directory(output_dir):
    print("\n=== Ensuring Output Directory ===")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Output directory exists: {output_dir}")

def find_largest_cluster(positions, eps=10.0, min_samples=5, logger=None):
    logger = logger or logging.getLogger('detail')
    logger.debug("\n=== Finding Largest Cluster ===")
    logger.debug(f"Input positions shape: {positions.shape}")
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(positions)
    labels = clustering.labels_
    logger.debug(f"Number of clusters found: {len(set(labels)) - (1 if -1 in labels else 0)}")

    if len(set(labels)) <= 1:
        logger.warning("No clusters found or all points are noise")
        return None

    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    largest_cluster_label = unique_labels[np.argmax(counts)]
    logger.debug(f"Largest cluster size: {np.sum(labels == largest_cluster_label)}")
    return labels == largest_cluster_label

def load_and_process_trajectory(topology_path, trajectory_path, output_dir, dipep_name, first=0, last=None, stride=1, logger=None):
    logger = logger or logging.getLogger('detail')
    main_logger = logging.getLogger('main')

    print("\n=== Loading and Processing Trajectory ===")

    # Check if files are already centered
    if os.path.basename(topology_path).startswith("centered_") or \
       os.path.basename(trajectory_path).startswith("centered_"):
        print("Warning: Input files appear to be already centered (filename starts with 'centered_')")
        return topology_path, trajectory_path

    print(f"Topology path exists: {os.path.exists(topology_path)}")
    print(f"Trajectory path exists: {os.path.exists(trajectory_path)}")

    # Add file size checks
    print("Loading trajectory...")
    u = mda.Universe(topology_path, trajectory_path)
    print(f"Trajectory file size: {os.path.getsize(trajectory_path) / (1024*1024):.2f} MB")

    # Load the universe with more detailed error handling
    print("Loading trajectory...")
    try:
        # First try loading just topology
        print("Attempting to load topology file...")
        u_top = mda.Universe(topology_path)
        if u_top.atoms is not None:
            print(f"Topology loaded successfully. Number of atoms: {len(u_top.atoms)}")
        else:
            print("Warning: No atoms found in the topology.")

        # Then try loading with trajectory
        print("Attempting to load trajectory file...")
        u = mda.Universe(topology_path, trajectory_path)
        print("Trajectory loaded successfully")

    except MemoryError as e:
        print(f"Memory Error: {str(e)}")
        print("Try reducing the trajectory size or using more memory")
        return None, None
    except IOError as e:
        print(f"IO Error: {str(e)}")
        print("Check if files are accessible and not corrupted")
        return None, None
    except Exception as e:
        print(f"Unexpected error loading trajectory: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return None, None

    total_frames = len(u.trajectory)
    print(f"Total frames in trajectory: {total_frames}")

    # Validate frame indices
    first = max(0, first)
    last = total_frames - 1 if last is None else min(last, total_frames - 1)
    stride = max(1, stride)

    if first > last:
        print("Error: 'first' frame is after 'last' frame.")
        return None, None

    print(f"Will process frames from {first} to {last} with stride {stride}")

    # Select atoms - update for MARTINI beads
    all_atoms = u.select_atoms("all")
    protein = u.select_atoms("name BB SC1 SC2 SC3")  # Select only peptide beads

    # Debug protein selection
    print(f"Number of peptide beads found: {len(protein)}")
    if len(protein) == 0:
        print("ERROR: No peptide beads found!")
        return None, None

    # Debug atom names
    if hasattr(u, 'atoms') and u.atoms is not None:
        print(f"Available atom names: {np.unique(u.atoms.names)}")
    else:
        print("Warning: Universe object has no atoms attribute")
    print("Note: Removing water beads from output")

    # Define timestamp for output files
    timestamp = datetime.now().strftime("%m%d_%H%M")

    # Define output trajectory file paths
    processed_gro = os.path.join(output_dir, f"centered_{dipep_name}_{timestamp}.gro")
    processed_xtc = os.path.join(output_dir, f"centered_{dipep_name}_{timestamp}.xtc")

    print(f"Writing {processed_gro} and {processed_xtc}...")

    print(f"Processing frames: first={first}, last={last}, stride={stride}")
    print(f"Number of protein atoms: {len(protein)}")

    try:
        with mda.Writer(processed_gro, n_atoms=len(protein), reindex=True) as w_gro, \
             mda.Writer(processed_xtc, n_atoms=len(protein), reindex=True) as w_xtc:

            frames = list(range(first, last + 1, stride))
            main_logger.info(f"Processing {len(frames)} frames")

            # Configure tqdm to be more minimal
            progress_bar = tqdm(
                u.trajectory[first:last + 1:stride],
                total=len(frames),
                desc="Processing frames",
                ncols=80,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}',
                ascii=True,
                file=sys.stdout
            )

            for ts in progress_bar:
                # Log frame progress only to file
                logger.debug(f"Processing frame {ts.frame}")

                protein_positions = protein.positions
                largest_cluster_mask = find_largest_cluster(protein_positions, logger=logger)

                if largest_cluster_mask is None:
                    logger.warning(f"No significant clusters found in frame {ts.frame}")
                    continue

                largest_cluster = protein[largest_cluster_mask]

                # Iterative centering with minimal console output
                for i in range(500):
                    all_atoms.wrap(compound='segments')
                    box_cog = u.dimensions[:3] / 2
                    cluster_cog = largest_cluster.center_of_geometry()
                    drift = cluster_cog - box_cog
                    drift_magnitude = np.linalg.norm(drift)

                    if i % 100 == 0:
                        logger.debug(f"Iteration {i}: drift magnitude = {drift_magnitude:.6f}")

                    all_atoms.positions -= drift
                    if drift_magnitude < 0.1:
                        logger.debug(f"Converged after {i+1} iterations")
                        break

                all_atoms.wrap(compound='atoms')
                w_xtc.write(protein)

            # Write final structure
            w_gro.write(protein)

    except Exception as e:
        print(f"Error saving processed trajectory: {e}")
        raise  # Raise the exception to see full traceback
        return None, None

    print("Processed trajectory saved successfully.")
    return processed_gro, processed_xtc

def parse_arguments():
    parser = argparse.ArgumentParser(description="Center Trajectory Using Iterative Centering")
    parser.add_argument('--dipep', type=str, help='Dipeptide folder name')
    parser.add_argument('--topology', type=str,
                       help='Path to the topology file (.gro)')
    parser.add_argument('--trajectory', type=str,
                       help='Path to the trajectory file (.xtc)')
    parser.add_argument('--output', type=str,
                       help='Output directory')
    parser.add_argument('--first', type=int, default=0, help='First frame index (default: 0)')
    parser.add_argument('--last', type=int, default=None, help='Last frame index (default: all frames)')
    parser.add_argument('--stride', type=int, default=1, help='Frame stride interval (default: 1)')
    return parser.parse_args()

def main():
    print("\n=== Starting Trajectory Processing ===")
    args = parse_arguments()
    print(f"Input parameters:")
    print(f"  Dipeptide: {args.dipep}")
    print(f"  Topology: {args.topology}")
    print(f"  Trajectory: {args.trajectory}")
    print(f"  Output dir: {args.output}")
    print(f"  Frames: first={args.first}, last={args.last}, stride={args.stride}")

    # Set default paths based on dipeptide name
    if not args.dipep:
        args.dipep = "FF"
    if not args.topology:
        args.topology = os.path.expanduser(f"~/Desktop/all_dipep_1200_60M/high_ap/{args.dipep}/eq.gro")
    if not args.trajectory:
        args.trajectory = os.path.expanduser(f"~/Desktop/all_dipep_1200_60M/high_ap/{args.dipep}/eq.xtc")
    if not args.output:
        args.output = os.path.expanduser(f"centered_files/{args.dipep}")

    ensure_output_directory(args.output)

    # Adjust last frame if default
    if args.last == -1:
        args.last = None  # Will be set in the function based on total frames

    # Debug paths
    print("\nChecking paths:")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Topology path: {args.topology}")
    print(f"Trajectory path: {args.trajectory}")

    processed_gro, processed_xtc = load_and_process_trajectory(
        args.topology, args.trajectory, args.output, args.dipep, args.first, args.last, args.stride)

    if processed_gro and processed_xtc:
        print("\n=== Trajectory Processing Completed Successfully ===")
        print(f"- Processed GRO file: {processed_gro}")
        print(f"- Processed XTC file: {processed_xtc}")
    else:
        print("\n=== Trajectory Processing Failed ===")

if __name__ == "__main__":
    main()