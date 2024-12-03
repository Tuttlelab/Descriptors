import argparse
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import MDAnalysis as mda
import time  # Add this import for timestamp


def parse_arguments():
    """
    Parse command-line arguments for cluster analysis.
    """
    parser = argparse.ArgumentParser(description='Cluster analysis with periodic boundary conditions.')
    parser.add_argument('-t', '--topology', default="centered_FF_1129_1320.gro", help='Topology file (e.g., .gro, .pdb)')
    parser.add_argument('-x', '--trajectory', default="centered_FF_1129_1320.xtc", help='Trajectory file (e.g., .xtc, .trr)')
    parser.add_argument('--first', type=int, default=4373, help='First frame to analyze')
    parser.add_argument('--last', type=int, default=4374, help='Last frame to analyze')
    parser.add_argument('--skip', type=int, default=1, help='Process every nth frame')
    parser.add_argument('--cutoff', type=float, default=4.5, help='Cutoff distance for clustering (default: 4.5)')
    parser.add_argument('--color_by', choices=['residue', 'atom'], default='atom',
                        help='Color clusters by residue type or atom type (default: residue)')
    args = parser.parse_args()
    return args


def unwrap_coordinates(positions, box_size, images):
    """
    Unwrap particle coordinates based on periodic boundary conditions.
    """
    unwrapped_positions = positions + images * box_size
    return unwrapped_positions


def cluster_analysis(positions, cutoff, box_dimensions):
    """
    Perform cluster analysis using a cutoff distance with periodic boundary conditions.
    """
    tree = cKDTree(positions, boxsize=box_dimensions)  # Enable periodic boundary handling
    num_particles = len(positions)
    visited = np.zeros(num_particles, dtype=bool)
    clusters = []

    for i in range(num_particles):
        if not visited[i]:
            cluster = []
            queue = [i]
            while queue:
                current = queue.pop()
                if not visited[current]:
                    visited[current] = True
                    cluster.append(current)
                    # Use periodic distance for neighbor search
                    neighbors = tree.query_ball_point(positions[current], cutoff)
                    queue.extend([n for n in neighbors if not visited[n]])
            clusters.append(cluster)

    # Sort clusters by size
    clusters = sorted(clusters, key=len, reverse=True)

    # Apply minimum cluster size threshold
    min_cluster_size = 20
    clusters = [cluster for cluster in clusters if len(cluster) >= min_cluster_size]

    cluster_sizes = [len(cluster) for cluster in clusters]
    return cluster_sizes, clusters


def visualize_clusters(positions, clusters, u):
    import time
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Determine if dipeptides have the same residue
    unique_residues = list(set(atom.resname for atom in u.atoms))
    if len(unique_residues) == 1:
        # Color by atom type ('BB' vs 'SC')
        atom_types = []
        for atom in u.atoms:
            if atom.name in ['N', 'CA', 'C', 'O']:  # Backbone atom names
                atom_types.append('BB')
            else:
                atom_types.append('SC')
        unique_atom_types = ['BB', 'SC']
        colors = plt.colormaps['tab20'](np.linspace(0, 1, len(unique_atom_types)))
        color_map = {atype: colors[idx] for idx, atype in enumerate(unique_atom_types)}
    else:
        # Color by residue type
        colors = plt.colormaps['tab20'](np.linspace(0, 1, len(unique_residues)))
        color_map = {resname: colors[idx] for idx, resname in enumerate(unique_residues)}

    # Plot clusters with appropriate coloring
    for idx, cluster in enumerate(clusters):
        cluster_positions = positions[cluster]
        if len(unique_residues) == 1:
            # Color by atom type
            atom_types_cluster = ['BB' if u.atoms[atom_idx].name in ['N', 'CA', 'C', 'O'] else 'SC' for atom_idx in cluster]
            cluster_colors = [color_map[atype] for atype in atom_types_cluster]
        else:
            # Color by residue type
            resnames = [u.atoms[atom_idx].resname for atom_idx in cluster]
            cluster_colors = [color_map[resname] for resname in resnames]

        ax.scatter(cluster_positions[:, 0], cluster_positions[:, 1], cluster_positions[:, 2],
                   label=f"Cluster {idx+1} (Size: {len(cluster)})", c=cluster_colors, alpha=0.8)

    ax.set_title("Cluster Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    ax.legend(loc='upper right', fontsize='small', markerscale=0.6)

    # Save the plot with high resolution as PDF
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(f"cluster_plot_{timestamp}.pdf", dpi=500, format='pdf')
    plt.close()


def save_clusters(universe, clusters, unwrapped_positions, frame_num):
    """
    Save each cluster to a separate file maintaining original box dimensions for superimposition.
    """
    import os
    output_dir = "clusters"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get original box dimensions
    original_dimensions = universe.dimensions

    for idx, cluster in enumerate(clusters):
        # Get cluster atoms and positions
        cluster_atoms = universe.atoms[cluster].copy()
        cluster_positions = unwrapped_positions[cluster]

        # Update positions while keeping original box
        cluster_atoms.positions = cluster_positions

        # Set original box dimensions
        cluster_atoms.dimensions = original_dimensions

        # Save cluster with original box
        cluster_size = len(cluster)
        cluster_filename = f"{output_dir}/frame{frame_num}_size{cluster_size}_cluster{idx+1}.gro"

        # Write cluster with original box dimensions
        with mda.Writer(cluster_filename, n_atoms=len(cluster_atoms)) as W:
            W.write(cluster_atoms)
            print(f"Cluster {idx+1} saved to {cluster_filename}")


def main():
    args = parse_arguments()

    # Load trajectory using MDAnalysis
    u = mda.Universe(args.topology, args.trajectory)

    for ts in u.trajectory[args.first:args.last:args.skip]:
        positions = u.select_atoms('all').positions
        box_dimensions = u.dimensions[:3]  # Get box dimensions for PBC

        # Perform cluster analysis with periodic boundaries
        cluster_sizes, clusters = cluster_analysis(positions, args.cutoff, box_dimensions)

        if clusters:
            print(f"Frame {ts.frame}:")
            print("Cluster sizes (sorted by size):", cluster_sizes)
            visualize_clusters(positions, clusters, u)
            save_clusters(u, clusters, positions, ts.frame)
        else:
            print(f"Frame {ts.frame}: No clusters with size >= 20 found.")


if __name__ == "__main__":
    main()
