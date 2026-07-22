import os
import numpy as np
import MDAnalysis as mda
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import argparse

def unwrap_positions(positions, box):
    # Unwrap positions using minimum image convention
    unwrapped = np.copy(positions)
    for i in range(1, len(unwrapped)):
        delta = unwrapped[i] - unwrapped[i-1]
        delta -= box * np.round(delta / box)
        unwrapped[i] = unwrapped[i-1] + delta
    return unwrapped

def extract_largest_clusters(gro_file, xtc_file, output_dir, min_beads=300, n_clusters=1, skip=100):
    os.makedirs(output_dir, exist_ok=True)
    universe = mda.Universe(gro_file, xtc_file)
    sel = universe.select_atoms('all')
    print(f"Loaded trajectory with {len(sel)} atoms per frame, {len(universe.trajectory)} frames.")
    print(f"Processing every {skip}th frame, saving only the largest cluster per frame.")

    for ts in tqdm(universe.trajectory[::skip]):
        positions = sel.positions
        box = ts.dimensions[:3]
        # DBSCAN clustering
        clustering = DBSCAN(eps=12.0, min_samples=5, metric='euclidean').fit(positions)
        labels = clustering.labels_
        clusters = [np.where(labels == i)[0] for i in set(labels) if i != -1]
        # Filter clusters by size
        large_clusters = [c for c in clusters if len(c) >= min_beads]
        # Sort by size, take only the largest cluster
        if large_clusters:
            large_clusters = [sorted(large_clusters, key=len, reverse=True)[0]]  # Only the largest cluster
        else:
            large_clusters = []  # No clusters found
        for idx, cluster_indices in enumerate(large_clusters):
            cluster_atoms = sel[cluster_indices]
            # Unwrap using MDAnalysis transformation
            from MDAnalysis.transformations import unwrap
            cluster_atoms_temp = cluster_atoms.copy()
            cluster_atoms_temp.universe.trajectory[ts.frame]
            cluster_atoms_temp.universe.trajectory.add_transformations(unwrap(cluster_atoms_temp))
            cluster_pos = cluster_atoms_temp.positions
            center = np.mean(cluster_pos, axis=0)
            cluster_pos_centered = (cluster_pos - center + box/2) % box  # wrap into box
            cluster_atoms.positions = cluster_pos_centered
            # Get current positions and box
            positions = cluster_atoms.positions.copy()
            box = universe.dimensions[:3]

            # Perform a graph-based unwrapping algorithm
            # 1. Select a reference atom (e.g., atom closest to center of cluster)
            ref_idx = 0  # Default to first atom

            # Create an array for the unwrapped positions
            unwrapped = np.zeros_like(positions)
            unwrapped[ref_idx] = positions[ref_idx].copy()

            # Build a distance matrix with PBC corrections
            from scipy.spatial import distance
            dist_matrix = np.zeros((len(positions), len(positions)))
            for i in range(len(positions)):
                for j in range(len(positions)):
                    if i == j:
                        dist_matrix[i,j] = 0
                    else:
                        # Calculate minimum image distance
                        delta = positions[i] - positions[j]
                        delta -= box * np.round(delta / box)
                        dist_matrix[i,j] = np.linalg.norm(delta)

            # Use a breadth-first approach to unwrap the cluster
            visited = [ref_idx]
            unvisited = list(range(len(positions)))
            unvisited.remove(ref_idx)

            while unvisited:
                # Find the unvisited atom closest to any visited atom
                next_idx = None
                from_idx = None
                min_dist = float('inf')

                for v in visited:
                    for u in unvisited:
                        if dist_matrix[v, u] < min_dist:
                            min_dist = dist_matrix[v, u]
                            next_idx = u
                            from_idx = v

                if next_idx is None:
                    # This shouldn't happen for connected clusters
                    break

                # Unwrap this atom relative to its closest visited neighbor
                delta = positions[next_idx] - positions[from_idx]
                delta -= box * np.round(delta / box)
                unwrapped[next_idx] = unwrapped[from_idx] + delta

                # Mark as visited
                visited.append(next_idx)
                unvisited.remove(next_idx)

            # Center the unwrapped cluster in the box
            center = np.mean(unwrapped, axis=0)
            cluster_pos_centered = unwrapped - center + box/2

            # Use these positions for the cluster
            cluster_atoms.positions = cluster_pos_centered

            # Create a box that comfortably fits the cluster with padding
            min_pos = np.min(cluster_pos_centered, axis=0)
            max_pos = np.max(cluster_pos_centered, axis=0)
            extents = max_pos - min_pos
            # Add padding of 5nm on each side
            new_box = extents + 10.0  # 5nm padding on each side
            new_dims = np.concatenate([new_box, [90., 90., 90.]])
            cluster_atoms.dimensions = new_dims
            # Save only GRO file for this cluster (no XTC)
            base = f"frame{ts.frame}_size{len(cluster_atoms)}"
            gro_out = os.path.join(output_dir, f"{base}.gro")
            # Box dimensions already set above in the new_dims calculation
            # So we don't need to set cluster_atoms.dimensions here again
            with mda.Writer(gro_out, n_atoms=len(cluster_atoms)) as W:
                W.write(cluster_atoms)
            # XTC files are no longer generated
        print(f"Frame {ts.frame}: saved {len(large_clusters)} clusters.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and save the largest cluster from every 100th frame, with periodic boundary and centering.")
    parser.add_argument('--gro', type=str, default='eq_FF1200.gro', help='Input GRO file (default: eq_FF1200.gro)')
    parser.add_argument('--xtc', type=str, default='eq_FF1200.xtc', help='Input XTC file (default: eq_FF1200.xtc)')
    parser.add_argument('--output', type=str, default='centered_clusters', help='Output directory')
    parser.add_argument('--min_beads', type=int, default=300, help='Minimum beads per cluster')
    parser.add_argument('--n_clusters', type=int, default=1, help='Number of largest clusters to extract (default: 1)')
    parser.add_argument('--skip', type=int, default=100, help='Frame skip (process every nth frame, default: 100)')
    args = parser.parse_args()
    extract_largest_clusters(args.gro, args.xtc, args.output, args.min_beads, args.n_clusters, args.skip)
