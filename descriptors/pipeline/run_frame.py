import numpy as np
import os
import MDAnalysis as mda
from descriptors.core.adi import calculate_adi
from descriptors.core.sfi import calculate_sfi
from descriptors.core.vfi import calculate_vfi
from descriptors.core.ffi import calculate_ffi
from descriptors.core.tfi import calculate_tfi
import argparse
from descriptors.core.features import build_feature_row
from descriptors.core.io import save_features_csv

def process_frame(topology, trajectory, frame_idx, selection='all', output_dir='frame_results', min_beads=50, n_clusters=3):
    """
    Process a single frame: load, select, cluster, compute descriptors, save results.
    Args:
        topology: path to topology file
        trajectory: path to trajectory file
        frame_idx: int, frame number to process
        selection: atom selection string
        output_dir: where to save results
        min_beads: minimum atoms per cluster
        n_clusters: number of top clusters to process
    Returns:
        List of dicts, one per cluster, with all descriptor results
    """
    os.makedirs(output_dir, exist_ok=True)
    u = mda.Universe(topology, trajectory)
    u.trajectory[frame_idx]
    sel = u.select_atoms(selection)
    # --- ADI: cluster finding ---
    adi_result = calculate_adi(sel, u.dimensions[:3], min_persistence=1, dynamic_cutoff=True)
    clusters = adi_result['clusters']
    # Rank clusters by size, then mass, then COM radius (here: just by size)
    cluster_order = np.argsort([-len(c) for c in clusters])[:n_clusters]
    results = []
    for k in cluster_order:
        indices = clusters[k]
        if len(indices) < min_beads:
            continue
        cluster_atoms = sel[indices]
        positions = cluster_atoms.positions
        # --- Compute descriptors ---
        sfi = calculate_sfi(positions, np.zeros_like(positions))  # Placeholder: need orientations
        vfi = calculate_vfi(positions)
        ffi = calculate_ffi(positions)
        tfi = calculate_tfi(positions)
        # --- Save cluster GRO ---
        gro_name = os.path.join(output_dir, f'frame{frame_idx}_cluster{k}.gro')
        cluster_atoms.write(gro_name)
        # --- Collect results ---
        result = {
            'frame': frame_idx,
            'cluster': k,
            'size': len(indices),
            'adi': adi_result,
            'sfi': sfi,
            'vfi': vfi,
            'ffi': ffi,
            'tfi': tfi,
            'gro': gro_name,
        }
        results.append(result)
    return results

def main():
    parser = argparse.ArgumentParser(description="Process a single frame and extract descriptors.")
    parser.add_argument('--topology', required=True, help='Input GRO file')
    parser.add_argument('--trajectory', required=True, help='Input XTC file')
    parser.add_argument('--frame', type=int, required=True, help='Frame index to process')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--selection', default='all', help='Atom selection string (default: all)')
    parser.add_argument('--min_beads', type=int, default=50, help='Minimum atoms per cluster')
    parser.add_argument('--n_clusters', type=int, default=3, help='Number of top clusters to process')
    args = parser.parse_args()

    print(f"Processing frame {args.frame} from {args.topology} and {args.trajectory}")
    try:
        results = process_frame(
            args.topology,
            args.trajectory,
            args.frame,
            selection=args.selection,
            output_dir=args.output,
            min_beads=args.min_beads,
            n_clusters=args.n_clusters
        )
        if not results:
            print("No clusters found or all below min_beads.")
        else:
            # Build feature rows and save as CSV
            feature_rows = [build_feature_row(r['frame'], r['cluster'], r['adi'], r['sfi'], r['vfi'], r['ffi'], r['tfi']) for r in results]
            csv_path = os.path.join(args.output, f"frame{args.frame}_features.csv")
            save_features_csv(feature_rows, csv_path)
            print(f"Saved features to {csv_path}")
    except Exception as e:
        print(f"Error processing frame {args.frame}: {e}")

if __name__ == "__main__":
    main()