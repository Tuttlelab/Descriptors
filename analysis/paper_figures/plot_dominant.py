#!/usr/bin/env python3
"""
plot_dominant.py

Unified script for analyzing and plotting dominant shape distributions across
different peptide systems (e.g. FF, RF, WI).
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def load_all_descriptors(data_dir):
    """Load and merge all shape descriptor CSV files for a given system directory"""
    print(f"Loading descriptor CSV files from '{data_dir}'...")
    try:
        adi = pd.read_csv(os.path.join(data_dir, 'adi_output.csv'))
        sfi = pd.read_csv(os.path.join(data_dir, 'sfi_output.csv'))
        ffi = pd.read_csv(os.path.join(data_dir, 'ffi_output.csv'))
        tfi = pd.read_csv(os.path.join(data_dir, 'tfi_output.csv'))
        vfi = pd.read_csv(os.path.join(data_dir, 'vfi_output.csv'))
    except FileNotFoundError as e:
        print(f"Error loading descriptor CSV files: {e}")
        print(f"Expected in '{data_dir}': adi_output.csv, sfi_output.csv, ffi_output.csv, tfi_output.csv, vfi_output.csv")
        return None

    merged = pd.merge(sfi, ffi, on='Frame')
    merged = pd.merge(merged, tfi, on='Frame')
    merged = pd.merge(merged, vfi, on='Frame')

    print(f"Successfully loaded {len(merged)} merged frames from '{data_dir}'.")
    return merged


def analyze_state_transitions(df):
    """Analyze dominant shape transitions with thresholding"""
    sheet_mask = df['avg_sheet_size'] >= 5
    fiber_mask = df['avg_fiber_size'] >= 5
    vesicle_mask = df['avg_vesicle_size'] >= 5
    tube_mask = df['avg_tube_size'] >= 5

    df['sheet_score'] = df.apply(
        lambda row: row['total_peptides_in_sheets'] * row['avg_sheet_size'] if sheet_mask[row.name] else 0, axis=1
    )
    df['fiber_score'] = df.apply(
        lambda row: row['total_peptides_in_fibers'] * row['avg_fiber_size'] if fiber_mask[row.name] else 0, axis=1
    )
    df['vesicle_score'] = df.apply(
        lambda row: row['total_peptides_in_vesicles'] * row['avg_vesicle_size'] if vesicle_mask[row.name] else 0, axis=1
    )
    df['tube_score'] = df.apply(
        lambda row: row['total_peptides_in_tubes'] * row['avg_tube_size'] if tube_mask[row.name] else 0, axis=1
    )

    scores = ['sheet_score', 'fiber_score', 'vesicle_score', 'tube_score']
    states = ['sheet', 'fiber', 'vesicle', 'tube']

    df['has_structures'] = df[scores].max(axis=1) > 0
    df['shapes'] = 'undetermined'
    if df['has_structures'].any():
        df.loc[df['has_structures'], 'shapes'] = pd.DataFrame(
            [df[score] for score in scores], index=states
        ).idxmax(axis=0)[df['has_structures']]

    return df


def plot_aggregation_analysis(df, timestamp, data_dir='FF'):
    """Create compact dominant shape timeline plot"""
    import matplotlib as mpl
    import seaborn as sns

    os.makedirs(data_dir, exist_ok=True)
    mpl.rcParams.update({'font.size': 16})

    fig, ax = plt.subplots(figsize=(7, 1))

    colors = sns.color_palette("husl", 4)
    color_map = {
        'sheet': colors[0],
        'fiber': colors[1],
        'vesicle': colors[2],
        'tube': colors[3],
        'undetermined': 'gray'
    }

    ordered_shapes = ['undetermined', 'sheet', 'vesicle', 'fiber', 'tube']
    unique_shapes = [shape for shape in ordered_shapes if shape in df['shapes'].unique()]

    if len(unique_shapes) > 1:
        shape_map = {
            shape: 0.15 + (idx * 0.7 / (len(unique_shapes) - 1))
            for idx, shape in enumerate(unique_shapes)
        }
    else:
        shape_map = {unique_shapes[0]: 0.5}

    ax.set_ylim(-0.1, 1.1)
    time_ns = df['Frame'] / 8

    for shape in unique_shapes:
        mask = df['shapes'] == shape
        ax.scatter(
            time_ns[mask],
            [shape_map[shape]] * mask.sum(),
            c=[color_map[shape]],
            alpha=0.5,
            s=2
        )

    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.spines['left'].set_visible(True)

    max_time = max(time_ns.max(), 1500) if len(time_ns) > 0 else 1500
    ax.set_xlim(0, max_time)

    plt.tight_layout()
    output_png = os.path.join(data_dir, f'dominant_{timestamp}.png')
    plt.savefig(output_png, dpi=600, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Unified dominant shape distribution analysis for peptide self-assembly systems."
    )
    parser.add_argument(
        "-s", "--system", "-d", "--data-dir",
        dest="data_dir",
        default="FF",
        help="System directory containing descriptor CSV files (default: FF)"
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%m%d_%H%M")
    print(f"\n=== Dominant Shape Analysis for System: {args.data_dir} ===")

    df = load_all_descriptors(args.data_dir)
    if df is None:
        return

    df = analyze_state_transitions(df)

    print("Generating dominant shape plot...")
    plot_aggregation_analysis(df, timestamp, data_dir=args.data_dir)

    os.makedirs(args.data_dir, exist_ok=True)
    output_csv = os.path.join(args.data_dir, f'dominant_{timestamp}.csv')
    df[['Frame', 'shapes']].to_csv(output_csv, index=False)

    print(f"Results saved in directory: {args.data_dir}")


if __name__ == "__main__":
    main()
