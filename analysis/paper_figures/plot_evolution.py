#!/usr/bin/env python3
"""
plot_evolution.py

Unified script for analyzing and plotting structural evolution timelines across
different peptide systems (e.g. FF, RF, WI).
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import savgol_filter
import seaborn as sns


def calculate_structure_fractions(df, window=101):
    """Calculate structure fractions with thresholds and Savitzky-Golay smoothing"""
    sheet_mask = df['avg_sheet_size'] >= 5
    fiber_mask = df['avg_fiber_size'] >= 5
    vesicle_mask = df['avg_vesicle_size'] >= 5
    tube_mask = df['avg_tube_size'] >= 5

    # Calculate scores
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
    df['has_structures'] = df[scores].max(axis=1) > 0

    total_score = df[scores].sum(axis=1)

    for score, structure in zip(scores, ['sheets', 'fibers', 'vesicles', 'tubes']):
        df[f'{structure}_fraction'] = df[score] / total_score
        df[f'{structure}_fraction'] = df[f'{structure}_fraction'].fillna(0)

        # Apply Savitzky-Golay smoothing if window length is valid
        eff_window = min(window, len(df))
        if eff_window % 2 == 0:
            eff_window -= 1
        if eff_window >= 5:
            df[f'{structure}_fraction_smooth'] = savgol_filter(
                df[f'{structure}_fraction'],
                window_length=eff_window,
                polyorder=min(3, eff_window - 1),
                mode='interp'
            )
        else:
            df[f'{structure}_fraction_smooth'] = df[f'{structure}_fraction']

    df['confident_classification'] = df['has_structures']
    return df


def identify_stable_transitions(df, min_stable_frames=20):
    """Identify genuine structure transitions, filtering out noise"""
    structures = ['sheets', 'fibers', 'vesicles', 'tubes']
    fraction_cols = [f'{structure}_fraction' for structure in structures]

    df['dominant_structure'] = 'undetermined'
    mask = df['has_structures']
    if mask.any():
        df.loc[mask, 'dominant_structure'] = pd.DataFrame(
            [df[col] for col in fraction_cols],
            index=structures
        ).idxmax(axis=0)[mask]

    structure_to_num = {s: i for i, s in enumerate(['undetermined'] + structures)}
    df['structure_code'] = df['dominant_structure'].map(structure_to_num)

    def safe_mode(x):
        if len(x) == 0:
            return 0
        vals, counts = np.unique(x, return_counts=True)
        return vals[counts.argmax()]

    df['stable_structure_code'] = (
        df['structure_code']
        .rolling(window=min_stable_frames, center=True, min_periods=1)
        .apply(safe_mode)
    )

    num_to_structure = {v: k for k, v in structure_to_num.items()}
    df['stable_structure'] = df['stable_structure_code'].map(num_to_structure)

    transitions = []
    if len(df) > 0:
        prev_state = df['stable_structure'].iloc[0]
        for idx, state in enumerate(df['stable_structure']):
            if state != prev_state and df['confident_classification'].iloc[idx]:
                transitions.append({
                    'frame': df['Frame'].iloc[idx],
                    'from_state': prev_state,
                    'to_state': state,
                    'confidence': df[f'{state}_fraction'].iloc[idx] if state != 'undetermined' else 0
                })
                prev_state = state

    return transitions


def plot_multi_scale_evolution(df, transitions, timestamp, data_dir='FF'):
    """Create evolution plot showing smoothed structure populations"""
    os.makedirs(data_dir, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.rcParams.update({'font.size': 11})

    colors = sns.color_palette("husl", 4)
    structures = ['sheets', 'fibers', 'vesicles', 'tubes']
    labels = [s.capitalize() for s in structures]

    time_ns = df['Frame'] / 8

    for i, structure in enumerate(structures):
        plt.plot(
            time_ns,
            df[f'{structure}_fraction_smooth'],
            label=labels[i],
            color=colors[i],
            linewidth=2
        )

    plt.axvspan(0, 200 / 8, color='gray', alpha=0.1, label='Transient')

    for t in transitions:
        if t['frame'] > 200:
            plt.axvline(x=t['frame'] / 8, color='gray', linestyle='--', alpha=0.3)

    ax = plt.gca()
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5, fontsize=11)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(False)
    plt.ylim(bottom=0)

    max_time = max(time_ns.max(), 1500) if len(time_ns) > 0 else 1500
    plt.xlim(0, max_time)

    plt.tight_layout()
    output_png = os.path.join(data_dir, f'evolution_{timestamp}.png')
    plt.savefig(output_png, dpi=600, bbox_inches='tight')
    plt.close()


def analyze_evolution_confidence(df, transitions):
    """Generate analysis summary with confidence metrics"""
    return {
        'total_frames': len(df),
        'confident_frames': df['confident_classification'].sum(),
        'confidence_ratio': df['confident_classification'].mean(),
        'transitions': len(transitions),
        'stable_states': df['stable_structure'].value_counts().to_dict()
    }


def load_structure_data(data_dir):
    """Load structure data from descriptor CSV outputs"""
    print(f"Loading structure data files from '{data_dir}'...")
    try:
        sheets_df = pd.read_csv(os.path.join(data_dir, 'sfi_output.csv'))
        vesicles_df = pd.read_csv(os.path.join(data_dir, 'vfi_output.csv'))
        tubes_df = pd.read_csv(os.path.join(data_dir, 'tfi_output.csv'))
        fibers_df = pd.read_csv(os.path.join(data_dir, 'ffi_output.csv'))

        df = pd.DataFrame()
        df['Frame'] = sheets_df['Frame']
        df['total_peptides_in_sheets'] = sheets_df['total_peptides_in_sheets']
        df['avg_sheet_size'] = sheets_df['avg_sheet_size']
        df['total_peptides_in_vesicles'] = vesicles_df['total_peptides_in_vesicles']
        df['avg_vesicle_size'] = vesicles_df['avg_vesicle_size']
        df['total_peptides_in_tubes'] = tubes_df['total_peptides_in_tubes']
        df['avg_tube_size'] = tubes_df['avg_tube_size']
        df['total_peptides_in_fibers'] = fibers_df['total_peptides_in_fibers']
        df['avg_fiber_size'] = fibers_df['avg_fiber_size']

        print(f"Successfully loaded {len(df)} frames from '{data_dir}'.")
        return df
    except FileNotFoundError as e:
        print(f"Error: Could not find required CSV file: {e}")
        print(f"Expected in '{data_dir}': sfi_output.csv, vfi_output.csv, tfi_output.csv, ffi_output.csv")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Unified structural evolution analysis for peptide self-assembly systems."
    )
    parser.add_argument(
        "-s", "--system", "-d", "--data-dir",
        dest="data_dir",
        default="FF",
        help="System directory containing descriptor CSV files (default: FF)"
    )
    parser.add_argument(
        "-w", "--window",
        type=int,
        default=101,
        help="Savitzky-Golay filter window size (default: 101)"
    )
    parser.add_argument(
        "--min-stable",
        type=int,
        default=20,
        help="Minimum rolling frame window to define stable states (default: 20)"
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%m%d_%H%M")
    print(f"\n=== Evolution Analysis for System: {args.data_dir} ===")

    df = load_structure_data(args.data_dir)
    if df is None:
        return

    print("Calculating structure fractions...")
    df = calculate_structure_fractions(df, window=args.window)

    print("Identifying stable transitions...")
    transitions = identify_stable_transitions(df, min_stable_frames=args.min_stable)

    print("Generating evolution plot...")
    plot_multi_scale_evolution(df, transitions, timestamp, data_dir=args.data_dir)

    summary = analyze_evolution_confidence(df, transitions)

    os.makedirs(args.data_dir, exist_ok=True)
    results = pd.DataFrame(transitions)
    output_csv = os.path.join(args.data_dir, f'evolution_{timestamp}.csv')
    results.to_csv(output_csv, index=False)

    print("\nAnalysis Summary:")
    print(f"  Total frames analyzed: {summary['total_frames']}")
    print(f"  Confident frames: {summary['confident_frames']}")
    print(f"  Transitions detected: {summary['transitions']}")
    print("\nResults saved in directory:", args.data_dir)


if __name__ == "__main__":
    main()
