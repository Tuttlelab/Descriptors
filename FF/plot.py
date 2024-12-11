import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import savgol_filter
from scipy.stats import mode
import seaborn as sns
import os

def calculate_structure_fractions(df, window_size=10, confidence_threshold=0.15):
    """Calculate structure fractions with confidence filtering"""
    total_peptides = (df['total_peptides_in_sheets'] +
                     df['total_peptides_in_fibers'] +
                     df['total_peptides_in_vesicles'] +
                     df['total_peptides_in_tubes'])

    structures = ['sheets', 'fibers', 'vesicles', 'tubes']

    # Initialize all fractions
    for structure in structures:
        df[f'{structure}_fraction'] = df[f'total_peptides_in_{structure}'] / total_peptides.where(total_peptides > 0, 0)

        # Set fraction to 0 where total_peptides is 0
        df[f'{structure}_fraction'] = df[f'{structure}_fraction'].fillna(0)

        # Apply Savitzky-Golay filter for smooth trend
        df[f'{structure}_smooth'] = savgol_filter(
            df[f'{structure}_fraction'],
            window_length=window_size*2+1,
            polyorder=3
        )

    # Mark non-aggregate states
    df['is_non_aggregate'] = total_peptides == 0

    # Mark low confidence predictions
    df['confident_classification'] = (
        (df[[f'{s}_smooth' for s in structures]].max(axis=1) > confidence_threshold) |
        df['is_non_aggregate']  # Non-aggregate state is always considered confident
    )

    return df

def identify_stable_transitions(df, min_stable_frames=20):
    """Identify genuine structure transitions using manual window"""
    structures = ['sheets', 'fibers', 'vesicles', 'tubes']
    smooth_cols = [f'{structure}_smooth' for structure in structures]

    # Get dominant structure at each frame
    df['dominant_structure'] = pd.DataFrame(
        [df[col] for col in smooth_cols],
        index=structures
    ).idxmax()

    # Mark low confidence as non_aggregate
    mask = df[smooth_cols].max(axis=1) < 0.15
    df.loc[mask, 'dominant_structure'] = 'non_aggregate'

    # Manual sliding window for stability check
    n = len(df)
    stable_structures = []

    for i in range(n):
        start = max(0, i - min_stable_frames//2)
        end = min(n, i + min_stable_frames//2)
        window = df['dominant_structure'].iloc[start:end]
        counts = window.value_counts()
        stable_structures.append(counts.index[0] if len(counts) > 0 else 'non_aggregate')

    df['stable_structure'] = stable_structures

    # Find transitions
    transitions = []
    prev_state = df['stable_structure'].iloc[0]

    for idx, state in enumerate(df['stable_structure']):
        if state != prev_state:
            conf = (df[f'{state}_smooth'].iloc[idx]
                   if state != 'non_aggregate' and state in structures
                   else 0.0)

            transitions.append({
                'frame': df['Frame'].iloc[idx],
                'from_state': prev_state,
                'to_state': state,
                'confidence': conf
            })
            prev_state = state

    return transitions

def plot_multi_scale_evolution(df, transitions, timestamp):
    """Create dual-view evolution plot showing both micro and macro trends"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[2, 1])

    colors = sns.color_palette("husl", 4)
    structures = ['sheets', 'fibers', 'vesicles', 'tubes']
    labels = [s.capitalize() for s in structures]

    # Top plot: Detailed view with all structures
    for i, structure in enumerate(structures):
        ax1.plot(df['Frame'], df[f'{structure}_smooth'],
                label=labels[i], color=colors[i], linewidth=2)

        # Add confidence bands
        ax1.fill_between(
            df['Frame'],
            df[f'{structure}_smooth'] - df[f'{structure}_fraction'].std(),
            df[f'{structure}_smooth'] + df[f'{structure}_fraction'].std(),
            color=colors[i], alpha=0.1
        )

    # Mark low confidence regions
    low_conf_regions = ~df['confident_classification']
    if low_conf_regions.any():
        ax1.fill_between(df['Frame'], 0, 1,
                        where=low_conf_regions,
                        color='gray', alpha=0.1, label='Low Confidence')

    # Bottom plot: Simplified state transitions
    ax2.scatter(df['Frame'], df['stable_structure'],
               c=df['confident_classification'].map({True: 'blue', False: 'gray'}),
               alpha=0.5, s=5)

    # Add transition markers
    for t in transitions:
        ax1.axvline(x=t['frame'], color='gray', linestyle='--', alpha=0.3)
        ax2.axvline(x=t['frame'], color='gray', linestyle='--', alpha=0.3)

    # Styling
    ax1.set_ylabel('Structure Population Fraction')
    ax1.set_title('Detailed Structure Evolution')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel('Dominant Structure')
    ax2.set_xlabel('Simulation Time (frames)')
    ax2.set_yticks(range(len(structures)))
    ax2.set_yticklabels(labels)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'assembly_evolution_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_aggregation_analysis(df, timestamp):
    """Create stacked area plot showing relative structure abundance"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])

    # Prepare scores for stacked plot
    scores = {
        'sheet_score': df['sheets_smooth'],
        'fiber_score': df['fibers_smooth'],
        'vesicle_score': df['vesicles_smooth'],
        'tube_score': df['tubes_smooth']
    }

    # Create DataFrame for normalized scores
    df_scores = pd.DataFrame(scores)
    df_norm = df_scores.div(df_scores.sum(axis=1), axis=0)

    # Stacked area plot
    labels = ['Sheets', 'Fibers', 'Vesicles', 'Tubes']
    ax1.stackplot(df['Frame'], [df_norm[score] for score in scores.keys()],
                 labels=labels, alpha=0.6)

    ax1.set_ylabel('Relative Abundance')
    ax1.set_title('Structure Evolution')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.grid(True, alpha=0.3)

    # State transitions
    unique_states = sorted(df['stable_structure'].unique())
    state_map = {state: idx for idx, state in enumerate(unique_states)}
    ax2.scatter(df['Frame'], df['stable_structure'].map(state_map),
                c='black', alpha=0.5, s=5)

    ax2.set_yticks(range(len(unique_states)))
    ax2.set_yticklabels(unique_states)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Dominant Structure')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'aggregation_evolution_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_evolution_confidence(df, transitions):
    """Generate analysis summary with confidence metrics"""
    summary = {
        'total_frames': len(df),
        'confident_frames': df['confident_classification'].sum(),
        'confidence_ratio': df['confident_classification'].mean(),
        'transitions': len(transitions),
        'stable_states': df['stable_structure'].value_counts().to_dict()
    }
    return summary

def load_structure_data():
    """Load structure data from multiple CSV files"""
    print("Attempting to load structure data files...")
    try:
        sheets_df = pd.read_csv('sfi_output.csv')
        vesicles_df = pd.read_csv('vfi_output.csv')
        tubes_df = pd.read_csv('tfi_output.csv')
        fibers_df = pd.read_csv('ffi_output.csv')

        # Apply size threshold filtering
        min_size = 100
        sheets_df.loc[sheets_df['avg_sheet_size'] < min_size, 'total_peptides_in_sheets'] = 0
        vesicles_df.loc[vesicles_df['avg_vesicle_size'] < min_size, 'total_peptides_in_vesicles'] = 0
        tubes_df.loc[tubes_df['avg_tube_size'] < min_size, 'total_peptides_in_tubes'] = 0
        fibers_df.loc[fibers_df['avg_fiber_size'] < min_size, 'total_peptides_in_fibers'] = 0

        print("Successfully loaded all CSV files")
        print("Columns found in vesicles file:", vesicles_df.columns.tolist())

        # Combine the data
        df = pd.DataFrame()
        df['Frame'] = sheets_df['Frame']
        df['total_peptides_in_sheets'] = sheets_df['total_peptides_in_sheets']
        df['total_peptides_in_vesicles'] = vesicles_df['total_peptides_in_vesicles']
        df['total_peptides_in_tubes'] = tubes_df['total_peptides_in_tubes']
        df['total_peptides_in_fibers'] = fibers_df['total_peptides_in_fibers']

        print(f"Combined data shape: {df.shape}")
        return df
    except FileNotFoundError as e:
        print(f"Error: Could not find file: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def plot_raw_assembly_evolution(df, timestamp):
    """Create simple two-panel plot showing raw structure fractions and dominant states"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])

    # Calculate total peptides and fractions
    total = (df['total_peptides_in_sheets'] +
             df['total_peptides_in_fibers'] +
             df['total_peptides_in_vesicles'] +
             df['total_peptides_in_tubes'])

    structures = ['sheets', 'fibers', 'vesicles', 'tubes']
    colors = ['blue', 'red', 'green', 'purple']

    # Top plot: Raw fractions
    for structure, color in zip(structures, colors):
        fraction = df[f'total_peptides_in_{structure}'] / total.where(total > 0, 1)
        ax1.plot(df['Frame'], fraction,
                label=structure.capitalize(),
                color=color,
                linewidth=1)

    # Mark non-aggregate frames
    non_agg_mask = total == 0
    if non_agg_mask.any():
        ax1.fill_between(df['Frame'], 0, 1,
                        where=non_agg_mask,
                        color='gray', alpha=0.2,
                        label='Non-aggregate')

    # Bottom plot: Dominant structure
    df['dominant_structure'] = 'non_aggregate'
    mask = total > 0
    if mask.any():
        structure_cols = [f'total_peptides_in_{s}' for s in structures]
        df.loc[mask, 'dominant_structure'] = (
            df.loc[mask, structure_cols]
            .idxmax(axis=1)
            .str.replace('total_peptides_in_', '')
        )

    # Plot dominant structure
    unique_states = ['non_aggregate'] + structures
    state_map = {state: idx for idx, state in enumerate(unique_states)}
    ax2.scatter(df['Frame'],
                df['dominant_structure'].map(state_map),
                c='black', alpha=0.5, s=2)

    # Styling
    ax1.set_ylabel('Structure Fraction')
    ax1.set_title('Raw Structure Evolution')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    ax2.set_ylabel('Dominant Structure')
    ax2.set_xlabel('Frame')
    ax2.set_yticks(range(len(unique_states)))
    ax2.set_yticklabels([s.capitalize() for s in unique_states])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'raw_assembly_evolution_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_and_plot_evolution(frame_results, output_dir):
    """Create evolution plots with both fractions and raw counts"""
    # Convert frame numbers to integer type if they aren't already
    frames = sorted(int(f) for f in frame_results.keys())

    # Prepare data arrays for both fractions and raw counts
    fractions = np.zeros((len(frames), 4))
    raw_counts = np.zeros((len(frames), 4))

    for i, frame in enumerate(frames):
        total = frame_results[frame]['total_peptides']
        # Store raw counts
        raw_counts[i, 0] = frame_results[frame]['sheets']
        raw_counts[i, 1] = frame_results[frame]['fibers']
        raw_counts[i, 2] = frame_results[frame]['vesicles']
        raw_counts[i, 3] = frame_results[frame]['tubes']

        # Calculate fractions
        if total > 0:
            fractions[i] = raw_counts[i] / total

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), height_ratios=[2, 2, 1])

    # Plot 1: Stacked fractions
    labels = ['Sheets', 'Fibers', 'Vesicles', 'Tubes']
    colors = ['blue', 'red', 'green', 'purple']
    ax1.stackplot(frames, fractions.T, labels=labels, colors=colors)
    ax1.set_ylabel('Population Fraction')
    ax1.set_title('Structure Evolution (Fractions)')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.grid(True, alpha=0.3)

    # Plot 2: Raw counts
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax2.plot(frames, raw_counts[:, i], label=label, color=color, linewidth=2)
    ax2.set_ylabel('Number of Peptides')
    ax2.set_title('Structure Evolution (Raw Counts)')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.grid(True, alpha=0.3)

    # Plot 3: Dominant structure
    dominant_structures = []
    for frame in frames:
        counts = frame_results[frame]['shape_counts']
        if not counts:
            dominant_structures.append('none')
        else:
            dominant = max(counts.items(), key=lambda x: x[1])[0]
            dominant_structures.append(dominant)

    unique_structures = sorted(set(dominant_structures))
    structure_to_y = {s: i for i, s in enumerate(unique_structures)}
    y_vals = [structure_to_y[s] for s in dominant_structures]

    ax3.scatter(frames, y_vals, c='black', s=10)
    ax3.set_yticks(range(len(unique_structures)))
    ax3.set_yticklabels([s.capitalize() for s in unique_structures])
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Dominant Structure')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'structure_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save numerical data to CSV
    df_evolution = pd.DataFrame({
        'Frame': frames,
        'Sheets_Count': raw_counts[:, 0],
        'Fibers_Count': raw_counts[:, 1],
        'Vesicles_Count': raw_counts[:, 2],
        'Tubes_Count': raw_counts[:, 3],
        'Sheets_Fraction': fractions[:, 0],
        'Fibers_Fraction': fractions[:, 1],
        'Vesicles_Fraction': fractions[:, 2],
        'Tubes_Fraction': fractions[:, 3],
        'Dominant_Structure': dominant_structures
    })
    df_evolution.to_csv(os.path.join(output_dir, 'structure_evolution_data.csv'), index=False)

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("\n=== Starting Assembly Analysis ===")

    # Load data
    df = load_structure_data()
    if df == None:
        print("Error: Could not load required data files")
        return

    # Generate plot
    print("Generating raw assembly evolution plot...")
    plot_raw_assembly_evolution(df, timestamp)
    print(f"\nPlot saved with timestamp {timestamp}")

if __name__ == "__main__":
    main()
