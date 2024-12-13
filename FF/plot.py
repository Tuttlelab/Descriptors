import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import savgol_filter
from scipy.stats import mode
import seaborn as sns
import os
from scipy.ndimage import gaussian_filter1d

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

def plot_aggregation_evolution(df, plots_dir, timestamp):
    """Plot size evolution and aggregate count"""
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Smooth the cluster size data
    size_smooth = gaussian_filter1d(df['LargestClusterSize'] / 8, sigma=3)  # Convert beads to dipeptides

    # Plot average aggregate size on y1-axis
    ax1.set_xlabel('Frame', fontsize=12)
    ax1.set_ylabel('Dipeptides per Aggregate', color='tab:blue', fontsize=12)
    ax1.plot(df['Frame'], size_smooth, color='tab:blue', alpha=0.6)
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    plt.title('Aggregation Evolution', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, f'aggregation_evolution_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_shape_analysis(df, plots_dir, timestamp):
    """Plot shape classification over time"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create boolean masks for each shape
    shapes = ['sheet', 'fiber', 'vesicle', 'tube', 'spherical_aggregate', 'non-aggregate']
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'gray']

    for shape, color in zip(shapes, colors):
        # Create mask for this shape (including when it's part of multiple detections)
        mask = df['Shape'].str.contains(shape, case=False, na=False)
        if mask.any():
            ax.scatter(df[mask]['Frame'], [shapes.index(shape)] * mask.sum(),
                      c=color, label=shape.capitalize(), alpha=0.6)

    ax.set_yticks(range(len(shapes)))
    ax.set_yticklabels([s.capitalize() for s in shapes])
    ax.set_xlabel('Frame', fontsize=12)
    plt.title('Shape Classification', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.savefig(os.path.join(plots_dir, f'shape_classification_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_shape_metrics(df, plots_dir, timestamp):
    """Plot metrics for each shape type"""
    # Sheet metrics
    if 'sheet_planarity_rmsd' in df.columns:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        ax1.plot(df['Frame'], df['sheet_planarity_rmsd'], label='Planarity RMSD')
        ax1.plot(df['Frame'], df['sheet_curvature_rmsd'], label='Curvature RMSD')
        ax1.set_ylabel('RMSD')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(df['Frame'], df['sheet_thickness_ratio'], label='Thickness Ratio')
        ax2.plot(df['Frame'], df['sheet_elongation_ratio'], label='Elongation Ratio')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Ratio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle('Sheet Metrics Evolution', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'sheet_metrics_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Fiber metrics
    if 'fiber_cylindrical_score' in df.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(df['Frame'], df['fiber_cylindrical_score'], label='Cylindrical Score')
        plt.plot(df['Frame'], df['fiber_cross_section_var'], label='Cross-section Variation')
        plt.xlabel('Frame')
        plt.ylabel('Score')
        plt.title('Fiber Metrics Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, f'fiber_metrics_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Vesicle metrics
    if 'vesicle_sphericity' in df.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(df['Frame'], df['vesicle_sphericity'], label='Sphericity')
        plt.plot(df['Frame'], df['vesicle_asphericity'], label='Asphericity')
        plt.xlabel('Frame')
        plt.ylabel('Score')
        plt.title('Vesicle Metrics Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, f'vesicle_metrics_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Tube metrics
    if 'tube_radial_std' in df.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(df['Frame'], df['tube_radial_std'], label='Radial STD')
        plt.plot(df['Frame'], df['tube_angular_uniformity'], label='Angular Uniformity')
        plt.xlabel('Frame')
        plt.ylabel('Score')
        plt.title('Tube Metrics Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, f'tube_metrics_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def analyze_and_plot_evolution(frame_results=None, output_dir='results'):
    """Generate comprehensive plots from shape evolution CSV"""
    analysis_dir = os.path.join(output_dir, 'analysis')
    if not os.path.exists(analysis_dir):
        print("No analysis directory found")
        return

    csv_files = [f for f in os.listdir(analysis_dir) if f.startswith('shape_evolution_') and f.endswith('.csv')]
    if not csv_files:
        print("No shape evolution CSV files found")
        return

    # Get most recent CSV file
    latest_csv = max(csv_files)
    csv_path = os.path.join(analysis_dir, latest_csv)

    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Create plots directory
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate all plots
        plot_aggregation_evolution(df, plots_dir, timestamp)
        plot_shape_analysis(df, plots_dir, timestamp)
        plot_shape_metrics(df, plots_dir, timestamp)

        print(f"Plots saved in {plots_dir}")

    except Exception as e:
        print(f"Error generating plots: {str(e)}")

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
