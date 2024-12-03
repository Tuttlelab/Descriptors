import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import savgol_filter
from scipy.stats import mode
import seaborn as sns

def calculate_structure_fractions(df, window_size=10, confidence_threshold=0.15):
    """Calculate structure fractions with confidence filtering"""
    total_peptides = (df['total_peptides_in_sheets'] +
                     df['total_peptides_in_fibers'] +
                     df['total_peptides_in_vesicles'] +
                     df['total_peptides_in_tubes'])

    structures = ['sheets', 'fibers', 'vesicles', 'tubes']

    # Calculate basic fractions
    for structure in structures:
        df[f'{structure}_fraction'] = df[f'total_peptides_in_{structure}'] / total_peptides

        # Apply Savitzky-Golay filter for smooth trend
        df[f'{structure}_smooth'] = savgol_filter(
            df[f'{structure}_fraction'],
            window_length=window_size*2+1,
            polyorder=3
        )

    # Mark low confidence predictions
    df['confident_classification'] = (
        df[[f'{s}_smooth' for s in structures]].max(axis=1) > confidence_threshold
    )

    return df

def get_most_common(x):
    """Helper function to get most common value in a series"""
    try:
        return pd.Series(x).mode().iloc[0]
    except:
        return x.iloc[0]

def identify_stable_transitions(df, min_stable_frames=20):
    """Identify genuine structure transitions, filtering out noise"""
    structures = ['sheets', 'fibers', 'vesicles', 'tubes']
    smooth_cols = [f'{structure}_smooth' for structure in structures]

    # Create mapping for structures to numeric values
    structure_to_idx = {s: i for i, s in enumerate(structures)}
    idx_to_structure = {i: s for i, s in enumerate(structures)}

    # Get dominant structure at each frame
    df['dominant_structure'] = pd.DataFrame(
        [df[col] for col in smooth_cols],
        index=structures
    ).idxmax()

    # Convert structure names to numeric values
    df['dominant_structure_numeric'] = df['dominant_structure'].map(structure_to_idx)

    # Apply rolling window with custom mode function
    df['stable_structure_numeric'] = df['dominant_structure_numeric'].rolling(
        window=min_stable_frames,
        center=True,
        min_periods=1
    ).apply(get_most_common)

    # Convert back to structure names
    df['stable_structure'] = df['stable_structure_numeric'].map(idx_to_structure)

    # Find significant transitions
    transitions = []
    prev_state = df['stable_structure'].iloc[0]

    for idx, state in enumerate(df['stable_structure']):
        if state != prev_state and df['confident_classification'].iloc[idx]:
            transitions.append({
                'frame': df['Frame'].iloc[idx],
                'from_state': prev_state,
                'to_state': state,
                'confidence': df[f'{state}_smooth'].iloc[idx]
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

    # Create structure to numeric mapping for consistent ordering
    structure_to_idx = {s: i for i, s in enumerate(structures)}

    # Bottom plot: Simplified state transitions with numeric mapping
    df['structure_numeric'] = df['stable_structure'].map(structure_to_idx)
    ax2.scatter(df['Frame'], df['structure_numeric'],
               c=df['confident_classification'].map({True: 'blue', False: 'gray'}),
               alpha=0.5, s=5)

    # Update y-axis with proper structure labels
    ax2.set_yticks(range(len(structures)))
    ax2.set_yticklabels(labels)
    ax2.set_ylim(-0.5, len(structures)-0.5)  # Add some padding

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
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'assembly_evolution_{timestamp}.png', dpi=300, bbox_inches='tight')
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

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("\n=== Starting Evolution Analysis ===")

    # Load and process data
    df = load_structure_data()
    if df is None:
        print("Error: Could not load required data files")
        return

    print("Calculating structure fractions...")
    df = calculate_structure_fractions(df)

    print("Identifying stable transitions...")
    transitions = identify_stable_transitions(df)

    # Generate visualizations and analysis
    print("Generating plots...")
    plot_multi_scale_evolution(df, transitions, timestamp)

    print("Analyzing confidence...")
    summary = analyze_evolution_confidence(df, transitions)

    # Save results
    results = pd.DataFrame(transitions)
    results.to_csv(f'evolution_transitions_{timestamp}.csv', index=False)

    # Print summary
    print("\nAnalysis Summary:")
    print(f"Total frames analyzed: {summary['total_frames']}")
    print(f"Confident frames: {summary['confident_frames']}")
    print(f"Number of transitions: {summary['transitions']}")
    print("\nStable state distribution:")
    for state, count in summary['stable_states'].items():
        print(f"  {state}: {count} frames")

    print(f"\nResults saved with timestamp {timestamp}")

if __name__ == "__main__":
    main()