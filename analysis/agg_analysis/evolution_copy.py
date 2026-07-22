import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import savgol_filter
from scipy.stats import mode
import seaborn as sns

def calculate_structure_fractions(df):
    """Calculate structure fractions with thresholds from decision.py"""
    # Apply thresholds as in decision.py
    sheet_mask = df['avg_sheet_size'] >= 150
    fiber_mask = df['avg_fiber_size'] >= 1500
    vesicle_mask = df['avg_vesicle_size'] >= 300
    tube_mask = df['avg_tube_size'] >= 130

    # Calculate scores
    df['sheet_score'] = df.apply(lambda row:
        row['total_peptides_in_sheets'] * row['avg_sheet_size'] if sheet_mask[row.name] else 0, axis=1)
    df['fiber_score'] = df.apply(lambda row:
        row['total_peptides_in_fibers'] * row['avg_fiber_size'] if fiber_mask[row.name] else 0, axis=1)
    df['vesicle_score'] = df.apply(lambda row:
        row['total_peptides_in_vesicles'] * row['avg_vesicle_size'] if vesicle_mask[row.name] else 0, axis=1)
    df['tube_score'] = df.apply(lambda row:
        row['total_peptides_in_tubes'] * row['avg_tube_size'] if tube_mask[row.name] else 0, axis=1)

    scores = ['sheet_score', 'fiber_score', 'vesicle_score', 'tube_score']

    # Check if any structures are present
    df['has_structures'] = df[scores].max(axis=1) > 0

    # Calculate total score for normalization
    total_score = df[scores].sum(axis=1)

    # Calculate fractions
    for score, structure in zip(scores, ['sheets', 'fibers', 'vesicles', 'tubes']):
        df[f'{structure}_fraction'] = df[score] / total_score
        df[f'{structure}_fraction'] = df[f'{structure}_fraction'].fillna(0)

    # Mark as undetermined when no significant structures
    df['confident_classification'] = df['has_structures']

    return df

def identify_stable_transitions(df, min_stable_frames=20):
    """Identify genuine structure transitions, filtering out noise"""
    structures = ['sheets', 'fibers', 'vesicles', 'tubes']
    fraction_cols = [f'{structure}_fraction' for structure in structures]

    # Get dominant structure at each frame
    df['dominant_structure'] = 'undetermined'
    mask = df['has_structures']
    df.loc[mask, 'dominant_structure'] = pd.DataFrame(
        [df[col] for col in fraction_cols],
        index=structures
    ).idxmax()[mask]

    # Convert to numeric for rolling calculation
    structure_to_num = {s: i for i, s in enumerate(['undetermined'] + structures)}
    df['structure_code'] = df['dominant_structure'].map(structure_to_num)

    # Apply rolling mode with handling for empty windows
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

    # Map back to structure names
    num_to_structure = {v: k for k, v in structure_to_num.items()}
    df['stable_structure'] = df['stable_structure_code'].map(num_to_structure)

    # Find transitions
    transitions = []
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

def plot_multi_scale_evolution(df, transitions, timestamp):
    """Create dual-view evolution plot showing both micro and macro trends"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[2, 1])

    colors = sns.color_palette("husl", 4)
    structures = ['sheets', 'fibers', 'vesicles', 'tubes']
    labels = [s.capitalize() for s in structures]

    # Top plot: Detailed view with all structures
    for i, structure in enumerate(structures):
        ax1.plot(df['Frame'], df[f'{structure}_fraction'],
                 label=labels[i], color=colors[i], linewidth=2)

        # Remove confidence bands related to smoothing
        # ...removed code for confidence bands...

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
    # Remove the title
    # ax1.set_title('Detailed Structure Evolution')
    ax1.legend(loc='best')  # Legend inside the plot
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel('Dominant Structure')
    ax2.set_xlabel('Simulation Time (frames)')
    ax2.set_yticks(range(len(structures)))
    ax2.set_yticklabels(labels)
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
        df['avg_sheet_size'] = sheets_df['avg_sheet_size']
        df['total_peptides_in_vesicles'] = vesicles_df['total_peptides_in_vesicles']
        df['avg_vesicle_size'] = vesicles_df['avg_vesicle_size']
        df['total_peptides_in_tubes'] = tubes_df['total_peptides_in_tubes']
        df['avg_tube_size'] = tubes_df['avg_tube_size']
        df['total_peptides_in_fibers'] = fibers_df['total_peptides_in_fibers']
        df['avg_fiber_size'] = fibers_df['avg_fiber_size']

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