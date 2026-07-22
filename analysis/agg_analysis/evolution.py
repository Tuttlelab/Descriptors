import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import savgol_filter
from scipy.stats import mode
import seaborn as sns

def calculate_structure_fractions(df, confidence_threshold=0.15):
    """Calculate structure fractions with confidence filtering"""
    total_peptides = (df['total_peptides_in_sheets'] +
                      df['total_peptides_in_fibers'] +
                      df['total_peptides_in_vesicles'] +
                      df['total_peptides_in_tubes'])

    structures = ['sheets', 'fibers', 'vesicles', 'tubes']

    # Calculate basic fractions
    for structure in structures:
        df[f'{structure}_fraction'] = df[f'total_peptides_in_{structure}'] / total_peptides

    # Remove smoothing
    # ...removed code related to smoothing...

    # Mark low confidence predictions using fractions directly
    df['confident_classification'] = (
        df[[f'{s}_fraction' for s in structures]].max(axis=1) > confidence_threshold
    )

    return df

def identify_stable_transitions(df, min_stable_frames=20):
    """Identify genuine structure transitions, filtering out noise"""
    structures = ['sheets', 'fibers', 'vesicles', 'tubes']
    fraction_cols = [f'{structure}_fraction' for structure in structures]

    # Get dominant structure at each frame using fractions
    df['dominant_structure'] = pd.DataFrame(
        [df[col] for col in fraction_cols],
        index=structures
    ).idxmax()

    # Convert dominant_structure to categorical codes
    df['dominant_structure_code'] = df['dominant_structure'].astype('category').cat.codes

    # Define a safe mode function that handles empty arrays
    def safe_mode(x):
        if len(x) == 0 or pd.isna(x).all():
            return 0
        try:
            # Use stats.mode and handle the case when mode is empty
            mode_result = mode(x, keepdims=False)
            return mode_result if isinstance(mode_result, (int, float)) else mode_result[0]
        except:
            # If mode fails, return the first value or 0
            return x[0] if len(x) > 0 else 0

    # Apply rolling mode with the safe function
    df['stable_structure_code'] = (
        df['dominant_structure_code']
        .rolling(window=min_stable_frames, center=True, min_periods=1)
        .apply(safe_mode)
    ).fillna(0).astype(int)

    # Map codes back to structure names
    category_mapping = dict(enumerate(df['dominant_structure'].astype('category').cat.categories))
    df['stable_structure'] = df['stable_structure_code'].map(category_mapping)

    # Find significant transitions
    transitions = []
    prev_state = df['stable_structure'].iloc[0]

    for idx, state in enumerate(df['stable_structure']):
        if state != prev_state and df['confident_classification'].iloc[idx]:
            transitions.append({
                'frame': df['Frame'].iloc[idx],
                'from_state': prev_state,
                'to_state': state,
                'confidence': df[f'{state}_fraction'].iloc[idx]
            })
            prev_state = state

    return transitions

def plot_multi_scale_evolution(df, transitions, timestamp):
    """Create evolution plot showing single view of structure populations"""
    plt.figure(figsize=(8, 5))

    colors = sns.color_palette("husl", 4)
    structures = ['sheets', 'fibers', 'vesicles', 'tubes']
    labels = [s.capitalize() for s in structures]

    # Plot structure fractions
    for i, structure in enumerate(structures):
        plt.plot(df['Frame'], df[f'{structure}_fraction'],
                label=labels[i], color=colors[i], linewidth=2)

    # Mark low confidence regions
    low_conf_regions = ~df['confident_classification']
    if low_conf_regions.any():
        plt.fill_between(df['Frame'], 0, 1,
                        where=low_conf_regions,
                        color='gray', alpha=0.1, label='Low Confidence')

    # Add transition markers
    for t in transitions:
        if t['frame'] > 200:  # Only show transitions after transient period
            plt.axvline(x=t['frame'], color='gray', linestyle='--', alpha=0.3)

    # Styling
    plt.ylabel('Structure Population Fraction')
    plt.xlabel('Time (ns)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5)
    plt.grid(False)
    plt.ylim(bottom=0)

    # Set x-axis limits and ticks
    plt.xlim(0, 1500)
    plt.xticks([0, 250, 500, 750, 1000, 1250, 1500])

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