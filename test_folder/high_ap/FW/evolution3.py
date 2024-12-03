import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Apply font settings from descriptor_plots.py
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

def validate_fractions(df):
    """Validate that fractions sum to approximately 1"""
    structures = ['sheets', 'vesicles', 'tubes', 'fibers']
    fraction_sum = sum(df[f'{s}_fraction'] for s in structures)
    tolerance = 0.01  # 1% tolerance

    if not np.allclose(fraction_sum, 1, rtol=tolerance):
        max_deviation = np.max(np.abs(fraction_sum - 1))
        print(f"Warning: Fractions don't sum to 1. Maximum deviation: {max_deviation:.3f}")
        return False
    return True

def load_and_process_data():
    """Load structure data and calculate fractions"""
    try:
        # Load all structure files
        dfs = {
            'sheets': pd.read_csv('sfi_output.csv'),
            'vesicles': pd.read_csv('vfi_output.csv'),
            'tubes': pd.read_csv('tfi_output.csv'),
            'fibers': pd.read_csv('ffi_output.csv')
        }

        # Combine into single dataframe
        df = pd.DataFrame()
        df['Frame'] = dfs['sheets']['Frame']

        # Calculate total peptides per frame
        total = pd.Series(0, index=df.index)
        for structure in dfs.keys():
            col = f'total_peptides_in_{structure}'
            df[col] = dfs[structure][col]
            total += df[col]

        if (total <= 0).any():
            raise ValueError("Total number of peptides is zero or negative in some frames")

        # Calculate fractions
        for structure in dfs.keys():
            col = f'total_peptides_in_{structure}'
            df[f'{structure}_fraction'] = df[col] / total

        # Validate fractions
        if not validate_fractions(df):
            print("Warning: Data validation failed. Results may be unreliable.")

        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def plot_evolution(df, timestamp):
    """Create visualization of structure evolution using matplotlib"""
    # Create figure with size similar to descriptor plots
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    colors = {
        'sheets': '#1f77b4',
        'vesicles': '#2ca02c',
        'tubes': '#ff7f0e',
        'fibers': '#d62728'
    }

    for structure, color in colors.items():
        ax.plot(
            df['Frame'],
            df[f'{structure}_fraction'],
            label=structure.capitalize(),
            linewidth=0.75,
            alpha=0.6,
            color=color
        )

    ax.set_xlabel('Frame', fontsize=7)
    ax.set_ylabel('Population Fraction', fontsize=7)
    ax.tick_params(axis='both', labelsize=6)
    ax.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(f'assembly_evolution_{timestamp}.pdf',
                dpi=1600, bbox_inches='tight', format='pdf')
    plt.close()

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load and process data
    df = load_and_process_data()
    if df is None:
        return

    # Create visualization
    plot_evolution(df, timestamp)

    # Save processed data
    df.to_csv(f'evolution_data_{timestamp}.csv', index=False)
    print(f"Analysis complete. Files saved with timestamp: {timestamp}")

if __name__ == "__main__":
    main()
