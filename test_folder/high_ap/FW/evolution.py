import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import seaborn as sns
from datetime import datetime

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

        # Calculate fractions
        total = sum(dfs[s][f'total_peptides_in_{s}'] for s in dfs.keys())
        for structure in dfs.keys():
            col = f'total_peptides_in_{structure}'
            df[col] = dfs[structure][col]
            df[f'{structure}_fraction'] = df[col] / total

            # Apply smoothing
            df[f'{structure}_smooth'] = savgol_filter(
                df[f'{structure}_fraction'],
                window_length=21,  # Adjust window size as needed
                polyorder=3
            )

        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def plot_evolution(df, timestamp):
    """Create high-level visualization of structure evolution"""
    plt.figure(figsize=(12, 8))

    # Color scheme
    colors = {
        'sheets': '#1f77b4',    # blue
        'vesicles': '#2ca02c',  # green
        'tubes': '#ff7f0e',     # orange
        'fibers': '#d62728'     # red
    }

    # Plot smoothed fractions
    for structure in colors.keys():
        plt.plot(df['Frame'],
                df[f'{structure}_smooth'],
                label=structure.capitalize(),
                color=colors[structure],
                linewidth=2)

        # Add light bands showing raw data variation
        plt.fill_between(
            df['Frame'],
            df[f'{structure}_fraction'],
            df[f'{structure}_smooth'],
            color=colors[structure],
            alpha=0.1
        )

    plt.xlabel('Simulation Time (frames)')
    plt.ylabel('Population Fraction')
    plt.title('Self-Assembly Evolution')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'assembly_evolution_{timestamp}.png',
                dpi=300, bbox_inches='tight')
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