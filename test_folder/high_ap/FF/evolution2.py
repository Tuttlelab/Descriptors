import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import plotly.graph_objects as go
from datetime import datetime

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

            # Apply smoothing
            df[f'{structure}_smooth'] = savgol_filter(
                df[f'{structure}_fraction'],
                window_length=21,  # Adjust window size as needed
                polyorder=3
            )

        # Validate fractions
        if not validate_fractions(df):
            print("Warning: Data validation failed. Results may be unreliable.")

        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def plot_evolution(df, timestamp):
    """Create interactive visualization of structure evolution"""
    fig = go.Figure()

    colors = {
        'sheets': '#1f77b4',
        'vesicles': '#2ca02c',
        'tubes': '#ff7f0e',
        'fibers': '#d62728'
    }

    for structure in colors.keys():
        # Add smooth line
        fig.add_trace(go.Scatter(
            x=df['Frame'],
            y=df[f'{structure}_smooth'],
            name=f"{structure.capitalize()} (Smoothed)",
            line=dict(color=colors[structure], width=2),
            hovertemplate="Frame: %{x}<br>Fraction: %{y:.3f}<extra></extra>"
        ))

        # Add raw data with lower opacity
        fig.add_trace(go.Scatter(
            x=df['Frame'],
            y=df[f'{structure}_fraction'],
            name=f"{structure.capitalize()} (Raw)",
            line=dict(color=colors[structure], width=1, dash='dot'),
            opacity=0.3,
            hovertemplate="Frame: %{x}<br>Raw Fraction: %{y:.3f}<extra></extra>"
        ))

    fig.update_layout(
        title='Self-Assembly Evolution',
        xaxis_title='Simulation Time (frames)',
        yaxis_title='Population Fraction',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        )
    )

    # Save as interactive HTML
    fig.write_html(f'assembly_evolution_{timestamp}.html')

    # Also save as static image for backup
    fig.write_image(f'assembly_evolution_{timestamp}.png')

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
