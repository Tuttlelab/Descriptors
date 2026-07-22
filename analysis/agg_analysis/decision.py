import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def load_all_descriptors():
    print("\n=== Loading Descriptor Files ===")

    adi = pd.read_csv('adi_output.csv')
    sfi = pd.read_csv('sfi_output.csv')
    ffi = pd.read_csv('ffi_output.csv')
    tfi = pd.read_csv('tfi_output.csv')
    vfi = pd.read_csv('vfi_output.csv')

    print("\nColumns in each file:")
    print("ADI columns:", adi.columns.tolist())
    print("SFI columns:", sfi.columns.tolist())
    print("FFI columns:", ffi.columns.tolist())
    print("TFI columns:", tfi.columns.tolist())
    print("VFI columns:", vfi.columns.tolist())

    merged = pd.merge(sfi, ffi, on='Frame')
    merged = pd.merge(merged, tfi, on='Frame')
    merged = pd.merge(merged, vfi, on='Frame')

    print("\nFinal merged columns:", merged.columns.tolist())
    print(f"Total frames: {len(merged)}")
    return merged

def analyze_state_transitions(df):
    """Analyze transitions with thresholds"""
    # Calculate relative dominance scores with thresholds
    sheet_mask = df['avg_sheet_size'] >= 150
    df['sheet_score'] = df.apply(lambda row:
        row['total_peptides_in_sheets'] * row['avg_sheet_size'] if sheet_mask[row.name] else 0, axis=1)

    # Apply fiber threshold
    fiber_mask = df['avg_fiber_size'] >= 1500
    df['fiber_score'] = df.apply(lambda row:
        row['total_peptides_in_fibers'] * row['avg_fiber_size'] if fiber_mask[row.name] else 0, axis=1)

    # Apply vesicle threshold
    vesicle_mask = df['avg_vesicle_size'] >= 300
    df['vesicle_score'] = df.apply(lambda row:
        row['total_peptides_in_vesicles'] * row['avg_vesicle_size'] if vesicle_mask[row.name] else 0, axis=1)

    # Apply tube threshold
    tube_mask = df['avg_tube_size'] >= 130
    df['tube_score'] = df.apply(lambda row:
        row['total_peptides_in_tubes'] * row['avg_tube_size'] if tube_mask[row.name] else 0, axis=1)

    scores = ['sheet_score', 'fiber_score', 'vesicle_score', 'tube_score']
    states = ['sheet', 'fiber', 'vesicle', 'tube']

    # Check if any structures are present
    df['has_structures'] = df[scores].max(axis=1) > 0

    # Assign dominant shapes or undetermined
    df['shapes'] = 'undetermined'
    df.loc[df['has_structures'], 'shapes'] = pd.DataFrame(
        [df[score] for score in scores], index=states
    ).idxmax(axis=0)[df['has_structures']]

    return df

def plot_aggregation_analysis(df, timestamp):
    import matplotlib as mpl
    mpl.rcParams.update({'font.size': 16})

    # Create figure with only one subplot now that top plot is commented out
    fig, ax = plt.subplots(figsize=(12, 5))

    """
    # Stacked area plot
    scores = ['sheet_score', 'fiber_score', 'vesicle_score', 'tube_score']
    labels = ['Sheets', 'Fibers', 'Vesicles', 'Tubes']

    df_norm = df[scores].copy()
    row_sums = df_norm.sum(axis=1)
    df_norm = df_norm.div(row_sums, axis=0).fillna(0)

    ax1.stackplot(df['Frame'], [df_norm[score] for score in scores],
                  labels=labels, alpha=0.6)

    ax1.set_ylabel('Relative Abundance', fontsize=16)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, fontsize=16)
    ax1.grid(False)
    ax1.set_xlabel('Time (ns)', fontsize=16)
    """

    # State transitions
    ordered_shapes = ['undetermined', 'sheet', 'vesicle', 'fiber', 'tube']
    unique_shapes = [shape for shape in ordered_shapes if shape in df['shapes'].unique()]
    shape_map = {shape: idx for idx, shape in enumerate(unique_shapes)}

    # Convert frames to nanoseconds by dividing by 8
    time_ns = df['Frame'] / 8

    ax.scatter(time_ns, df['shapes'].map(shape_map),
               c='black', alpha=0.5, s=5)

    ax.set_yticks(range(len(unique_shapes)))
    ax.set_yticklabels(unique_shapes)
    ax.set_xlabel('Time (ns)', fontsize=16)
    ax.set_ylabel('Shapes', labelpad=0, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(False)

    # Set x-axis limits and ticks
    ax.set_xlim(0, 1500)
    ax.set_xticks([0, 250, 500, 750, 1000, 1250, 1500])

    plt.tight_layout()
    plt.savefig(f'aggregation_evolution_{timestamp}.png', dpi=600, bbox_inches='tight')

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("\n=== Starting Analysis ===")

    df = load_all_descriptors()
    df = analyze_state_transitions(df)

    # Generate evolution plot
    plot_aggregation_analysis(df, timestamp)

    # Save analysis results
    df[['Frame', 'shapes']].to_csv(
        f'aggregation_analysis_{timestamp}.csv', index=False)

    print(f"\nResults saved with timestamp {timestamp}:")
    print(f"- Evolution plot: aggregation_evolution_{timestamp}.png")
    print(f"- Analysis data: aggregation_analysis_{timestamp}.csv")

if __name__ == "__main__":
    main()