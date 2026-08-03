import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def load_all_descriptors(data_dir='FF'):
    print(f"\n=== Loading Descriptor Files from '{data_dir}' ===")
    try:
        adi = pd.read_csv(os.path.join(data_dir, 'adi_output.csv'))
        sfi = pd.read_csv(os.path.join(data_dir, 'sfi_output.csv'))
        ffi = pd.read_csv(os.path.join(data_dir, 'ffi_output.csv'))
        tfi = pd.read_csv(os.path.join(data_dir, 'tfi_output.csv'))
        vfi = pd.read_csv(os.path.join(data_dir, 'vfi_output.csv'))
    except FileNotFoundError as e:
        print(f"Error loading descriptor CSV files: {e}")
        print(f"Expected files in directory '{data_dir}': adi_output.csv, sfi_output.csv, ffi_output.csv, tfi_output.csv, vfi_output.csv")
        return None

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

def plot_aggregation_analysis(df, timestamp, data_dir='FF'):
    import matplotlib as mpl
    import seaborn as sns
    os.makedirs(data_dir, exist_ok=True)
    mpl.rcParams.update({'font.size': 16})

    # Create a more compact figure
    fig, ax = plt.subplots(figsize=(7, 1))

    # Get same color palette as evolution plot
    colors = sns.color_palette("husl", 4)
    color_map = {
        'sheet': colors[0],
        'fiber': colors[1],
        'vesicle': colors[2],
        'tube': colors[3],
        'undetermined': 'gray'
    }

    # State transitions with compact spacing
    ordered_shapes = ['undetermined', 'sheet', 'vesicle', 'fiber', 'tube']
    unique_shapes = [shape for shape in ordered_shapes if shape in df['shapes'].unique()]

    shape_map = {shape: 0.15 + (idx * 0.7/(len(unique_shapes)-1))
                 for idx, shape in enumerate(unique_shapes)}

    ax.set_ylim(-0.1, 1.1)

    # Convert frames to nanoseconds
    time_ns = df['Frame'] / 8

    # Plot with smaller markers
    for shape in unique_shapes:
        mask = df['shapes'] == shape
        ax.scatter(time_ns[mask],
                  [shape_map[shape]] * mask.sum(),
                  c=[color_map[shape]],
                  alpha=0.5,
                  s=2)

    ax.set_yticks([])
    ax.set_yticklabels([])  # Hide y tick labels
    ax.set_xticklabels([])  # Hide x tick labels
    ax.spines['left'].set_visible(True)  # Hide y axis line

    # Set x-axis limits and ticks
    ax.set_xlim(0, 1500)
    ax.set_xticks([0, 250, 500, 750, 1000, 1250, 1500])

    plt.tight_layout()
    output_png = os.path.join(data_dir, f'dominant_{timestamp}.png')
    plt.savefig(output_png, dpi=600, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Dominant shape analysis for FF peptide structures.")
    parser.add_argument("-d", "--data-dir", default="FF", help="Directory containing descriptor CSV files (default: FF)")
    args = parser.parse_args()

    data_dir = args.data_dir
    timestamp = datetime.now().strftime("%m%d_%H%M")
    print(f"\n=== Starting Dominant Shape Analysis ({data_dir}) ===")

    df = load_all_descriptors(data_dir)
    if df is None:
        print(f"Error: Could not load descriptor files from '{data_dir}'")
        return

    df = analyze_state_transitions(df)

    # Generate evolution plot
    plot_aggregation_analysis(df, timestamp, data_dir)

    # Save analysis results
    os.makedirs(data_dir, exist_ok=True)
    output_csv = os.path.join(data_dir, f'dominant_{timestamp}.csv')
    df[['Frame', 'shapes']].to_csv(output_csv, index=False)

    print(f"\nResults saved to '{data_dir}' with timestamp {timestamp}")

if __name__ == "__main__":
    main()