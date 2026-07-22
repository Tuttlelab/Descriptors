import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def load_all_descriptors():
    print("\n=== Loading Descriptor Files ===")

    adi = pd.read_csv('RF/adi_output.csv')
    sfi = pd.read_csv('RF/sfi_output.csv')
    ffi = pd.read_csv('RF/ffi_output.csv')
    tfi = pd.read_csv('RF/tfi_output.csv')
    vfi = pd.read_csv('RF/vfi_output.csv')

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
    sheet_mask = df['avg_sheet_size'] >= 5 #150
    df['sheet_score'] = df.apply(lambda row:
        row['total_peptides_in_sheets'] * row['avg_sheet_size'] if sheet_mask[row.name] else 0, axis=1)

    # Apply fiber threshold
    fiber_mask = df['avg_fiber_size'] >= 5# 1500
    df['fiber_score'] = df.apply(lambda row:
        row['total_peptides_in_fibers'] * row['avg_fiber_size'] if fiber_mask[row.name] else 0, axis=1)

    # Apply vesicle threshold
    vesicle_mask = df['avg_vesicle_size'] >= 5 #300
    df['vesicle_score'] = df.apply(lambda row:
        row['total_peptides_in_vesicles'] * row['avg_vesicle_size'] if vesicle_mask[row.name] else 0, axis=1)

    # Apply tube threshold
    tube_mask = df['avg_tube_size'] >= 5 #130
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
    import seaborn as sns
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

    # Define display labels
    label_map = {
        'undetermined': 'undet.',
        'sheet': 'sheet',
        'vesicle': 'vesicle',
        'fiber': 'fibre',
        'tube': 'tube'
    }

    # State transitions with compact spacing
    ordered_shapes = ['undetermined', 'sheet', 'vesicle', 'fiber', 'tube']
    unique_shapes = [shape for shape in ordered_shapes if shape in df['shapes'].unique()]
    # shape_map = {shape: idx/4 for idx, shape in enumerate(unique_shapes)}  # Divide by 4 to reduce spacing

    shape_map = {shape: 0.15 + (idx * 0.7/(len(unique_shapes)-1))
                 for idx, shape in enumerate(unique_shapes)}

        # Set wider y-axis limits for padding
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

    # ax.set_yticks([idx/4 for idx in range(len(unique_shapes))])  # Adjust ticks to match new spacing
    ax.set_yticks([])
    ax.set_yticklabels([])  # Hide y tick labels
    ax.set_xticklabels([])  # Hide x tick labels
    # ax.set_yticklabels([label_map[shape] for shape in unique_shapes])
    # ax.set_xlabel('Time (ns)', fontsize=16)
    # ax.set_ylabel('Dominant Shape', labelpad=0, fontsize=16)
    ax.spines['left'].set_visible(True)  # Hide y axis line
    # ax.tick_params(axis='both', which='major', labelsize=16)
    # ax.grid(False)

    # Set x-axis limits and ticks
    ax.set_xlim(0, 1500)
    ax.set_xticks([0, 250, 500, 750, 1000, 1250, 1500])

    plt.tight_layout()
    plt.savefig(f'RF/dominant_{timestamp}.png', dpi=600, bbox_inches='tight')

def main():
    timestamp = datetime.now().strftime("%m%d_%H%M")
    print("\n=== Starting Analysis ===")

    df = load_all_descriptors()
    df = analyze_state_transitions(df)

    # Generate evolution plot
    plot_aggregation_analysis(df, timestamp)

    # Save analysis results
    df[['Frame', 'shapes']].to_csv(
        f'RF/dominant_{timestamp}.csv', index=False)

    print(f"\nResults saved with timestamp {timestamp}")

if __name__ == "__main__":
    main()