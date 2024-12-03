import pandas as pd
import glob
import os
import numpy as np
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt

def read_descriptor_data(descriptor_name):
    """Read and process data for a single descriptor"""
    results_dir = f'{descriptor_name}_results/'
    pattern = f'{descriptor_name}_frame_results_*.csv'
    files = sorted(glob.glob(os.path.join(results_dir, pattern)))

    if not files:
        print(f"No files found for {descriptor_name}")
        return None

    # Read and combine all files for this descriptor
    dfs = []
    for file in files:
        df = pd.read_csv(file, usecols=['Frame', 'Peptides'])
        dfs.append(df)

    return pd.concat(dfs).sort_values('Frame')

def calculate_overlap(descriptors_data, frame):
    """Calculate overlap between all descriptors for a specific frame"""
    # Get peptides for each descriptor at this frame
    frame_data = {}
    for desc in descriptors_data:
        frame_peptides = descriptors_data[desc][
            descriptors_data[desc]['Frame'] == frame]['Peptides'].iloc[0]
        # Convert string representation of list to actual list
        frame_data[desc] = set(eval(frame_peptides))

    # Calculate pairwise overlaps
    overlaps = {}
    for desc1, desc2 in combinations(descriptors_data.keys(), 2):
        overlap = len(frame_data[desc1].intersection(frame_data[desc2]))
        overlaps[f'{desc1}_vs_{desc2}'] = overlap

    # Calculate overlap between all descriptors
    all_overlap = len(set.intersection(*[frame_data[desc] for desc in descriptors_data]))
    overlaps['all_descriptors'] = all_overlap

    return overlaps

def main():
    # descriptors = ["sfi", "vfi", "tfi", "ffi"]

    # # Read data for each descriptor
    # descriptors_data = {}
    # for desc in descriptors:
    #     data = read_descriptor_data(desc)
    #     if data is not None:
    #         descriptors_data[desc] = data

    # if len(descriptors_data) < 2:
    #     print("Not enough descriptor data found")
    #     return

    # # Calculate overlap for each frame
    # results = []
    # frames = range(0, 12001)  # 0 to 12000

    # for frame in frames:
    #     try:
    #         overlap_data = calculate_overlap(descriptors_data, frame)
    #         overlap_data['Frame'] = frame
    #         results.append(overlap_data)
    #     except (IndexError, KeyError):
    #         print(f"Skipping frame {frame}: data not available")
    #         continue

    # # Create DataFrame and save results
    # results_df = pd.DataFrame(results)
    # timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    # output_file = f'descriptor_overlap_{timestamp}.csv'
    # results_df.to_csv(output_file, index=False)
    # print(f"Results saved to {output_file}")

    # Set style
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'

    # Read data
    files = glob.glob('descriptor_overlap_20241202_1430.csv')
    if not files:
        print("No descriptor overlap files found")
        exit()

    df = pd.read_csv(files[0])

    # Create plot
    fig, ax = plt.subplots(figsize=(3.27, 1.97))  # RSC single column

    # Plot each overlap with different colors and transparency
    overlaps = ['sfi_vs_vfi', 'sfi_vs_tfi', 'sfi_vs_ffi',
                'vfi_vs_tfi', 'vfi_vs_ffi', 'tfi_vs_ffi']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c',
              '#d62728', '#9467bd', '#8c564b']
    alpha = 0.7  # Set transparency level

    for overlap, color in zip(overlaps, colors):
        ax.plot(df['Frame'], df[overlap],
                label=overlap.replace('_', '-'),
                color=color, linewidth=0.75, alpha=alpha)

    # Formatting
    ax.set_xlabel('Frame', fontsize=7)
    ax.set_ylabel('Number of overlapping peptides', fontsize=7)
    ax.tick_params(axis='both', labelsize=6)
    ax.legend(fontsize=6)

    # Save plot
    plt.tight_layout()
    plt.savefig('descriptor_overlap.png',
                dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()
