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

    # Define overlap pairs
    overlaps = ['sfi_vs_vfi', 'sfi_vs_tfi', 'sfi_vs_ffi',
                'vfi_vs_tfi', 'vfi_vs_ffi', 'tfi_vs_ffi']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c',
              '#d62728', '#9467bd', '#8c564b']

    # Initialize DataFrame to store top values
    top_values = pd.DataFrame()

    # Select top 5 values for each overlap
    for overlap in overlaps:
        top_overlap = df.nlargest(5, overlap)[['Frame', overlap]]
        top_overlap['Overlap'] = overlap  # Add a column to identify the overlap
        top_values = pd.concat([top_values, top_overlap], ignore_index=True)

    # Remove duplicates in case the same frame appears in multiple overlaps
    top_values = top_values.drop_duplicates(subset=['Frame', 'Overlap'])

    # Save top values to CSV
    top_values.to_csv('top_overlap_frames.csv', index=False)

    # Create plot
    fig, ax = plt.subplots(figsize=(3.27, 1.97))

    # Plot top values
    for overlap, color in zip(overlaps, colors):
        data = top_values[top_values['Overlap'] == overlap]
        if not data.empty:
            ax.scatter(data['Frame'], data[overlap],
                       label=overlap.replace('_', '-'),
                       color=color, s=5)

            # Add frame numbers as annotations
            for _, row in data.iterrows():
                ax.annotate(f'{int(row["Frame"])}',
                            xy=(row['Frame'], row[overlap]),
                            xytext=(5, 5),
                            textcoords='offset points',
                            fontsize=4,
                            color=color,
                            rotation=45)

    # Formatting
    ax.set_xlabel('Frame', fontsize=7)
    ax.set_ylabel('Number of overlapping peptides', fontsize=7)
    ax.tick_params(axis='both', labelsize=6)
    ax.legend(fontsize=6, loc='upper right')

    # Save plot
    plt.savefig('top_overlap_points.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
