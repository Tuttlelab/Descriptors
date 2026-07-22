import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_descriptors(base_dirs):
    all_data = []
    for base_dir in base_dirs:
        print(f"\nScanning base directory: {base_dir}")
        # Get list of all subdirectories in base_dir
        dataset_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        for dataset_dir in dataset_dirs:
            print(f"\nProcessing dataset directory: {dataset_dir}")
            dataset_name = os.path.basename(dataset_dir)
            adi_file = os.path.join(dataset_dir, 'adi_output.csv')
            sfi_file = os.path.join(dataset_dir, 'sfi_output.csv')
            ffi_file = os.path.join(dataset_dir, 'ffi_output.csv')
            tfi_file = os.path.join(dataset_dir, 'tfi_output.csv')
            vfi_file = os.path.join(dataset_dir, 'vfi_output.csv')

            print(f"Looking for files in {dataset_dir}:")
            print(f"ADI: {adi_file}")
            print(f"SFI: {sfi_file}")
            print(f"FFI: {ffi_file}")
            print(f"TFI: {tfi_file}")
            print(f"VFI: {vfi_file}")

            if os.path.isfile(adi_file) and os.path.isfile(sfi_file) and \
               os.path.isfile(ffi_file) and os.path.isfile(tfi_file) and \
               os.path.isfile(vfi_file):
                adi_df = pd.read_csv(adi_file)
                sfi_df = pd.read_csv(sfi_file)
                ffi_df = pd.read_csv(ffi_file)
                tfi_df = pd.read_csv(tfi_file)
                vfi_df = pd.read_csv(vfi_file)

                # Merge dataframes on 'Frame' or relevant column
                merged = adi_df.merge(sfi_df, on='Frame').merge(ffi_df, on='Frame')\
                               .merge(tfi_df, on='Frame').merge(vfi_df, on='Frame')
                merged['Dataset'] = dataset_name
                all_data.append(merged)
            else:
                print(f"Missing files in {dataset_dir}, skipping.")

    if not all_data:
        print("\nNo data was loaded. Please check the file paths and ensure the descriptor files exist.")
        return None
    else:
        print(f"\nLoaded data from {len(all_data)} datasets.")
        return pd.concat(all_data, ignore_index=True)

def determine_dominant_shape(df):
    # Calculate individual shape scores
    df['sheet_score'] = df['total_peptides_in_sheets'] * df['avg_sheet_size']
    df['fiber_score'] = df['total_peptides_in_fibers'] * df['avg_fiber_size']
    df['vesicle_score'] = df['total_peptides_in_vesicles'] * df['avg_vesicle_size']
    df['tube_score'] = df['total_peptides_in_tubes'] * df['avg_tube_size']

    # Calculate total aggregates and undetermined
    total_agg = df['total_peptides_in_aggregate'] * df['avg_aggregate_size']
    known_shapes = df['sheet_score'] + df['fiber_score'] + df['vesicle_score'] + df['tube_score']
    df['undetermined_score'] = total_agg - known_shapes

    # Ensure undetermined score doesn't go negative
    df['undetermined_score'] = df['undetermined_score'].clip(lower=0)

    score_cols = ['sheet_score', 'fiber_score', 'vesicle_score', 'tube_score', 'undetermined_score']

    # Calculate mean scores for each dataset
    shape_scores = df.groupby('Dataset')[score_cols].mean()

    # Normalize scores
    shape_scores_normalized = shape_scores.div(shape_scores.sum(axis=1), axis=0)
    return shape_scores_normalized

def create_heat_map(shape_scores):
    # Save the data to CSV first
    csv_filename = 'shape_distribution_table.csv'
    shape_scores.columns = ['Sheets', 'Fibers', 'Vesicles', 'Tubes', 'Undetermined']
    shape_scores.to_csv(csv_filename, float_format='%.3f')

    plt.figure(figsize=(12, 20))  # Increased height to fit all y-axis labels

    # Rename columns for better visualization
    shape_scores.columns = ['Sheets', 'Fibers', 'Vesicles', 'Tubes', 'Undetermined']

    # Create heatmap
    sns.heatmap(shape_scores, cmap='YlOrRd',
                annot=True, fmt='.2f',
                cbar_kws={'label': 'Normalized Score'},
                annot_kws={"size": 12})  # Increase font size for annotations

    # plt.title('Shape Distribution Across Datasets', fontsize=16)
    plt.xlabel('Shape Type', fontsize=14)
    plt.ylabel('Dataset', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig('shape_distribution_heatmap.png', dpi=1000, bbox_inches='tight')
    # plt.savefig('shape_distribution_heatmap.tiff', dpi=600, bbox_inches='tight')
    plt.show()

def main():
    base_dirs = [
        'centered_files/high_ap/',
        'centered_files/mid_ap/'
    ]
    print("Loading descriptor files...")
    df = load_descriptors(base_dirs)
    if df is not None:
        print("Calculating shape scores...")
        shape_scores = determine_dominant_shape(df)

        print("Creating heat map and saving data...")
        create_heat_map(shape_scores)
        print("Heat map saved as 'shape_distribution_heatmap.png'")
        print("Data table saved as 'shape_distribution_table.csv'")
    else:
        print("No data to process.")

if __name__ == "__main__":
    main()