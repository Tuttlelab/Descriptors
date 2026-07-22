import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_descriptors(base_dirs):
    all_data = []
    for base_dir in base_dirs:
        print(f"\nScanning base directory: {base_dir}")
        # CSV files are directly in the peptide folder
        dataset_name = os.path.basename(base_dir)
        adi_file = os.path.join(base_dir, 'adi_output.csv')
        sfi_file = os.path.join(base_dir, 'sfi_output.csv')
        ffi_file = os.path.join(base_dir, 'ffi_output.csv')
        tfi_file = os.path.join(base_dir, 'tfi_output.csv')
        vfi_file = os.path.join(base_dir, 'vfi_output.csv')

        print(f"Looking for files in {base_dir}:")
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
            print(f"Missing files in {base_dir}, skipping.")

    if not all_data:
        print("\nNo data was loaded. Please check the file paths and ensure the descriptor files exist.")
        return None
    else:
        print(f"\nLoaded data from {len(all_data)} datasets.")
        return pd.concat(all_data, ignore_index=True)

def get_detected_shapes(data, num_frames=10):
    """Calculate normalized shape scores averaged over the last n frames"""
    # Get the last n frames
    last_frame = data['Frame'].max()
    start_frame = max(0, last_frame - num_frames + 1)
    last_frames_data = data[data['Frame'] >= start_frame]

    # Calculate average scores for each shape over the last n frames
    shape_scores = {
        'Sheets': last_frames_data['total_peptides_in_sheets'] * last_frames_data['avg_sheet_size'],
        'Fibers': last_frames_data['total_peptides_in_fibers'] * last_frames_data['avg_fiber_size'],
        'Vesicles': last_frames_data['total_peptides_in_vesicles'] * last_frames_data['avg_vesicle_size'],
        'Tubes': last_frames_data['total_peptides_in_tubes'] * last_frames_data['avg_tube_size']
    }

    # Calculate mean scores over the frames
    mean_scores = {shape: scores.mean() for shape, scores in shape_scores.items()}

    # Normalize scores
    total_score = sum(mean_scores.values())
    normalized_scores = {shape: score/total_score if total_score > 0 else 0
                        for shape, score in mean_scores.items()}

    # Filter and sort shapes by score in descending order
    normalized_scores = {k: v for k, v in sorted(normalized_scores.items(),
                                               key=lambda item: item[1],
                                               reverse=True)
                        if v > 0.001}

    # Debug print
    print("\nNormalized shape scores (last 10 frames):")
    for shape, score in normalized_scores.items():
        print(f"{shape}: {score:.2f}")

    return normalized_scores

def categorize_by_final_shape(df, base_path):
    # Split data by AP type and process separately
    results_by_ap = {'high_ap': {}, 'mid_ap': {}}
    all_distributions = {}  # New dict to track all unique distributions

    for dataset in df['Dataset'].unique():
        # Check if dataset is in high_ap or mid_ap folder
        for ap_type in ['high_ap', 'mid_ap']:
            if os.path.exists(os.path.join(base_path, ap_type, dataset)):
                print(f"\nAnalyzing dataset: {dataset} (Type: {ap_type})")
                dataset_data = df[df['Dataset'] == dataset]

                # Get normalized shape scores
                shape_scores = get_detected_shapes(dataset_data)

                # Format scores as string with descending order
                shape_key = ', '.join([f"{shape}({score:.2f})"
                                     for shape, score in shape_scores.items()])

                # Store in AP-specific results
                if shape_key not in results_by_ap[ap_type]:
                    results_by_ap[ap_type][shape_key] = []
                results_by_ap[ap_type][shape_key].append(dataset)

                # Store in combined results
                if shape_key not in all_distributions:
                    all_distributions[shape_key] = {'high_ap': [], 'mid_ap': []}
                all_distributions[shape_key][ap_type].append(dataset)

                print(f"Final categorization: {dataset}: {shape_key}")
                break

    # Save separate results for each AP type
    for ap_type, distributions in results_by_ap.items():
        if distributions:
            results_data = []
            for shape_dist, datasets in distributions.items():
                results_data.append({
                    'Shape Distribution': shape_dist,
                    'Datasets': ', '.join(datasets),
                    'Count': len(datasets)
                })

            results_df = pd.DataFrame(results_data)
            results_df = results_df.sort_values('Count', ascending=False)
            output_file = f'shape_categories_{ap_type}.csv'
            results_df.to_csv(output_file, index=False)
            print(f"\nResults saved to {output_file}")

    # Save combined results with just the shape distributions and folder names
    if all_distributions:
        combined_data = []
        for shape_dist, ap_datasets in all_distributions.items():
            # Get the highest value from the shape distribution
            values = [float(val.split('(')[1].strip(')'))
                     for val in shape_dist.split(', ')]
            max_value = max(values)

            combined_data.append({
                'Shape Distribution': shape_dist,
                'High AP': ', '.join(ap_datasets['high_ap']) if ap_datasets['high_ap'] else '-',
                'Mid AP': ', '.join(ap_datasets['mid_ap']) if ap_datasets['mid_ap'] else '-',
                'Max Value': max_value  # Hidden column for sorting
            })

        combined_df = pd.DataFrame(combined_data)
        # Sort by highest value in descending order
        combined_df = combined_df.sort_values('Max Value', ascending=False)
        combined_df = combined_df.drop('Max Value', axis=1)
        combined_df.to_csv('shape_categories_combined.csv', index=False)
        print("\nResults saved to shape_categories_combined.csv")

if __name__ == "__main__":
    base_path = "centered_files"
    ap_types = ["high_ap", "mid_ap"]

    base_dirs = []
    for ap_type in ap_types:
        ap_dir = os.path.join(base_path, ap_type)
        if os.path.exists(ap_dir):
            peptide_dirs = [os.path.join(ap_dir, d) for d in os.listdir(ap_dir)
                          if os.path.isdir(os.path.join(ap_dir, d))]
            base_dirs.extend(peptide_dirs)

    if base_dirs:
        data = load_descriptors(base_dirs)
        if data is not None:
            categorize_by_final_shape(data, base_path)
    else:
        print("No valid directories found to process.")