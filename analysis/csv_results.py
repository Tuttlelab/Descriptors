import pandas as pd
import glob
import os
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

descriptors = ["adi", "sfi", "vfi", "tfi", "ffi"]

for descriptor_name in descriptors:
    descriptor_results_dir = f'{descriptor_name}_results/'
    pattern = f'{descriptor_name}_frame_results_*.csv'
    matching_files = sorted(glob.glob(os.path.join(descriptor_results_dir, pattern)))

    if not matching_files:
        print(f"Skipping {descriptor_name}: no matching files found in {descriptor_results_dir}")
        continue

    # Use most recent file
    frame_results_file = matching_files[-1]
    print(f"Loading {descriptor_name} results from {frame_results_file}")

    df = pd.read_csv(frame_results_file)
    df = df.drop('Peptides', axis=1)

    # Identify the last column
    last_column = df.columns[-1]

    # Round the last column values to one decimal
    df[last_column] = df[last_column].round(1)

    df.to_csv(f'{descriptor_name}_output.csv', index=False)

    # Generate a line plot with all columns
    # Apply Savitzky-Golay filter to smooth the data in column 4 with a larger window length
    smoothed_column = savgol_filter(df.iloc[:, 3], window_length=251, polyorder=3)

    # Plot the original data in column 2
    plt.plot(df.index, df.iloc[:, 1], label=f'Original {df.columns[1]}')

    # Plot the smoothed data in column 4
    plt.plot(df.index, smoothed_column, label=f'Smoothed {df.columns[3]}')

    plt.title(f'{descriptor_name} Results')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend(title='Columns')

    # Save the plot to a file
    plot_filename = f'{descriptor_name}_results_plot.png'
    plt.savefig(plot_filename)
    plt.clf()  # Clear the plot for the next iteration

    print(f"Plot saved as {plot_filename}")

    # Parse the data
    # Add your parsing logic here
