import pandas as pd
import numpy as np
import os
import sys

# filepath: load_csv_head_tail.py
def load_csv_preview():
    # Define the file path
    csv_path = 'test_folder/high_ap/FF/ffi_results/ffi_frame_results_1130_1522.csv'

    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Remove Peptides column
        df = df.drop('Peptides', axis=1)

        # Print the total number of rows
        print(f"\nTotal number of rows in CSV: {len(df)}\n")

        # Display first 5 rows
        print("First 5 rows of the CSV:")
        print(df.head())

        print("\n" + "="*50 + "\n")

        # Display last 5 rows
        print("Last 5 rows of the CSV:")
        print(df.tail())

    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")

if __name__ == "__main__":
    load_csv_preview()