import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def process_evolution_data(evolution_df):
    """Convert transition data to population fractions"""
    # Get all unique frames
    all_frames = range(int(evolution_df['frame'].min()), int(evolution_df['frame'].max()) + 1)

    # Initialize processed dataframe with all frames
    processed_df = pd.DataFrame(index=all_frames)
    processed_df.index.name = 'frame'

    # Initialize all structure columns with zeros
    structures = ['sheets', 'fibers', 'vesicles', 'tubes', 'unstructured']
    for struct in structures:
        processed_df[struct] = 0.0

    # Set initial state as unstructured
    processed_df['unstructured'] = 1.0

    # Process each transition
    for _, row in evolution_df.iterrows():
        frame = int(row['frame'])
        from_state = row['from_state']
        to_state = row['to_state']

        if from_state in structures:
            processed_df.loc[frame:, from_state] -= 1.0
        if to_state in structures:
            processed_df.loc[frame:, to_state] += 1.0

        # Adjust unstructured fraction
        if from_state == 'undetermined':
            processed_df.loc[frame:, 'unstructured'] -= 1.0
        if to_state == 'undetermined':
            processed_df.loc[frame:, 'unstructured'] += 1.0

    # Normalize fractions to be between 0 and 1
    processed_df = processed_df.clip(0, 1)

    # Add time column
    processed_df['time'] = processed_df.index / 8

    # Improve smoothing with better window size and method
    window = 24  # 3ns window (8 frames per ns)
    for structure in structures:
        # Use rolling mean and handle NaN values properly
        smoothed = processed_df[structure].rolling(
            window=window, center=True, min_periods=1).mean()
        processed_df[f'{structure}_smooth'] = smoothed.bfill().ffill()

    # Ensure sum of fractions equals 1
    total = sum(processed_df[f'{structure}_smooth'] for structure in structures)
    for structure in structures:
        processed_df[f'{structure}_smooth'] = processed_df[f'{structure}_smooth'] / total

    return processed_df

def load_data():
    """Load and process both CSV files"""
    evolution_file = 'WI/evolution_0121_1024.csv'
    dominant_file = 'WI/dominant_0121_1027.csv'

    if not (os.path.exists(evolution_file) and os.path.exists(dominant_file)):
        raise FileNotFoundError("CSV files not found")

    # Load raw data with error checking
    try:
        evolution_df = pd.read_csv(evolution_file)
        dominant_df = pd.read_csv(dominant_file)

        # Validate required columns
        required_evolution_cols = ['frame', 'from_state', 'to_state']
        required_dominant_cols = ['Frame', 'shapes']

        if not all(col in evolution_df.columns for col in required_evolution_cols):
            raise ValueError(f"Missing required columns in evolution file: {required_evolution_cols}")
        if not all(col in dominant_df.columns for col in required_dominant_cols):
            raise ValueError(f"Missing required columns in dominant file: {required_dominant_cols}")

    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

    # Store transitions before processing
    transitions = []
    for _, row in evolution_df.iterrows():
        transitions.append({
            'frame': row['frame'],
            'from_state': row['from_state'],
            'to_state': row['to_state']
        })

    # Process evolution data
    processed_evolution = process_evolution_data(evolution_df)

    return processed_evolution, dominant_df, transitions

def create_combined_plot(evolution_df, dominant_df, transitions):
    """Create combined plot with smoothed lines"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), height_ratios=[1.2, 1])
    plt.rcParams.update({'font.size': 16})

    # Top plot with smoothed lines
    colors = sns.color_palette("husl", 4)
    structures = ['sheets', 'fibers', 'vesicles', 'tubes']
    labels = [s.capitalize() for s in structures]

    for i, structure in enumerate(structures):
        ax1.plot(evolution_df['time'],
                evolution_df[f'{structure}_smooth'],  # Use smoothed data
                label=labels[i],
                color=colors[i],
                linewidth=2)

    # Improved transition markers
    for t in transitions:
        if t['frame'] > 200:  # Skip transient period
            time = t['frame']/8
            if 0 <= time <= 1500:  # Only show transitions within plot range
                ax1.axvline(x=time, color='gray', linestyle='--', alpha=0.3)

    # Mark transient period
    ax1.axvspan(0, 25, color='gray', alpha=0.1, label='Transient')

    # Style top plot
    ax1.set_ylabel('Structure Population Fraction')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)
    ax1.grid(False)
    ax1.set_ylim(bottom=0)

    # Bottom plot - Dominant States
    ordered_shapes = ['undetermined', 'sheet', 'vesicle', 'fiber', 'tube']
    unique_shapes = [shape for shape in ordered_shapes
                    if shape in dominant_df['shapes'].unique()]
    shape_map = {shape: idx for idx, shape in enumerate(unique_shapes)}

    ax2.scatter(dominant_df['Frame']/8,
                dominant_df['shapes'].map(shape_map),
                c='black', alpha=0.5, s=5)

    # Style bottom plot
    ax2.set_yticks(range(len(unique_shapes)))
    ax2.set_yticklabels(unique_shapes)
    ax2.set_ylabel('Dominant Shape')
    ax2.grid(False)

    # Shared styling for both plots
    for ax in [ax1, ax2]:
        ax.set_xlim(0, 1500)
        ax.set_xticks([0, 250, 500, 750, 1000, 1250, 1500])
    ax2.set_xlabel('Time (ns)')

    plt.tight_layout()
    timestamp = datetime.now().strftime("%m%d_%H%M")
    plt.savefig(f'WI/combined_{timestamp}.png',
                dpi=600, bbox_inches='tight')
    plt.close()

def main():
    print("\n=== Creating Combined Plot ===")
    evolution_df = None  # Initialize variables for error handling

    try:
        evolution_df, dominant_df, transitions = load_data()
        create_combined_plot(evolution_df, dominant_df, transitions)
        print("Combined plot saved successfully")
    except Exception as e:
        print(f"Error: {e}")
        if evolution_df is not None:
            # Debug info only if data was loaded
            print("\nEvolution DataFrame columns:", evolution_df.columns.tolist())
            print("Evolution DataFrame head:\n", evolution_df.head())

if __name__ == "__main__":
    main()