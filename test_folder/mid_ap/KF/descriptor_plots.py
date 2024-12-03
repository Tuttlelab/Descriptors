import pandas as pd
import matplotlib.pyplot as plt
import glob
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# Set style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

# Descriptor configurations
descriptors = {
    'adi': {
        'file': 'adi_output.csv',
        'y1_column': 'total_peptides_in_aggregate',
        'y2_column': 'aggregate_count',
        'y1_label': 'Peptides in aggregates\n(normalised)',
        'y2_label': 'Number of aggregates'
    },
    'sfi': {
        'file': 'sfi_output.csv',
        'y1_column': 'total_peptides_in_sheets',
        'y2_column': 'sheet_count',
        'y1_label': 'Peptides in sheets\n(normalised)',
        'y2_label': 'Number of sheets'
    },
    'vfi': {
        'file': 'vfi_output.csv',
        'y1_column': 'total_peptides_in_vesicles',
        'y2_column': 'vesicle_count',
        'y1_label': 'Peptides in vesicles\n(normalised)',
        'y2_label': 'Number of vesicles'
    },
    'tfi': {
        'file': 'tfi_output.csv',
        'y1_column': 'total_peptides_in_tubes',
        'y2_column': 'tube_count',
        'y1_label': 'Peptides in tubes\n(normalised)',
        'y2_label': 'Number of tubes'
    },
    'ffi': {
        'file': 'ffi_output.csv',
        'y1_column': 'total_peptides_in_fibers',
        'y2_column': 'fiber_count',
        'y1_label': 'Peptides in fibers\n(normalised)',
        'y2_label': 'Number of fibers'
    }
}

def create_descriptor_plot(descriptor_name, config):
    # Read CSV data
    df = pd.read_csv(config['file'])

    # Get total peptides from gro file if not already normalized
    if max(df[config['y1_column']]) > 1:
        gro_file = glob.glob('*.gro')[0]
        with open(gro_file, 'r') as f:
            next(f)
            total_peptides = int(f.readline().strip())
        df['normalized_y1'] = df[config['y1_column']] / total_peptides
    else:
        df['normalized_y1'] = df[config['y1_column']]

    # Create figure
    fig, ax1 = plt.subplots(figsize=(3.5, 2.1))

    # Debug print
    print(f"{descriptor_name} max normalized value: {df['normalized_y1'].max():.2f}")

    # Primary axis
    line1 = ax1.plot(df['Frame'], df['normalized_y1'],
                     color='#1f77b4', linewidth=0.75)
    ax1.set_xlabel('Frame', fontsize=7)
    ax1.set_ylabel(config['y1_label'], fontsize=7, color='#1f77b4')
    ax1.tick_params(axis='both', labelsize=6)
    ax1.tick_params(axis='y', labelcolor='#1f77b4')

    # Format y1 ticks to 2 decimal places
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Set axes limits with proportional offset
    x_max = max(df['Frame'])
    y1_max = df['normalized_y1'].max()

    # Round y1_max up to next 0.1 and ensure it's above max value
    y1_max = 0.1 * (int(y1_max * 10 + 1))  # Changed rounding to ensure it's above max

    ax1.set_xlim(-0.05 * x_max, x_max * 1.05)
    ax1.set_ylim(-0.05 * y1_max, y1_max * 1.05)

    ax1.yaxis.set_major_locator(MultipleLocator(y1_max/5))

    # Secondary axis
    ax2 = ax1.twinx()
    line2 = ax2.plot(df['Frame'], df[config['y2_column']],
                     color='#ff7f0e', linewidth=0.75, alpha=0.7)  # Added transparency
    ax2.set_ylabel(config['y2_label'], fontsize=7, color='#ff7f0e')
    ax2.tick_params(axis='y', labelsize=6, labelcolor='#ff7f0e')

    # Set secondary y-axis limits with offset
    max_count = df[config['y2_column']].max()
    magnitude = 10 ** (len(str(int(max_count))) - 1)  # Get the order of magnitude
    y2_max = magnitude * (max_count // magnitude + 1)  # Round up to nearest magnitude
    ax2.set_ylim(-0.05 * y2_max, y2_max * 1.05)

    ax2.yaxis.set_major_locator(MultipleLocator(y2_max/5))

    plt.tight_layout()
    plt.savefig(f'{descriptor_name}_plot.pdf',
                dpi=1600, bbox_inches='tight', format='pdf')
    plt.close()

# Generate plots for all descriptors
for descriptor_name, config in descriptors.items():
    if glob.glob(config['file']):  # Only create plot if file exists
        create_descriptor_plot(descriptor_name, config)