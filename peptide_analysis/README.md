# Peptide Analysis Toolkit

This toolkit provides Python scripts for analyzing peptide self-assembly simulations. It includes modules for calculating various structural formation indices and tracking the evolution of shapes over time.

## Project Structure

```
peptide_analysis/
├── __init__.py
├── des/
│   ├── __init__.py
│   ├── adi.py
│   ├── ffi.py
│   ├── sfi.py
│   ├── tfi.py
│   └── vfi.py
├── util/
│   ├── __init__.py
│   ├── io.py
│   ├── geometry.py
│   ├── clustering.py
│   ├── data.py
│   ├── visualization.py
│   └── logging.py
├── shape_tracker.py
├── environment.yml
├── requirements.txt
├── .gitignore
├── README.md
```

## Scripts

The toolkit includes the following scripts:

- `des/adi.py`: Aggregate Detection Index (ADI) Analysis
  - Detects and characterizes aggregates in the simulation.

- `des/ffi.py`: Fiber Formation Index (FFI) Analysis
  - Identifies and analyzes fiber-like structures.

- `des/sfi.py`: Sheet Formation Index (SFI) Analysis
  - Detects beta-sheet formations.

- `des/tfi.py`: Tube Formation Index (TFI) Analysis
  - Identifies tubular structures.

- `des/vfi.py`: Vesicle Formation Index (VFI) Analysis
  - Detects vesicular structures.

- `shape_tracker.py`: Tracks structural changes over time using the descriptors.

## Installation

### Using Conda

Create a Conda environment and install the dependencies:

```bash
conda env create -f environment.yml
conda activate peptide_analysis
```

### Using pip

Alternatively, you can use pip to install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Place your topology and trajectory files in a `data/` directory or specify the paths to your files.

### Run Analysis Scripts

Each analysis script can be run from the command line. For example:

```bash
python des/adi.py -t data/your_topology.gro -x data/your_trajectory.xtc -s 'resname PEP' -o results/adi_results
```

```bash
python des/sfi.py -t data/your_topology.gro -x data/your_trajectory.xtc -s 'resname PEP' -o results/sfi_results
```

```bash
python des/vfi.py -t data/your_topology.gro -x data/your_trajectory.xtc -s 'resname PEP' -o results/vfi_results
```

```bash
python des/tfi.py -t data/your_topology.gro -x data/your_trajectory.xtc -s 'resname PEP' -o results/tfi_results
```

```bash
python des/ffi.py -t data/your_topology.gro -x data/your_trajectory.xtc -s 'resname PEP' -o results/ffi_results
```

### Run Shape Tracker Script

The `shape_tracker.py` script tracks the evolution of shapes over time by utilizing the analysis modules.

```bash
python shape_tracker.py -t data/your_topology.gro -x data/your_trajectory.xtc -s 'resname PEP' -o results/shape_tracker_results
```

### Common Arguments

- `-t`, `--topology`: Path to the topology file (e.g., `.gro` file).
- `-x`, `--trajectory`: Path to the trajectory file (e.g., `.xtc` file).
- `-s`, `--selection`: Atom selection string for MDAnalysis (e.g., `'resname PEP'`).
- `-o`, `--output`: Output directory to save results.

Each script may have additional arguments specific to the analysis. Use the `-h` or `--help` flag to see all available options:

```bash
python des/adi.py -h
```

### Example

To run the ADI analysis:

```bash
python des/adi.py \
  -t data/topology.gro \
  -x data/trajectory.xtc \
  -s 'resname PEP' \
  -o results/adi_results \
  --cutoff 6.0 \
  --persistence 5
```

## Shape Classification Categories

The shape tracker classifies aggregates into the following categories:

- **Spherical Aggregates (ADI)**
  - Micelles
  - Vesicles (VFI)
  - Spheres

- **Cylindrical Aggregates**
  - Fibers (FFI)
  - Tubes (TFI)
  - Helices

- **Planar Aggregates**
  - Bilayers
  - Sheets (SFI)
  - Disks
  - Ribbons

- **Irregular Aggregates**
  - Amorphous

- **Non-Aggregates**

## High-Throughput Screening

The toolkit is designed to handle high-throughput screening of multiple simulations without the need for manual visualization. By automating the shape classification, you can efficiently process and analyze large datasets.

## Requirements

Install the required Python packages:

- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `MDAnalysis`
- `scikit-learn`
- `seaborn` (optional, for advanced visualization)
- `networkx` (optional, if used in clustering algorithms)

These are listed in `requirements.txt` and `environment.yml`.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

Specify your project's license here (e.g., MIT License).

## Author

Your Name or Organization

