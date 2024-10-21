# Shape Descriptor Toolkit

This toolkit provides Python scripts for analyzing peptide self-assembly simulations.

## Scripts

    adi_analysis.py: Aggregate Dynamics Index (ADI)
    sfi_analysis.py: Sheet Formation Index (SFI)
    vfi_analysis.py: Vesicle Formation Index (VFI)
    tfi_analysis.py: Tube Formation Index (TFI)
    ffi_analysis.py: Fiber Formation Index (FFI)
    shape_tracker.py: Tracks structural changes over time

## Installation
### Using Conda

    conda env create -f environment.yml
    conda activate descriptors

### Using pip

    pip install -r requirements.txt

## Usage

Place your topology and trajectory files in the data/ directory.

### Run Analysis Scripts

    python adi_analysis.py -t data/your_topology.gro -x data/your_trajectory.xtc -s 'resname PEP' -o results/adi_results
    python sfi_analysis.py -t data/your_topology.gro -x data/your_trajectory.xtc -s 'resname PEP' -o results/sfi_results
    python vfi_analysis.py -t data/your_topology.gro -x data/your_trajectory.xtc -s 'resname PEP' -o results/vfi_results
    python tfi_analysis.py -t data/your_topology.gro -x data/your_trajectory.xtc -s 'resname PEP' -o results/tfi_results
    python ffi_analysis.py -t data/your_topology.gro -x data/your_trajectory.xtc -s 'resname PEP' -o results/ffi_results

### Run Tracking Script

    python shape_tracker.py \
    -adi results/adi_results \
    -sfi results/sfi_results \
    -vfi results/vfi_results \
    -tfi results/tfi_results \
    -ffi results/ffi_results \
    -o results/tracking_results

## Requirements

Install the required Python packages:

    numpy
    scipy
    pandas
    matplotlib
    MDAnalysis
    scikit-image

These are listed in requirements.txt and environment.yml.