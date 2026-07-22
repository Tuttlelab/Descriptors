# Shape Descriptor Toolkit for Peptide Self-Assembly

[![Publication](https://img.shields.io/badge/Faraday--Discussions-2025-blue.svg)](https://pubs.rsc.org/en/content/articlelanding/2025/fd/d4fd00201f)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

An open-source Python toolkit for analyzing, quantifying, and tracking morphological transformations in peptide self-assembly molecular dynamics simulations.

> **Publication Citation:**  
> Baskaran et al., *"Shape Descriptors for Peptide Self-Assembly"*, *Faraday Discussions*, 2025.  
> [Read Paper on Royal Society of Chemistry](https://pubs.rsc.org/en/content/articlelanding/2025/fd/d4fd00201f) | **DOI:** [10.1039/D4FD00201F](https://doi.org/10.1039/D4FD00201F)

---

## Overview

Peptide self-assembly leads to diverse supramolecular architectures such as oligomeric aggregates, sheets, hollow vesicles, nanotubes, and fibrillar networks. The **Shape Descriptor Toolkit** provides mathematically rigorous, automated metrics to classify and track these morphological transitions directly from Gromacs trajectory files (`.gro`, `.xtc`, `.pdb`).

---

## Available Descriptors

| Descriptor | Full Name | Target Morphology | Key Metric / Method |
| :--- | :--- | :--- | :--- |
| **ADI** | Aggregate Dynamics Index | General Aggregates | RDF adaptive cutoffs & contact persistence |
| **SFI** | Sheet Formation Index | Planar $\beta$-Sheets | Planarity RMSD & orientational alignment |
| **VFI** | Vesicle Formation Index | Hollow Vesicles | Convex hull sphericity & radial density void check |
| **TFI** | Tube Formation Index | Cylindrical Tubes | Gyration tensor asphericity & radial uniformity |
| **FFI** | Fiber Formation Index | Fibrillar Networks | Elongation ratio & Fibrillar Order Parameter (FOP) |
| **Tracker** | Multi-Descriptor Tracker | Morphological Transitions | Integrated multi-descriptor temporal tracking |

---

## Installation

### Option 1: Conda Environment (Recommended)

```bash
git clone https://github.com/Tuttlelab/Descriptors.git
cd Descriptors
conda env create -f environment.yml
conda activate descriptors
pip install -e .
```

### Option 2: Pip / Virtual Environment

```bash
pip install -r requirements.txt
pip install -e .
```

---

## Quickstart & Usage

### 1. Command Line Interface (CLI)

Run individual descriptor analyses on your simulation topology (`.gro`) and trajectory (`.xtc`):

```bash
# Aggregate Dynamics Index (ADI)
descriptors-adi -t data/topology.gro -x data/trajectory.xtc -o results/adi_results

# Sheet Formation Index (SFI)
descriptors-sfi -t data/topology.gro -x data/trajectory.xtc -o results/sfi_results

# Vesicle Formation Index (VFI)
descriptors-vfi -t data/topology.gro -x data/trajectory.xtc -o results/vfi_results

# Tube Formation Index (TFI)
descriptors-tfi -t data/topology.gro -x data/trajectory.xtc -o results/tfi_results

# Fiber Formation Index (FFI)
descriptors-ffi -t data/topology.gro -x data/trajectory.xtc -o results/ffi_results
```

### 2. Integrative Multi-Descriptor Tracking

Track structural evolution and morphological transitions over the entire trajectory:

```bash
descriptors-tracker -t data/topology.gro -x data/trajectory.xtc -o results/tracking_results
```

Alternatively, run using the standard Python script entry point:

```bash
python shape_tracker.py -t data/topology.gro -x data/trajectory.xtc -o results/tracking_results
```

### 3. Programmatic Python API

You can also import and execute descriptors directly within Python scripts or Jupyter Notebooks:

```python
from descriptors import calculate_adi, calculate_sfi, track_shapes

# Run ADI analysis programmatically
adi_results = calculate_adi(
    topology="data/topology.gro",
    trajectory="data/trajectory.xtc",
    output_dir="results/adi_results"
)

# Run full multi-descriptor tracking
track_shapes(
    topology="data/topology.gro",
    trajectory="data/trajectory.xtc",
    output_dir="results/tracking_results"
)
```

---

## Repository Structure

```
Descriptors/
├── README.md                      # Documentation and publication reference
├── pyproject.toml                 # Package configuration & CLI scripts
├── environment.yml                # Conda environment definition
├── requirements.txt               # Dependencies list
├── descriptors/                   # Core Python library
│   ├── __init__.py                # Package exports & metadata
│   ├── adi.py                     # Aggregate Dynamics Index
│   ├── sfi.py                     # Sheet Formation Index
│   ├── vfi.py                     # Vesicle Formation Index
│   ├── tfi.py                     # Tube Formation Index
│   ├── ffi.py                     # Fiber Formation Index
│   ├── tracker.py                 # Integrative Shape Tracker
│   ├── centering.py               # PBC centering tools
│   └── utils.py                   # Shared utilities & logging
├── scripts/                       # Command-line runner scripts
├── examples/                      # Automated pipeline runner
│   └── run_full_pipeline.sh
├── slurm/                         # HPC batch submission templates
├── analysis/                      # Post-processing & figure generation
└── tests/                         # Unit tests suite
```

---

## Testing

Run unit tests to verify mathematical descriptor implementations:

```bash
pytest tests/
```

---

## Citation

If you use this toolkit in your research, please cite our paper:

```bibtex
@article{Baskaran2025Descriptors,
  title     = {Shape Descriptors for Peptide Self-Assembly},
  author    = {Baskaran, Raj Kumar Rajaram and Tuttle, Tell},
  journal   = {Faraday Discussions},
  year      = {2025},
  publisher = {Royal Society of Chemistry},
  doi       = {10.1039/D4FD00201F},
  url       = {https://pubs.rsc.org/en/content/articlelanding/2025/fd/d4fd00201f}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).