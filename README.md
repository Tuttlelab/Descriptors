# Shape Descriptor Toolkit for Peptide Self-Assembly

[![Publication](https://img.shields.io/badge/Faraday--Discussions-2025-blue.svg)](https://pubs.rsc.org/en/content/articlelanding/2025/fd/d4fd00201f)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Official Python toolkit for calculating shape descriptors and tracking morphological transitions in peptide self-assembly simulations.

> **Publication Citation:**  
> Baskaran et al., *"Shape Descriptors for Peptide Self-Assembly"*, *Faraday Discussions*, 2025.  
> [DOI: 10.1039/D4FD00201F](https://doi.org/10.1039/D4FD00201F) | [Article Link](https://pubs.rsc.org/en/content/articlelanding/2025/fd/d4fd00201f)

---

## Quickstart (3 Steps)

### Step 1: Install Dependencies
```bash
git clone https://github.com/Tuttlelab/Descriptors.git
cd Descriptors
pip install -e .
```
*(Or create a Conda environment: `conda env create -f environment.yml && conda activate descriptors`)*

### Step 2: Run Analysis on Trajectories
Run any of the 5 shape descriptors or the integrative shape tracker from [`analysis/`](analysis/) on your Gromacs topology (`.gro`) and trajectory (`.xtc`):

```bash
# 1. Aggregate Dynamics Index (ADI)
python analysis/adi_analysis.py -t topology.gro -x trajectory.xtc -o results/adi

# 2. Sheet Formation Index (SFI)
python analysis/sfi_analysis.py -t topology.gro -x trajectory.xtc -o results/sfi

# 3. Vesicle Formation Index (VFI)
python analysis/vfi_analysis.py -t topology.gro -x trajectory.xtc -o results/vfi

# 4. Tube Formation Index (TFI)
python analysis/tfi_analysis.py -t topology.gro -x trajectory.xtc -o results/tfi

# 5. Fiber Formation Index (FFI)
python analysis/ffi_analysis.py -t topology.gro -x trajectory.xtc -o results/ffi

# 6. Integrative Shape Tracker (Runs all descriptors & tracks temporal transitions)
python analysis/shape_tracker.py -t topology.gro -x trajectory.xtc -o results/tracking
```

### Step 3: Reproduce Paper Plotting & Figures
Post-processing figure scripts from the paper are located in [`analysis/`](analysis/):
```bash
python analysis/evolution_FF.py
python analysis/evolution_RF.py
python analysis/evolution_WI.py
```

---

## Repository Structure

- [`descriptors/`](descriptors/): Core Python library (`descriptors.adi`, `descriptors.sfi`, etc.).
- [`analysis/`](analysis/): Descriptor entrypoint scripts (`adi_analysis.py`, `shape_tracker.py`) and paper figure plotting scripts.
- [`slurm/`](slurm/): Slurm HPC batch submission scripts (`centering_job.sh`, `tracking_job.sh`).
- [`tests/`](tests/): Test suite (`pytest tests/`).

---

## License & Citation

Licensed under the [MIT License](LICENSE). Please cite our paper when using this toolkit in your work:

```bibtex
@article{Baskaran2025Descriptors,
  title     = {Shape Descriptors for Peptide Self-Assembly},
  author    = {Baskaran, Raj Kumar Rajaram and Tuttle, Tell},
  journal   = {Faraday Discussions},
  year      = {2025},
  publisher = {Royal Society of Chemistry},
  doi       = {10.1039/D4FD00201F}
}
```