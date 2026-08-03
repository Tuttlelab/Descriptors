# Descriptor Analysis & Figure Generation

This directory contains the main generic descriptor analysis tools and paper figure generation scripts for the *Faraday Discussions* (2025) study.

## Generic Descriptor Analysis Entrypoints

These generic tools analyze any peptide MD simulation topology (`.gro`) and trajectory (`.xtc`):

- `shape_tracker.py`: Integrative Multi-Descriptor Shape Tracker (runs all descriptors & tracks transitions over time).
- `adi_analysis.py`: Aggregate Dynamics Index (ADI) runner script.
- `sfi_analysis.py`: Sheet Formation Index (SFI) runner script.
- `vfi_analysis.py`: Vesicle Formation Index (VFI) runner script.
- `tfi_analysis.py`: Tube Formation Index (TFI) runner script.
- `ffi_analysis.py`: Fiber Formation Index (FFI) runner script.
- `centering.py`: Trajectory PBC centering and box wrapping script.

## Paper Figure Plotting Scripts ([`paper_figures/`](paper_figures/))

Unified post-processing scripts used to generate timelines and dominant shape figures for any peptide system (e.g. FF, RF, WI) from the paper:

- `paper_figures/plot_evolution.py`: Structural evolution timelines (`python analysis/paper_figures/plot_evolution.py --system FF`).
- `paper_figures/plot_dominant.py`: Dominant shape distribution figures (`python analysis/paper_figures/plot_dominant.py --system FF`).

