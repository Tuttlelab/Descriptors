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

System-specific post-processing scripts used to generate timelines and dominant shape figures for specific peptide systems (FF, RF, WI) from the paper:

- `paper_figures/evolution_FF.py`, `evolution_RF.py`, `evolution_WI.py`: Structural evolution timelines.
- `paper_figures/dominant_FF.py`, `dominant_RF.py`, `dominant_WI.py`: Dominant shape distribution figures.
