# Trajectory Analysis & Figure Generation Scripts

This directory contains the main descriptor analysis entrypoints and post-processing figure generation scripts for the *Faraday Discussions* (2025) study.

## Descriptor Analysis Entrypoints

- `adi_analysis.py`: Aggregate Dynamics Index (ADI) runner script.
- `sfi_analysis.py`: Sheet Formation Index (SFI) runner script.
- `vfi_analysis.py`: Vesicle Formation Index (VFI) runner script.
- `tfi_analysis.py`: Tube Formation Index (TFI) runner script.
- `ffi_analysis.py`: Fiber Formation Index (FFI) runner script.
- `shape_tracker.py`: Integrative Multi-Descriptor Shape Tracker runner script.
- `centering.py`: Trajectory PBC centering and box wrapping script.

## Paper Figure Plotting Scripts

- `evolution_FF.py`, `evolution_RF.py`, `evolution_WI.py`: Compute structural evolution timelines for Diphenylalanine (FF), Arg-Phe (RF), and Trp-Ile (WI) systems.
- `dominant_FF.py`, `dominant_RF.py`, `dominant_WI.py`: Extract dominant aggregate shape distributions.
