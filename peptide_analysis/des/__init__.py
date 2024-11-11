# des/__init__.py

"""
Descriptor (des) Module for Peptide Analysis

This module contains scripts for calculating various structural formation indices in peptide simulations.
Each script performs a specific type of analysis and can be executed independently.

Scripts Included:

- **adi.py**: Aggregate Detection Index (ADI) Analysis
  - Detects and characterizes aggregates in the simulation.
  - Tracks persistent aggregates over time.

- **ffi.py**: Fiber Formation Index (FFI) Analysis
  - Identifies and analyzes fiber-like structures.
  - Computes shape descriptors and alignment metrics.

- **sfi.py**: Sheet Formation Index (SFI) Analysis
  - Detects beta-sheet formations.
  - Analyzes sheet size, orientation, and persistence.

- **tfi.py**: Tube Formation Index (TFI) Analysis
  - Identifies tubular structures.
  - Analyzes tube morphology and persistence.

- **vfi.py**: Vesicle Formation Index (VFI) Analysis
  - Detects vesicular structures.
  - Analyzes vesicle size, shape, and hollowness.

Usage:

Each script can be run from the command line with the `-h` flag to display available options and arguments.

Example:
    python adi.py -h

Dependencies:

- Python 3.x
- MDAnalysis
- NumPy
- SciPy
- Matplotlib
- scikit-learn

Note:

Ensure that the `util` package is accessible in your Python path, as the scripts in this module depend on utility functions provided there.

"""
