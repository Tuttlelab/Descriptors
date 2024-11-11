# __init__.py

"""
Peptide Analysis Package

This package provides tools for analyzing peptide simulations, focusing on the detection and characterization of various structural formations such as aggregates, fibers, sheets, tubes, and vesicles.

Package Structure:

- **des/**: Descriptor modules containing analysis scripts for different structural indices.
  - `adi.py`: Aggregate Detection Index analysis.
  - `ffi.py`: Fiber Formation Index analysis.
  - `sfi.py`: Sheet Formation Index analysis.
  - `tfi.py`: Tube Formation Index analysis.
  - `vfi.py`: Vesicle Formation Index analysis.

- **util/**: Utility modules providing common functions used across analysis scripts.
  - `io.py`: Input/output operations, argument parsing, trajectory loading.
  - `geometry.py`: Geometric computations and shape analysis.
  - `clustering.py`: Clustering algorithms and related functions.
  - `data.py`: Data manipulation and saving utilities.
  - `visualization.py`: Plotting functions for analysis results.
  - `logging.py`: Logging setup and configuration.

Usage:

Import the necessary modules and functions as needed in your analysis scripts or run the provided scripts in the `des` directory directly.

Example:
    python des/adi.py -t topology.gro -x trajectory.xtc -o results/adi

Dependencies:

- Python 3.x
- MDAnalysis
- NumPy
- SciPy
- Matplotlib
- scikit-learn
- Other dependencies as specified in individual modules.

License:

Specify your project's license here (e.g., MIT License).

Author:

Your Name or Organization

"""
