# Descriptors Project

Modular pipeline for per-frame, per-cluster descriptor extraction and feature engineering.

## Structure
- `core/`: Stateless, reusable logic for I/O, PBC, clustering, descriptors, features
- `pipeline/`: Orchestration and workflow
- `configs/`: User-tunable parameters
- `slurm/`: HPC job scripts
- `tests/`: Unit tests for each module

## Usage
See `cli.py` and `README.md` for getting started.
