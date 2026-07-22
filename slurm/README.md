# Slurm HPC Submission Scripts

This directory contains HPC batch submission scripts for executing trajectory preprocessing, descriptor computations, and integrative shape tracking on Slurm-managed clusters.

## Submission Scripts

1. `centering_script.sh`: Batch job for wrapping and PBC centering MD simulation trajectories.
2. `csv_script.sh`: Batch job for processing trajectory frames and exporting raw shape feature matrices to CSV.
3. `tracking_script.sh`: Batch job for executing the full multi-descriptor tracking pipeline across multiple peptide systems.

## Example Submission

```bash
sbatch slurm/centering_script.sh
sbatch slurm/tracking_script.sh
```
