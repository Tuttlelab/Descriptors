#!/usr/bin/env bash
#SBATCH --job-name=descriptors_centering
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

# ==============================================================================
# Centering & PBC Processing Job Script
# ==============================================================================

TOPOLOGY="${1:-data/topology.gro}"
TRAJECTORY="${2:-data/trajectory.xtc}"
OUTPUT_DIR="${3:-centered_files}"

echo "Starting PBC Centering..."
python centering.py -t "${TOPOLOGY}" -x "${TRAJECTORY}" -o "${OUTPUT_DIR}"
echo "Centering finished."
