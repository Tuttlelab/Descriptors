#!/usr/bin/env bash
#SBATCH --job-name=descriptors_csv
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

# ==============================================================================
# Single Descriptor Feature Extraction Job Script
# ==============================================================================

TOPOLOGY="${1:-data/topology.gro}"
TRAJECTORY="${2:-data/trajectory.xtc}"
OUTPUT_DIR="${3:-results/descriptors_csv}"

echo "Starting Descriptor Calculations..."
python adi_analysis.py -t "${TOPOLOGY}" -x "${TRAJECTORY}" -o "${OUTPUT_DIR}/adi"
python sfi_analysis.py -t "${TOPOLOGY}" -x "${TRAJECTORY}" -o "${OUTPUT_DIR}/sfi"
python vfi_analysis.py -t "${TOPOLOGY}" -x "${TRAJECTORY}" -o "${OUTPUT_DIR}/vfi"
python tfi_analysis.py -t "${TOPOLOGY}" -x "${TRAJECTORY}" -o "${OUTPUT_DIR}/tfi"
python ffi_analysis.py -t "${TOPOLOGY}" -x "${TRAJECTORY}" -o "${OUTPUT_DIR}/ffi"
echo "Descriptor Calculations finished."
