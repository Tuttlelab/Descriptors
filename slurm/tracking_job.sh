#!/usr/bin/env bash
#SBATCH --job-name=shape_tracker
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00

# ==============================================================================
# Shape Tracking Job Script
# ==============================================================================

TOPOLOGY="${1:-data/topology.gro}"
TRAJECTORY="${2:-data/trajectory.xtc}"
OUTPUT_DIR="${3:-results/tracking_results}"

echo "Starting Shape Tracking Analysis..."
python shape_tracker.py -t "${TOPOLOGY}" -x "${TRAJECTORY}" -o "${OUTPUT_DIR}"
echo "Shape Tracking finished."
