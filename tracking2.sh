#!/bin/bash

#======================================================
#
# Job script for running a parallel job on a single GPU node
#
#======================================================

#======================================================
# Propagate environment variables to the compute node
#SBATCH --export=ALL
#
# Run in the standard partition (queue)
#SBATCH --partition=standard
#
# Specify project account (replace as required)
#SBATCH --account=tuttle-rmss
#
# Request any GPU
#SBATCH --ntasks=1
#
# Specify (hard) runtime (HH:MM:SS)
#SBATCH --time=08:00:00
#
# Job name
#SBATCH --job-name=high_ap/WW_tfi
#
# Output file
#SBATCH --output=slurm-%j.out
#======================================================

module purge
module load nvidia/sdk/21.3
module load miniconda/3.11.4
module load gromacs/intel-2020.4/2020.7-single
module load vmd

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate peptide_analysis

#=========================================================
# Prologue script to record job details
# Do not change the line below
#=========================================================
/opt/software/scripts/job_prologue.sh
#----------------------------------------------------------

input_dir="centered_files/high_ap/WW"

cd "${input_dir}"

# Check if files exist
python3 ../../../tfi_analysis.py \
        --topology "centered_WW_1129_1323.gro" \
        --trajectory "centered_WW_1129_1323.xtc" \
        --output "tfi_results/"

#=========================================================
# Epilogue script to record job endtime and runtime
# Do not change the line below
#=========================================================
/opt/software/scripts/job_epilogue.sh
#----------------------------------------------------------

