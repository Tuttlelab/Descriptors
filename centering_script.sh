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
#SBATCH --time=2:00:00
#
# Job name
#SBATCH --job-name=centre_all
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

# Process single directory based on input parameters
category=$1
dipep_name=$2
base_dir=$3

# Create output directory
mkdir -p "centered_files/${category}"

dipep_dir="${base_dir}/${category}/${dipep_name}"
echo "Processing ${category}/${dipep_name}"

# Check if input files exist
if [ -f "${dipep_dir}/eq.gro" ] && [ -f "${dipep_dir}/eq.xtc" ]; then
    python3 centering.py \
        --dipep "${dipep_name}" \
        --topology "${dipep_dir}/eq.gro" \
        --trajectory "${dipep_dir}/eq.xtc" \
        --output "centered_files/${category}/${dipep_name}"
else
    echo "Warning: Missing input files for ${dipep_name}"
fi

#=========================================================
# Epilogue script to record job endtime and runtime
# Do not change the line below
#=========================================================
/opt/software/scripts/job_epilogue.sh
#----------------------------------------------------------

