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
#SBATCH --time=00:10:00
#
# Job name
#SBATCH --job-name=csv_script
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

# Add logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

input_dir=$1
log "Starting processing for ${input_dir}"

# Change to input directory
cd "${input_dir}" || {
    log "ERROR: Failed to change to directory ${input_dir}"
    exit 1
}

# # Find first .gro and .xtc files
# log "Searching for .gro and .xtc files..."
# gro_file=$(ls -1 *.gro 2>/dev/null | head -1)
# xtc_file=$(ls -1 *.xtc 2>/dev/null | head -1)

# # Check if files exist
# if [ -n "$gro_file" ] && [ -n "$xtc_file" ]; then
#     log "Found files:"
#     log "  GRO: ${gro_file}"
#     log "  XTC: ${xtc_file}"

    log "Starting shape_tracker.py analysis..."
    # python3 ../../../analysis/csv_results.py
    # python3 ../../../analysis/decision.py
    python3 ../../../evolution.py

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log "Analysis completed successfully"
    else
        log "ERROR: Analysis failed with exit code ${PIPESTATUS[0]}"
        exit 1
    fi
# else
#     log "ERROR: Missing required files in ${input_dir}"
#     log "  GRO files found: $(ls -1 *.gro 2>/dev/null | wc -l)"
#     log "  XTC files found: $(ls -1 *.xtc 2>/dev/null | wc -l)"
#     exit 1
# fi

#=========================================================
# Epilogue script to record job endtime and runtime
# Do not change the line below
#=========================================================
/opt/software/scripts/job_epilogue.sh
#----------------------------------------------------------