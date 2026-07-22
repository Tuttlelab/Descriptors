#!/bin/bash

#======================================================
# Job script for running a single frame descriptor job
#======================================================

#SBATCH --export=ALL
#SBATCH --partition=dev
#SBATCH --account=tuttle-rmss
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --job-name=desc_test
#SBATCH --output=desc_test_%j.out

module purge

# Initialize micromamba
eval "$(micromamba shell hook --shell=bash)"
micromamba activate md_analysis


#=========================================================
# Prologue script to record job details
#=========================================================
/opt/software/scripts/job_prologue.sh
#----------------------------------------------------------

# Ensure Python can find the descriptors package
export PYTHONPATH=/users/mib23220/Documents/Descriptors:$PYTHONPATH

input_dir="/users/mib23220/Documents/Descriptors/centered_files/high_ap/FF/"
topology=$(ls -1 ${input_dir}/*.gro | head -1)
trajectory=$(ls -1 ${input_dir}/*.xtc | head -1)
output_dir="/users/mib23220/Documents/Descriptors/descriptors/results/FF/"
mkdir -p "$output_dir"

frame_idx=5000

echo "Processing frame $frame_idx"

python3 -m descriptors.pipeline.run_frame \
    --topology "$topology" \
    --trajectory "$trajectory" \
    --frame "$frame_idx" \
    --output "$output_dir"

#=========================================================
# Epilogue script to record job endtime and runtime
#=========================================================
/opt/software/scripts/job_epilogue.sh
#----------------------------------------------------------
