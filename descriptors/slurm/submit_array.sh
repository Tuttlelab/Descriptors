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
#SBATCH --ntasks=40
#
# Specify (hard) runtime (HH:MM:SS)
#SBATCH --time=00:10:00
#
# Job name
#SBATCH --job-name=descriptor_array
#
# Output file
#SBATCH --output=%j.out
#======================================================

module purge

# Initialize micromamba
eval "$(micromamba shell hook --shell=bash)"
micromamba activate md_analysis

#=========================================================
# Prologue script to record job details
# Do not change the line below
#=========================================================
/opt/software/scripts/job_prologue.sh
#----------------------------------------------------------


# Use .gro and .xtc from the specified directory
input_dir="/users/mib23220/Documents/Descriptors/centered_files/high_ap/FF/"
topology=$(ls -1 ${input_dir}/*.gro | head -1)
trajectory=$(ls -1 ${input_dir}/*.xtc | head -1)
output_dir="/users/mib23220/Documents/Descriptors/descriptors/results/FF/"
mkdir -p "$output_dir"

# Frame index from SLURM_ARRAY_TASK_ID
frame_idx=${SLURM_ARRAY_TASK_ID}

echo "Processing frame $frame_idx"

# Run the per-frame pipeline (update path as needed)
python3 -m descriptors.pipeline.run_frame \
	--topology "$topology" \
	--trajectory "$trajectory" \
	--frame "$frame_idx" \
	--output "$output_dir"

#=========================================================
# Epilogue script to record job endtime and runtime
# Do not change the line below
#=========================================================
/opt/software/scripts/job_epilogue.sh
#----------------------------------------------------------
