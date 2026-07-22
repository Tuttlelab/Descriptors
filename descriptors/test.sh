#!/bin/bash

#======================================================
# Job script for running a single frame descriptor job
#======================================================

#SBATCH --export=ALL
#SBATCH --partition=dev
#SBATCH --account=tuttle-rmss
#SBATCH --ntasks=40
#SBATCH --time=00:30:00
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

cd /users/mib23220/Documents/Descriptors
PYTHONPATH=$(pwd) pytest descriptors/tests/test_adi.py -v

#=========================================================
# Epilogue script to record job endtime and runtime
#=========================================================
/opt/software/scripts/job_epilogue.sh
#----------------------------------------------------------
