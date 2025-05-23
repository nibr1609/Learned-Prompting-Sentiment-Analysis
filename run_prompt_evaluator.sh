#!/bin/bash
#SBATCH --chdir .
#SBATCH --account=cil_jobs
#SBATCH --time=24:00:00
#SBATCH --job-name=prompt#####
#SBATCH -o /home/%u/logs/prompt_output__%x-%j.out
#SBATCH -e /home/%u/logs/prompt_error__%x-%j.err
#SBATCH --mail-type=FAIL

set -e
set -o xtrace

echo PWD:$(pwd)
echo STARTING at $(date)

# Load Conda environment
source /cluster/courses/cil/envs/miniforge3/etc/profile.d/conda.sh
conda activate cil

# Navigate and run script
cd src/models
python3 prompt_evaluator.py

echo FINISHED at $(date)