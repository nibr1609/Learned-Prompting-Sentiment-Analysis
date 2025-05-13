#!/bin/bash
#SBATCH --chdir .
#SBATCH --account=cil_jobs
#SBATCH --time=24:00:00
#SBATCH -o /home/%u/logs/slurm_output__%x-%j.out
#SBATCH --error=/home/%u/logs/slurm_err__%x-%j.err
#SBATCH --mail-type=FAIL

set -e
set -o xtrace
echo PWD:$(pwd)
echo STARTING AT $(date)

# Environment
source /cluster/courses/cil/envs/miniforge3/etc/profile.d/conda.sh
conda activate cil

cd src/models

export DGGML_CUDA=ON
python3 prompt_evaluator.py

echo "Done."
echo FINISHED at $(date)
