#!/bin/bash
#SBATCH --chdir .
#SBATCH --account cil_cpu
#SBATCH --time=01:00:00
#SBATCH -o /home/%u/logs/slurm_output__%x-%j.out
#SBATCH --error=/home/%u/logs/slurm_err__%x-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mem-per-cpu=3000

set -e
set -o xtrace
echo PWD:$(pwd)
echo STARTING AT $(date)

# Arguments
USER_ARG=$1
CONFIG_ARG=$2

echo "Running as $USER_ARG with config $CONFIG_ARG"

# Environment
source /cluster/courses/cil/envs/miniforge3/etc/profile.d/conda.sh
conda activate /home/"$USER_ARG"/cil

cd src
python3 run_experiment.py -c "$CONFIG_ARG"

echo "Done."
echo FINISHED at $(date)
