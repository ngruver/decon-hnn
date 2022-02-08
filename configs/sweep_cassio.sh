#!/usr/bin/env bash

#SBATCH --job-name=GFP-REGRESSION-NB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_12gb
#SBATCH --exclude=vine3,loopy8

set -e

if [[ -z "${WANDB_SWEEP_ID}" ]]; then
  echo "Missing WANDB_SWEEP_ID"
  exit 1
fi

source "${HOME}/.slurm_bash_profile"

export WANDB_MODE=run
export WANDB_DIR="${LOGDIR}"
export WANDB_PROJECT="gfp-regression-nb"
export WANDB_NAME="${SLURM_JOB_NAME}--${SLURM_JOB_ID}"

cd "${WORKDIR}/gfp-bayesopt"

export PYTHONPATH="$(pwd):${PYTHONPATH}"

# source $(conda info --base)/bin/deactivate
# source $(conda info --base)/bin/activate uq-playground

wandb agent --count=1 ${WANDB_SWEEP_ID}
