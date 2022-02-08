#!/usr/bin/env bash

#SBATCH --job-name=GFP-REGRESSION-NB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif /bin/bash -c '

set -e

if [[ -z "${WANDB_SWEEP_ID}" ]]; then
  echo "Missing WANDB_SWEEP_ID"
  exit 1
fi

source "${HOME}/.bash_profile"

export WANDB_MODE=run
export WANDB_DIR="${LOGDIR}"
export WANDB_PROJECT="physics-uncertainty-exps"
export WANDB_NAME="${SLURM_JOB_NAME}--${SLURM_JOB_ID}"

cd "${WORKDIR}/physics-uncertainty"

export PYTHONPATH="$(pwd):${PYTHONPATH}"

source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH
conda activate

wandb agent --count=1 ${WANDB_SWEEP_ID}
'
