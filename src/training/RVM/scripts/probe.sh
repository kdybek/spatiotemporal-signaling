#!/bin/bash
#SBATCH --job-name=spatiotemporal_probing
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx6000:1

#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G

export SCRATCH=~/myscratch
export DATA_DIR=$SCRATCH/datasets

export XDG_CACHE_HOME=$SCRATCH/.cache
export WANDB_API_KEY=$(cat ~/.wandb_key)

cd $SCRATCH/spatiotemporal-signaling/
git restore .
git pull
cd src/training/RVM
source .venv/bin/activate

python -u probe.py
