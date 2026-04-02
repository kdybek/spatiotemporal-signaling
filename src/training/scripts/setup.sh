#!/bin/bash
#SBATCH --job-name=spatiotemporal_signaling_setup
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1

#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module load CUDA/12.2.0

export SCRATCH=~/myscratch
export DATA_DIR=~/myimaging

export XDG_CACHE_HOME=$SCRATCH/.cache
export WANDB_API_KEY=$(cat ~/.wandb_key)

cd ~/spatiotemporal-signaling/src/training
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
