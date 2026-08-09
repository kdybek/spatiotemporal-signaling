#!/bin/bash
#SBATCH --job-name=cp_from_scratch
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --partition=all
#SBATCH --nodelist=izbdodoma
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

export SCRATCH=~/myscratch

export XDG_CACHE_HOME=$SCRATCH/.cache
export WANDB_API_KEY=$(cat ~/.wandb_key)

mkdir -p ~/spatiotemporal-signaling/src/training/RVM/checkpoints
mkdir -p ~/spatiotemporal-signaling/src/training/RVM/latents
cp -a $SCRATCH/spatiotemporal-signaling/src/training/RVM/checkpoints/. ~/spatiotemporal-signaling/src/training/RVM/checkpoints/
cp -a $SCRATCH/spatiotemporal-signaling/src/training/RVM/latents/. ~/spatiotemporal-signaling/src/training/RVM/latents/
cp $SCRATCH/spatiotemporal-signaling/src/training/RVM/analyze_latent.ipynb ~/spatiotemporal-signaling/src/training/RVM/analyze_latent.ipynb
