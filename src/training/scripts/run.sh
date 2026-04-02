#!/bin/bash
#SBATCH --job-name=spatiotemporal_signaling_run
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx6000:1

#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

export SCRATCH=~/myscratch
export DATA_DIR=/mnt/imaging.data/zppmimuw

export XDG_CACHE_HOME=$SCRATCH/.cache
export WANDB_API_KEY=$(cat ~/.wandb_key)

cd ~/spatiotemporal-signaling/src/training
source .venv/bin/activate

python main.py --train_dataset_path $DATA_DIR/geminin_drugs_16x2x224x224_train.zarr \
               --test_dataset_path $DATA_DIR/geminin_drugs_16x2x224x224_test.zarr \
               --batch_size 64 \
