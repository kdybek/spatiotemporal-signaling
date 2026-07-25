#!/bin/bash
#SBATCH --job-name=spatiotemporal_signaling_rvm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx6000:1

#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

export SCRATCH=~/myscratch
export DATA_DIR=~/myimaging

export XDG_CACHE_HOME=$SCRATCH/.cache
export WANDB_API_KEY=$(cat ~/.wandb_key)

cd ~/spatiotemporal-signaling/src/training/RVM
source .venv/bin/activate

python main.py --dataset_path $DATA_DIR/geminin_drugs_full_vid_2.zarr \
               --run_group rvm \
               --save_dir checkpoints/rvm \
               --steps 100000 \
               --eval_interval 1000 \
               --save_interval 10000 \
               --train_split 0.8 \
               --batch_size 16
