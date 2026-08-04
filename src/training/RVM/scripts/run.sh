#!/bin/bash
#SBATCH --job-name=spatiotemporal_signaling_rvm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx6000:1

#SBATCH --time=256:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G

export SCRATCH=~/myscratch
export DATA_DIR=~/myimaging

export XDG_CACHE_HOME=$SCRATCH/.cache
export WANDB_API_KEY=$(cat ~/.wandb_key)

cd $SCRATCH/spatiotemporal-signaling/src/training/RVM
git pull
source .venv/bin/activate

python main.py --dataset_path $DATA_DIR/geminin_drugs_full_vid_3.zarr \
               --run_group rvm_200K_2Ch \
               --save_dir checkpoints/rvm_200K_2Ch \
               --channel_names "Ch_ERK-KTR Ch_H2B" \
               --steps 200000 \
               --eval_interval 5000 \
               --save_interval 20000 \
               --batch_size 16

