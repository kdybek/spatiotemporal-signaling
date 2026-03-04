#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=test.out
#SBATCH --error=test.err

cd ~/zppmimuw
source .venv/bin/activate
python create_dataset.py
