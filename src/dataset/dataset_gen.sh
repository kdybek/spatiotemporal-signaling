#!/bin/bash
#SBATCH --job-name=dataset_gen
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

cd /home/zppmimuw/spatiotemporal-signaling/src/dataset
source .venv/bin/activate

python dataset_gen.py --input=matched.pkl --output=/mnt/imaging.data/zppmimuw/spatiotemp_full_vid.zarr
