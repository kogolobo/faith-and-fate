#!/bin/bash

#SBATCH --job-name=infer_pythia

#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=4
#SBATCH --time=10:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --chdir=/mmfs1/home/kogolobo/repos/faith-and-fate/
#SBATCH --export=all
#SBATCH --output=mult_infer_log_pythia.txt # where STDOUT goes
#SBATCH --error=mult_infer_err_pythia.txt # where STDERR goes

# Modules to use (optional).
source /mmfs1/home/kogolobo/.bashrc
conda activate dl
module load cuda
export CUTLASS_PATH=/mmfs1/home/kogolobo/repos/cutlass
python multiplication/generate_pipeline.py \
    --model_name_or_path /gscratch/amath/kogolobo/faith-and-fate/multiplication/best_model_pythia/ \
    --max_memory_MB 80000 \
    --data_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication/big_data/dataset \
    --output_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication/predictions_pythia \
    --max_length 1024 \
    --num_workers 8 \
    --batch_size 64 \
    --use_peft \
    --seed 42
