#!/bin/bash

#SBATCH --job-name=infer_mobile_llama

#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=4
#SBATCH --time=10:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --chdir=/mmfs1/home/kogolobo/repos/faith-and-fate/
#SBATCH --export=all
#SBATCH --output=multiplication/mult_infer_log_mobile_llama.txt # where STDOUT goes
#SBATCH --error=multiplication/mult_infer_err_mobile_llama.txt # where STDERR goes

# Modules to use (optional).
source /mmfs1/home/kogolobo/.bashrc
conda activate dl
module load cuda
export CUTLASS_PATH=/mmfs1/home/kogolobo/repos/cutlass
python generate_pipeline.py \
    --model_name_or_path /gscratch/amath/kogolobo/faith-and-fate/multiplication/best_model_mobile_llama/ \
    --max_memory_MB 80000 \
    --data_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication/big_data/dataset \
    --output_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication/predictions_mobile_llama \
    --task multiplication \
    --max_length 1024 \
    --num_workers 8 \
    --batch_size 128 \
    --seed 42

