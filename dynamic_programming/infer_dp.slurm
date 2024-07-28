#!/bin/bash

#SBATCH --job-name=dp_infer_llama_3_8B

#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=20:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --chdir=/mmfs1/home/kogolobo/repos/faith-and-fate/
#SBATCH --export=all
#SBATCH --output=dynamic_programming/infer_log_llama_3_8B.txt # where STDOUT goes
#SBATCH --error=dynamic_programming/infer_err_llama_3_8B.txt # where STDERR goes

# Modules to use (optional).
source /mmfs1/home/kogolobo/.bashrc
conda activate dl
module load cuda
export CUTLASS_PATH=/mmfs1/home/kogolobo/repos/cutlass
python generate_pipeline.py \
    --model_name_or_path /gscratch/amath/kogolobo/faith-and-fate/dynamic_programming/best_model_llama_3_8B/ \
    --max_memory_MB 80000 \
    --data_dir /gscratch/amath/kogolobo/faith-and-fate/dynamic_programming/big_data/dataset \
    --output_dir /gscratch/amath/kogolobo/faith-and-fate/dynamic_programming/predictions_llama_3_8B \
    --task dynamic_programming \
    --use_peft \
    --max_length 1024 \
    --num_workers 8 \
    --batch_size 64 \
    --seed 42

