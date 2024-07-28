#!/bin/bash

#SBATCH --job-name=dp_train_llama_3_8B

#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=4
#SBATCH --time=100:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --chdir=/mmfs1/home/kogolobo/repos/faith-and-fate/
#SBATCH --export=all
#SBATCH --output=dynamic_programming/train_log_llama_3_8B.txt # where STDOUT goes
#SBATCH --error=dynamic_programming/train_err_llama_3_8B.txt # where STDERR goes

# Modules to use (optional).
source /mmfs1/home/kogolobo/.bashrc
conda activate dl
module load cuda
export CUTLASS_PATH=/mmfs1/home/kogolobo/repos/cutlass
OMP_NUM_THREADS=4 accelerate launch --use_deepspeed --mixed_precision bf16 train_model.py \
    --model_name meta-llama/Meta-Llama-3-8B \
    --max_memory_MB 80000 \
    --output_dir /gscratch/amath/kogolobo/faith-and-fate/dynamic_programming/best_model_llama_3_8B \
    --num_train_epochs 2 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --eval_steps 150 \
    --log_steps 20 \
    --data_dir /gscratch/amath/kogolobo/faith-and-fate/dynamic_programming/big_data/dataset \
    --seed 42 \
    --gradient_accumulation_steps 4 \
    --learning_rate 0.0001 \
    --use_peft \
    --lora_alpha 16 \
    --lora_r 16 \
    --lora_dropout 0.1
