#!/bin/bash

#SBATCH --job-name=train_mobile_llama

#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=4
#SBATCH --time=100:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --chdir=/mmfs1/home/kogolobo/repos/faith-and-fate/
#SBATCH --export=all
#SBATCH --output=mult_logs_mobile_llama.txt # where STDOUT goes
#SBATCH --error=mult_err_mobile_llama.txt # where STDERR goes

# Modules to use (optional).
source /mmfs1/home/kogolobo/.bashrc
conda activate dl
module load cuda
export CUTLASS_PATH=/mmfs1/home/kogolobo/repos/cutlass
OMP_NUM_THREADS=4 accelerate launch --use_deepspeed --mixed_precision bf16 multiplication/train_model.py \
    --model_name mtgv/MobileLLaMA-1.4B-Base \
    --max_memory_MB 80000 \
    --output_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication/best_model_mobile_llama \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --eval_steps 500 \
    --log_steps 50 \
    --data_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication/big_data/dataset \
    --seed 42 \
    --gradient_accumulation_steps 4 \
    --learning_rate 0.0003 \
    --lora_alpha 16 \
    --lora_r 16 \
    --lora_dropout 0.1
