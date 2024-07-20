#!/bin/bash

#SBATCH --job-name=train_mult

#SBATCH --account=stf
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --gpus=7
#SBATCH --cpus-per-gpu=4
#SBATCH --time=100:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --chdir=/mmfs1/home/kogolobo/repos/faith-and-fate/
#SBATCH --export=all
#SBATCH --output=mult_logs.txt # where STDOUT goes
#SBATCH --error=mult_err.txt # where STDERR goes

# Modules to use (optional).
source /mmfs1/home/kogolobo/.bashrc
conda activate dl
module load cuda
export CUTLASS_PATH=/mmfs1/home/kogolobo/repos/cutlass
OMP_NUM_THREADS=4 accelerate launch --use_deepspeed --mixed_precision bf16 multiplication/train_model.py --model_name meta-llama/Meta-Llama-3-8B-Instruct --max_memory_MB 48000 --output_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication/best_model --num_train_epochs 1 --per_device_train_batch_size 16 --per_device_eval_batch_size 8 --eval_steps 500 --log_steps 50  --data_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication/big_data/dataset --seed 42