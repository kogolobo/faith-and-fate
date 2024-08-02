#!/bin/bash

#SBATCH --job-name=train_mobile_llama

#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=6
#SBATCH --time=250:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --chdir=/mmfs1/home/kogolobo/repos/faith-and-fate/
#SBATCH --export=all
#SBATCH --output=multiplication/mult_train_logs_mobile_llama.txt # where STDOUT goes
#SBATCH --error=multiplication/mult_train_err_mobile_llama.txt # where STDERR goes

# Modules to use (optional).
source /mmfs1/home/kogolobo/.bashrc
conda activate dl
module load cuda
export CUTLASS_PATH=/mmfs1/home/kogolobo/repos/cutlass
OMP_NUM_THREADS=4 accelerate launch --use_deepspeed --mixed_precision bf16 train_model.py \
    --model_name mtgv/MobileLLaMA-1.4B-Base \
    --max_memory_MB 80000 \
    --output_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication/models/answer_mobile_llama \
    --num_train_epochs 3 \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 256 \
    --eval_steps 500 \
    --log_steps 50 \
    --data_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication_repro/dataset_tokenized_mtgv_MobileLLaMA-1_4B-Base \
    --seed 42 \
    --gradient_accumulation_steps 4 \
    --learning_rate 0.0005 \
    --lora_alpha 16 \
    --lora_r 16 \
    --lora_dropout 0.1

python generate_pipeline.py \
    --model_name_or_path /gscratch/amath/kogolobo/faith-and-fate/multiplication/models/answer_mobile_llama \
    --max_memory_MB 80000 \
    --data_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication_repro/dataset \
    --output_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication/predictions/ft_answers_mobile_llama \
    --task multiplication \
    --nshots 0 \
    --max_new_tokens 20 \
    --num_workers 12 \
    --batch_size 32 \
    --seed 42

python generate_pipeline.py \
    --model_name_or_path /gscratch/amath/kogolobo/faith-and-fate/multiplication/models/answer_mobile_llama \
    --max_memory_MB 80000 \
    --data_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication_repro/dataset \
    --output_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication/predictions/ft_answers_ood_mobile_llama \
    --task multiplication \
    --nshots 0 \
    --use_ood \
    --max_new_tokens 20 \
    --num_workers 12 \
    --batch_size 32 \
    --seed 42

OMP_NUM_THREADS=4 accelerate launch --use_deepspeed --mixed_precision bf16 train_model.py \
    --model_name mtgv/MobileLLaMA-1.4B-Base \
    --max_memory_MB 80000 \
    --output_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication/models/scratchpad_mobile_llama \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --eval_steps 500 \
    --log_steps 50 \
    --data_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication_repro/dataset_tokenized_mtgv_MobileLLaMA-1_4B-Base \
    --use_scratchpad \
    --seed 42 \
    --gradient_accumulation_steps 4 \
    --learning_rate 0.0003 \
    --lora_alpha 16 \
    --lora_r 16 \
    --lora_dropout 0.1

python generate_pipeline.py \
    --model_name_or_path /gscratch/amath/kogolobo/faith-and-fate/multiplication/models/scratchpad_mobile_llama \
    --max_memory_MB 80000 \
    --data_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication_repro/dataset \
    --output_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication/predictions/ft_scratchpad_mobile_llama \
    --task multiplication \
    --use_scratchpad \
    --nshots 0 \
    --max_length 2048 \
    --num_workers 12 \
    --batch_size 32 \
    --seed 42

python generate_pipeline.py \
    --model_name_or_path /gscratch/amath/kogolobo/faith-and-fate/multiplication/models/scratchpad_mobile_llama \
    --max_memory_MB 80000 \
    --data_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication_repro/dataset \
    --output_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication/predictions/ft_scratchpad_ood_mobile_llama \
    --task multiplication \
    --use_scratchpad \
    --nshots 0 \
    --use_ood \
    --max_length 2048 \
    --num_workers 12 \
    --batch_size 32 \
    --seed 42
