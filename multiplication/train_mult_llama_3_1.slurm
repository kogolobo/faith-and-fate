#!/bin/bash

#SBATCH --job-name=train_llama_3_1

#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=6
#SBATCH --time=100:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --chdir=/mmfs1/home/kogolobo/repos/faith-and-fate/
#SBATCH --export=all
#SBATCH --output=multiplication/mult_train_logs_llama_3_1.txt # where STDOUT goes
#SBATCH --error=multiplication/mult_train_err_llama_3_1.txt # where STDERR goes

# Modules to use (optional).
source /mmfs1/home/kogolobo/.bashrc
conda activate dl
module load cuda
export CUTLASS_PATH=/mmfs1/home/kogolobo/repos/cutlass
OMP_NUM_THREADS=4 accelerate launch --use_deepspeed --mixed_precision bf16 train_model.py \
    --model_name meta-llama/Meta-Llama-3.1-8B \
    --max_memory_MB 80000 \
    --output_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication/models/answer_llama_3_1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --eval_steps 500 \
    --log_steps 50 \
    --data_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication_repro/dataset_tokenized_meta-llama_Meta-Llama-3_1-8B \
    --seed 42 \
    --gradient_accumulation_steps 4 \
    --learning_rate 0.0001 \
    --use_peft \
    --lora_alpha 16 \
    --lora_r 16 \
    --lora_dropout 0.1

python generate_pipeline.py \
    --model_name_or_path /gscratch/amath/kogolobo/faith-and-fate/multiplication/models/answer_llama_3_1 \
    --max_memory_MB 80000 \
    --data_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication_repro/dataset \
    --output_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication/ft_answers_llama_3_1 \
    --task multiplication \
    --nshots 0 \
    --use_peft \
    --max_length 4096 \
    --num_workers 12 \
    --batch_size 64 \
    --seed 42

python generate_pipeline.py \
    --model_name_or_path /gscratch/amath/kogolobo/faith-and-fate/multiplication/models/answer_ood_llama_3_1 \
    --max_memory_MB 80000 \
    --data_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication_repro/dataset \
    --output_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication/ft_answers_llama_3_1 \
    --task multiplication \
    --nshots 0 \
    --use_ood \
    --use_peft \
    --max_length 4096 \
    --num_workers 12 \
    --batch_size 32 \
    --seed 42

OMP_NUM_THREADS=4 accelerate launch --use_deepspeed --mixed_precision bf16 train_model.py \
    --model_name meta-llama/Meta-Llama-3.1-8B \
    --max_memory_MB 80000 \
    --output_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication/models/scratchpad_llama_3_1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --eval_steps 500 \
    --log_steps 50 \
    --data_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication_repro/dataset_tokenized_meta-llama_Meta-Llama-3_1-8B \
    --use_scratchpad \
    --seed 42 \
    --gradient_accumulation_steps 4 \
    --learning_rate 0.0001 \
    --use_peft \
    --lora_alpha 16 \
    --lora_r 16 \
    --lora_dropout 0.1

python generate_pipeline.py \
    --model_name_or_path /gscratch/amath/kogolobo/faith-and-fate/multiplication/models/scratchpad_llama_3_1 \
    --max_memory_MB 80000 \
    --data_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication_repro/dataset \
    --output_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication/ft_scratchpad_llama_3_1 \
    --task multiplication \
    --use_scratchpad \
    --nshots 0 \
    --use_peft \
    --max_length 4096 \
    --num_workers 12 \
    --batch_size 32 \
    --seed 42

python generate_pipeline.py \
    --model_name_or_path /gscratch/amath/kogolobo/faith-and-fate/multiplication/models/scratchpad_ood_llama_3_1 \
    --max_memory_MB 80000 \
    --data_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication_repro/dataset \
    --output_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication/ft_scratchpad_llama_3_1 \
    --task multiplication \
    --use_scratchpad \
    --nshots 0 \
    --use_ood \
    --use_peft \
    --max_length 4096 \
    --num_workers 12 \
    --batch_size 32 \
    --seed 42
