#!/bin/bash

#SBATCH --job-name=infer_llama_3_1

#SBATCH --account=amath
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=8
#SBATCH --time=250:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --chdir=/mmfs1/home/kogolobo/repos/faith-and-fate/
#SBATCH --export=all
#SBATCH --output=multiplication/mult_infer_log_llama_3_1.txt # where STDOUT goes
#SBATCH --error=multiplication/mult_infer_err_llama_3_1.txt # where STDERR goes

# Modules to use (optional).
source /mmfs1/home/kogolobo/.bashrc
conda activate dl
module load cuda
export CUTLASS_PATH=/mmfs1/home/kogolobo/repos/cutlass
python generate_pipeline.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
    --max_memory_MB 40000 \
    --data_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication_repro/dataset \
    --output_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication/predictions/0shot_answers_llama_3_1 \
    --task multiplication \
    --nshots 0 \
    --max_new_tokens 20 \
    --num_workers 16 \
    --batch_size 8 \
    --seed 42

python generate_pipeline.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
    --max_memory_MB 40000 \
    --data_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication_repro/dataset \
    --output_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication/predictions/0shot_ood_answers_llama_3_1 \
    --task multiplication \
    --nshots 0 \
    --use_ood \
    --max_new_tokens 20 \
    --num_workers 16 \
    --batch_size 8 \
    --seed 42

python generate_pipeline.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
    --max_memory_MB 40000 \
    --data_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication_repro/dataset \
    --output_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication/predictions/5shot_answers_llama_3_1 \
    --task multiplication \
    --nshots 5 \
    --max_new_tokens 20 \
    --num_workers 16 \
    --batch_size 8 \
    --seed 42

python generate_pipeline.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
    --max_memory_MB 40000 \
    --data_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication_repro/dataset \
    --output_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication/predictions/5shot_ood_answers_llama_3_1 \
    --task multiplication \
    --nshots 5 \
    --use_ood \
    --max_new_tokens 20 \
    --num_workers 16 \
    --batch_size 8 \
    --seed 42

python generate_pipeline.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
    --max_memory_MB 40000 \
    --data_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication_repro/dataset \
    --output_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication/predictions/5shot_scratchpads_llama_3_1 \
    --task multiplication \
    --use_scratchpad \
    --nshots 5 \
    --max_length 4096 \
    --num_workers 16 \
    --batch_size 8 \
    --seed 42

python generate_pipeline.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
    --max_memory_MB 40000 \
    --data_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication_repro/dataset \
    --output_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication/predictions/5shot_ood_scratchpads_llama_3_1 \
    --task multiplication \
    --use_scratchpad \
    --nshots 5 \
    --use_ood \
    --max_length 4096 \
    --num_workers 16 \
    --batch_size 8 \
    --seed 42