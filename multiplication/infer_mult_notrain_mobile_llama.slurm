#!/bin/bash

#SBATCH --job-name=infer_mobile_llama

#SBATCH --account=amath
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=8
#SBATCH --time=100:00:00 # Max runtime in DD-HH:MM:SS format.

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
    --model_name_or_path mtgv/MobileLLaMA-1.4B-Base \
    --max_memory_MB 40000 \
    --data_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication_repro/dataset \
    --output_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication/0shot_answers_mobile_llama \
    --task multiplication \
    --nshots 0 \
    --max_length 2048 \
    --num_workers 16 \
    --batch_size 128 \
    --seed 42

python generate_pipeline.py \
    --model_name_or_path mtgv/MobileLLaMA-1.4B-Base \
    --max_memory_MB 40000 \
    --data_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication_repro/dataset \
    --output_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication/0shot_ood_answers_mobile_llama \
    --task multiplication \
    --nshots 0 \
    --use_ood \
    --max_length 2048 \
    --num_workers 16 \
    --batch_size 128 \
    --seed 42

python generate_pipeline.py \
    --model_name_or_path mtgv/MobileLLaMA-1.4B-Base \
    --max_memory_MB 40000 \
    --data_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication_repro/dataset \
    --output_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication/0shot_scratchpads_mobile_llama \
    --task multiplication \
    --use_scratchpad \
    --nshots 0 \
    --max_length 2048 \
    --num_workers 16 \
    --batch_size 128 \
    --seed 42

python generate_pipeline.py \
    --model_name_or_path mtgv/MobileLLaMA-1.4B-Base \
    --max_memory_MB 40000 \
    --data_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication_repro/dataset \
    --output_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication/0shot_ood_scratchpads_mobile_llama \
    --task multiplication \
    --use_scratchpad \
    --nshots 0 \
    --use_ood \
    --max_length 2048 \
    --num_workers 16 \
    --batch_size 128 \
    --seed 42

python generate_pipeline.py \
    --model_name_or_path mtgv/MobileLLaMA-1.4B-Base \
    --max_memory_MB 40000 \
    --data_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication_repro/dataset \
    --output_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication/5shot_answers_mobile_llama \
    --task multiplication \
    --nshots 5 \
    --max_length 2048 \
    --num_workers 16 \
    --batch_size 128 \
    --seed 42

python generate_pipeline.py \
    --model_name_or_path mtgv/MobileLLaMA-1.4B-Base \
    --max_memory_MB 40000 \
    --data_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication_repro/dataset \
    --output_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication/5shot_ood_answers_mobile_llama \
    --task multiplication \
    --nshots 5 \
    --use_ood \
    --max_length 2048 \
    --num_workers 16 \
    --batch_size 128 \
    --seed 42

python generate_pipeline.py \
    --model_name_or_path mtgv/MobileLLaMA-1.4B-Base \
    --max_memory_MB 40000 \
    --data_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication_repro/dataset \
    --output_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication/3shot_scratchpads_mobile_llama \
    --task multiplication \
    --use_scratchpad \
    --nshots 3 \
    --max_length 2048 \
    --num_workers 16 \
    --batch_size 128 \
    --seed 42

python generate_pipeline.py \
    --model_name_or_path mtgv/MobileLLaMA-1.4B-Base \
    --max_memory_MB 40000 \
    --data_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication_repro/dataset \
    --output_dir /gscratch/amath/kogolobo/faith-and-fate/multiplication/3shot_ood_scratchpads_mobile_llama \
    --task multiplication \
    --use_scratchpad \
    --nshots 3 \
    --use_ood \
    --max_length 2048 \
    --num_workers 16 \
    --batch_size 128 \
    --seed 42