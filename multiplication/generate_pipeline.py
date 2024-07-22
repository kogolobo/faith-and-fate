from functools import partial
import os
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import set_seed
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_from_disk
from argparse import ArgumentParser
from prepare_big_data import PREFIX
from tqdm import tqdm

def main():
    parser = ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='gpt2')
    parser.add_argument('--data_dir', type=str, default='multiplication/scratchpad')
    parser.add_argument('--output_dir', type=str, default='multiplication/predictions')
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_memory_MB', type=int, default=12000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    n_gpus = torch.cuda.device_count()
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}
        
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map=device_map, max_memory=max_memory)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.padding_side = 'left'
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    dataset = load_from_disk(args.data_dir)['test'].map(
        lambda example:
        {
            "context": f"{PREFIX}\nQuestion: {example['prompt']}\nAnswer: Let's perform the multiplication step by step:\n\n"
        }, num_proc=args.num_workers
    )


    pipe = partial(
        generator, 
        num_return_sequences=1, 
        truncation=True, 
        return_full_text=False,
        max_length=args.max_length, 
        num_workers=args.num_workers,
        batch_size=args.batch_size
    )
    generated_text = []
    for result in tqdm(pipe(KeyDataset(dataset, key='context')), total=len(dataset), desc='Generating'):
        generated_text.extend([res['generated_text'] for res in result])

    dataset = dataset.map(
        lambda example, idx:
        {
            "generated": generated_text[idx]
        }, with_indices=True, num_proc=args.num_workers
    )
    dataset.to_json(
        f'{args.output_dir}/predictions.json', 
        num_proc=args.num_workers,
        orient='records', 
        lines=False, 
        indent=4
    )


if __name__ == '__main__':
    main()