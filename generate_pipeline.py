from functools import partial
import os
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from transformers.trainer_utils import set_seed
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_from_disk
from argparse import ArgumentParser
from prepare_big_data import PROMPTS
from tqdm import tqdm
import numpy as np

PROMPTS = {
    'multiplication': {
        'prefix': 'I am a highly intelligent bot answering math questions.',
        'instruction': "Let's perform the multiplication step by step:"
    },
    'dynamic_programming': {
        'prefix': 'I am a highly intelligent bot solving logic puzzles.',
        'instruction': "We will solve any task instance by using dynamic programming. We define dp[i] as the maximum sum of a subsequence that does not include adjacent elements, when considering only the elements of the input from the i-th position onwards."
    }
}

def create_context(
        dataset, 
        validation_dataset, 
        prefix, 
        instruction, 
        use_scratchpad: bool = False, 
        nshots=5, 
        num_proc=4
):
    def craft_prompt(example) -> str:
        answer_key = 'completion' if not use_scratchpad else 'scratchpad'
        instruction_str = instruction + "\n\n" if use_scratchpad else ""
        if nshots > 0:
            few_shot_idx = np.random.choice(len(validation_dataset), nshots, replace=False)
            few_shots = validation_dataset.select(few_shot_idx)
            postfix = " ###" if not use_scratchpad else ""
            few_shot_str = "\n".join([
                f"Question: {demo['prompt']}\nAnswer: {instruction_str}{demo[answer_key]}{postfix}"
                for demo in few_shots
            ]) + "\n\n"
        else:
            few_shot_str = ""
        return f"{prefix}\n{few_shot_str}Question: {example['prompt']}\nAnswer: {instruction_str}"

    
    return dataset.map(
        lambda example: {
            'context': craft_prompt(example)
        }, num_proc=num_proc
    )

def main():
    parser = ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='/gscratch/amath/kogolobo/faith-and-fate/multiplication_repro/best_model_pythia/')
    parser.add_argument('--data_dir', type=str, default='/gscratch/amath/kogolobo/faith-and-fate/multiplication/big_data/dataset')
    parser.add_argument('--use_ood', action='store_true', default=False)
    parser.add_argument('--use_scratchpad', action='store_true', default=False)
    parser.add_argument('--nshots', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='multiplication/predictions_pythia')
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--max_new_tokens', type=int, default=-1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_memory_MB', type=int, default=80000)
    parser.add_argument('--use_peft', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--task', choices=['multiplication', 'dynamic_programming'], default='multiplication')
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
        
    if args.use_peft:
        peft_config = PeftConfig.from_pretrained(args.model_name_or_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path, 
            torch_dtype=torch.bfloat16,
            device_map=device_map, 
            max_memory=max_memory,
            attn_implementation="flash_attention_2"
        )
        model = PeftModel.from_pretrained(model=base_model, model_id=args.model_name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map=device_map, max_memory=max_memory)
    
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.padding_side = 'left'
    if not tokenizer.pad_token_id:
        if model.config.pad_token_id is not None:
            tokenizer.pad_token_id = model.config.pad_token_id
        else: 
            tokenizer.pad_token_id = model.config.eos_token_id
            
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

    prefix, instruction = PROMPTS[args.task]['prefix'], PROMPTS[args.task]['instruction']

    dataset = load_from_disk(args.data_dir)
    split = 'ood' if args.use_ood else 'test'
    dataset = create_context(
        dataset[split], 
        dataset['validation'], 
        prefix, 
        instruction, 
        use_scratchpad=args.use_scratchpad, 
        nshots=args.nshots, 
        num_proc=args.num_workers
    )
    print(f"First data example: {dataset['context'][0]}")

    generaion_kwargs = {
        'num_return_sequences': 1,
        'truncation': True,
        'return_full_text': False,
        'num_workers': args.num_workers,
        'batch_size': args.batch_size
    }

    if args.max_new_tokens > 0:
        generaion_kwargs['max_new_tokens'] = args.max_new_tokens
    else:
        generaion_kwargs['max_length'] = args.max_length

    pipe = partial(generator, **generaion_kwargs)
    
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