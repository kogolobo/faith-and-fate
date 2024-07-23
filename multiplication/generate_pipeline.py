from functools import partial
import os
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from transformers.trainer_utils import set_seed
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_from_disk
from argparse import ArgumentParser
from prepare_big_data import PREFIX
from tqdm import tqdm

def main():
    parser = ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='/gscratch/amath/kogolobo/faith-and-fate/multiplication/best_model_pythia/')
    parser.add_argument('--data_dir', type=str, default='/gscratch/amath/kogolobo/faith-and-fate/multiplication/big_data/dataset')
    parser.add_argument('--output_dir', type=str, default='multiplication/predictions_pythia')
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_memory_MB', type=int, default=80000)
    parser.add_argument('--use_peft', action='store_true', default=False)
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