from functools import partial
from transformers import AutoTokenizer
from datasets import load_from_disk
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='multiplication/big_data/dataset')
    parser.add_argument('--model_name_or_path', type=str, default='gpt2')
    parser.add_argument('--num_proc', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    dataset = load_from_disk(args.data_dir).shuffle(seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    def tokenize_function(examples, key):
        return tokenizer(examples[key], padding=False, truncation=True)
    
    scratchpad_tok_func = partial(tokenize_function, key="scratchpad_text")
    answer_tok_func = partial(tokenize_function, key="answer_text")

    dataset = dataset.map(answer_tok_func, batched=True, num_proc=args.num_proc)\
                .rename_column('input_ids', 'answer_input_ids')\
                .rename_column('attention_mask', 'answer_attention_mask')
    
    dataset = dataset.map(scratchpad_tok_func, batched=True, num_proc=args.num_proc)\
                .rename_column('input_ids', 'scratchpad_input_ids')\
                .rename_column('attention_mask', 'scratchpad_attention_mask')
    
    dataset.save_to_disk(args.data_dir + '_tokenized_' + args.model_name_or_path.replace('/', '_').replace('.', '_'))

if __name__ == '__main__':
    main()
