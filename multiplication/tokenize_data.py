from transformers import AutoTokenizer
from datasets import load_from_disk
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='multiplication/big_data/dataset')
    parser.add_argument('--model_name_or_path', type=str, default='gpt2')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    dataset = load_from_disk(args.data_dir).shuffle(seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding=False, truncation=True)
    
    dataset = dataset.map(tokenize_function, batched=True)
    dataset.save_to_disk(args.data_dir + '_tokenized_' + args.model_name_or_path.replace('/', '_').replace('.', '_'))

if __name__ == '__main__':
    main()