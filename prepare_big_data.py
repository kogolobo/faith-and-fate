import glob
import fnmatch
import os

from argparse import ArgumentParser
from datasets import load_dataset, concatenate_datasets, DatasetDict

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

def main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='multiplication/big_data/scratchpad')
    parser.add_argument("--output_dir", type=str, default='multiplication/big_data/dataset')
    parser.add_argument("--max_train_problem_size", type=int, default=3)
    parser.add_argument("--test_size", type=int, default=1000)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--task', choices=['multiplication', 'dynamic_programming'], default='multiplication')
    args = parser.parse_args()

    data_pattern = os.path.join(args.data_dir, '*.jsonl')

    prefix, instruction = PROMPTS[args.task]['prefix'], PROMPTS[args.task]['instruction']
    if args.task == 'multiplication':
        filter_pattern = os.path.join(args.data_dir, f'scratchpad_[0-{args.max_train_problem_size}]_by_[0-{args.max_train_problem_size}]*.jsonl')
    elif args.task == 'dynamic_programming':
        filter_pattern = os.path.join(args.data_dir, f'data_scratchpad_n_[0-{args.max_train_problem_size}]*.jsonl')

    data_files = fnmatch.filter(glob.glob(data_pattern), filter_pattern)
    print(data_files)
    datasets = []
    for data_file in data_files:
        dataset = load_dataset('json', data_files=data_file)['train']
        dataset.shuffle(seed=args.seed)

        test_size = min(args.test_size, int(len(dataset) * args.test_ratio))
        data_split = dataset.train_test_split(test_size=test_size, seed=args.seed)
        
        val_size = min(args.test_size, int(len(data_split['train']) * args.test_ratio))
        train_split = data_split['train'].train_test_split(test_size=val_size, seed=args.seed)

        dataset = DatasetDict({
            'train': train_split['train'],
            'validation': train_split['test'],
            'test': data_split['test']
        })
        datasets.append(dataset)

    dataset = DatasetDict({
        split : concatenate_datasets([dataset[split] for dataset in datasets]) 
        for split in datasets[0].keys()
    }).shuffle(seed=args.seed)

    dataset = dataset.map(
        lambda example, idx: 
        {
            "id": idx, 
            "text": f"{prefix}\nQuestion: {example['prompt']}\nAnswer: {instruction}\n\n{example['completion']}"
        }, with_indices=True)
    dataset.save_to_disk(args.output_dir)


if __name__ == '__main__':
    main()