from functools import partial
import glob
import numpy as np
import os

from argparse import ArgumentParser
from datasets import load_dataset, concatenate_datasets, DatasetDict
from generate_graph_from_scratchpad import extract_numbers

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

def example_has_numbers(example, numbers: set) -> bool:
    numbers_present = extract_numbers(example['prompt'])
    return any(num in numbers for num in numbers_present)

def remove_test_numbers(train_dataset, test_dataset, num_proc=4):
    test_numbers = test_dataset.map(
        lambda example: {
            'numbers': extract_numbers(example['prompt'])
        }, num_proc=num_proc
    )['numbers']
    test_numbers = set(num for tup in test_numbers for num in tup)
    return train_dataset.filter(
        lambda example: not example_has_numbers(example, test_numbers), num_proc=num_proc
    )

def get_completion(example):
    a, b = extract_numbers(example['prompt'])
    return f"{a * b}\n"


def main():
    parser = ArgumentParser()
    parser.add_argument("--scratchpad_dir", type=str, default='multiplication/scratchpads_from_ft/')
    parser.add_argument("--test_dir", type=str, default='multiplication/scratchpad/')
    parser.add_argument("--output_dir", type=str, default='multiplication/repro_dataset')

    parser.add_argument("--test_size", type=int, default=1000)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--num_proc", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--task', choices=['multiplication'], default='multiplication')
    args = parser.parse_args()

    prefix, instruction = PROMPTS[args.task]['prefix'], PROMPTS[args.task]['instruction']

    max_problem_size = 9
    datasets = []
    test_datasets_ood = []
    for a_num in range(1, 6):
        for b_num in range(1, a_num + 1):
            if a_num > 4 or b_num > 4 or a_num * b_num > max_problem_size:
                test_dataset_pat = os.path.join(args.test_dir, f'scratchpad_{a_num}_by_{b_num}*.json')
                test_dataset_path = glob.glob(test_dataset_pat)[0]
                test_dataset = load_dataset('json', data_files=test_dataset_path)['train']\
                                .shuffle(seed=args.seed)\
                                .remove_columns('id')\
                                .rename_column('prompt', 'scratchpad')\
                                .rename_column('question', 'prompt')\
                                .map(lambda example: {
                                    "completion": get_completion(example),
                                }, num_proc=args.num_proc)
                test_datasets_ood.append(test_dataset)
            else:
                scratchpad_file = os.path.join(args.scratchpad_dir, f'scratchpad_{a_num}_by_{b_num}.jsonl')
                assert os.path.exists(scratchpad_file), f"File {scratchpad_file} does not exist"
                dataset = load_dataset('json', data_files=scratchpad_file)['train']
                dataset.shuffle(seed=args.seed)

                test_size = min(args.test_size, int(len(dataset) * args.test_ratio))
                data_split = dataset.train_test_split(test_size=test_size, seed=args.seed)

                val_size = min(args.test_size, int(len(data_split['train']) * args.test_ratio))
                train_dev_split = data_split['train'].train_test_split(test_size=val_size, seed=args.seed)

                dataset = DatasetDict({
                    'train': train_dev_split['train'],
                    'validation': train_dev_split['test'],
                    'test': data_split['test']
                })
                datasets.append(dataset)

    dataset = DatasetDict({
        split : concatenate_datasets([dataset[split] for dataset in datasets]) 
        for split in datasets[0].keys()
    }).shuffle(seed=args.seed)
    dataset['ood'] = concatenate_datasets(test_datasets_ood).shuffle(seed=args.seed)

    # dataset['train'] = remove_test_numbers(dataset['train'], dataset['test'], num_proc=args.num_proc)
    dataset = dataset.map(
        lambda example, idx: 
        {
            "id": idx, 
            "answer_text": f"{prefix}\nQuestion: {example['prompt']}\nAnswer: {example['completion']} ###",
            "scratchpad_text": f"{prefix}\nQuestion: {example['prompt']}\nAnswer: {instruction}\n\n{example['scratchpad']}"
        }, with_indices=True, num_proc=args.num_proc)
    
    dataset.save_to_disk(args.output_dir)
    print("Data size: ", len(dataset['train']), len(dataset['validation']), len(dataset['test']), len(dataset['ood']))


if __name__ == '__main__':
    main()