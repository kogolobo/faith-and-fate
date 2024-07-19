import glob
import fnmatch
import os

from argparse import ArgumentParser
from datasets import load_dataset, concatenate_datasets, DatasetDict

PREFIX = 'I am a highly intelligent bot answering math questions.'

def main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='multiplication/big_data/scratchpad')
    parser.add_argument("--output_dir", type=str, default='multiplication/big_data/dataset')
    parser.add_argument("--max_train_digits", type=int, default=3)
    parser.add_argument("--test_size", type=int, default=1000)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_pattern = os.path.join(args.data_dir, '*.jsonl')
    filter_pattern = os.path.join(args.data_dir, f'scratchpad_[0-{args.max_train_digits}]_by_[0-{args.max_train_digits}]*.jsonl')

    data_files = fnmatch.filter(glob.glob(data_pattern), filter_pattern)
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

    dataset = dataset.map(lambda example, idx: {
            "id": idx, 
            "text": f"{PREFIX}\nQuestion: {example['prompt']}\nAnswer: Let's perform the multiplication step by step:\n\n{example['completion']}"
        }, with_indices=True)
    dataset.save_to_disk(args.output_dir)


if __name__ == '__main__':
    main()