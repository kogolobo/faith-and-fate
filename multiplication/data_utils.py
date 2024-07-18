import fnmatch
import glob
from datasets import load_dataset

def combine_question_and_prompt(example):
    example['text'] = example['question'] + ' ' + example['prompt']
    return example

def load_data(data_dir: str, max_train_digits: int, seed: int = 42):
    data_pattern = f'{data_dir}/*.json'
    data_files = fnmatch.filter(glob.glob(data_pattern), f'{data_dir}/scratchpad_[0-{max_train_digits}]_by_[0-{max_train_digits}]*.json')
    dataset = load_dataset('json', data_files=data_files)['train'] #.rename_columns({'text': 'question', 'label': 'prompt'})
    dataset = dataset.map(combine_question_and_prompt)
    
    dataset = dataset.train_test_split(test_size=0.1, seed=seed)
    validation_split = dataset['train'].train_test_split(test_size=0.1, seed=seed)
    dataset['train'] = validation_split['train']
    dataset['validation'] = validation_split['test']
    return dataset