import fnmatch
import glob
import random
from datasets import load_dataset

STOP_TOKEN = "###"
PREFIX = 'I am a highly intelligent bot answering math questions.'

def create_shots(example):
    example['shot'] = f"Question: {example['question']}\nAnswer: {example['prompt']}"
    return example

def combine_question_and_prompt(example):
    example['text'] = f"{PREFIX}\nQuestion: {example['question']}\nAnswer: {example['prompt']} {STOP_TOKEN}"
    return example

def create_cot_context(example, nshots=0, shot_bank=None):
    shots = []
    if nshots > 0:
        shots = random.sample(shot_bank, nshots) 

    shots_str = '\n'.join(shots) + '\n' if nshots > 0 else ''
    example['context'] = f"{PREFIX}\n{shots_str}Question: {example['question']}\nAnswer: Let's perform the multiplication step by step:\n\n"
    return example

def load_data(data_dir: str, max_train_digits: int, seed: int = 42, nshots=0, shot_bank=None):
    data_pattern = f'{data_dir}/*.json'
    data_files = fnmatch.filter(glob.glob(data_pattern), f'{data_dir}/scratchpad_[0-{max_train_digits}]_by_[0-{max_train_digits}]*.json')
    dataset = load_dataset('json', data_files=data_files)['train'] #.rename_columns({'text': 'question', 'label': 'prompt'})
    dataset = dataset.map(combine_question_and_prompt)
    
    dataset = dataset.train_test_split(test_size=0.1, seed=seed)
    validation_split = dataset['train'].train_test_split(test_size=0.1, seed=seed)
    dataset['train'] = validation_split['train']
    dataset['validation'] = validation_split['test']
    return dataset