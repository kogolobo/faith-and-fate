import torch
import fnmatch
import glob
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import set_seed

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

def main():
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--output_dir', type=str, default='multiplication/output')
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--eval_steps', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--max_train_digits', type=int, default=3)
    parser.add_argument('--data_dir', type=str, default='multiplication/scratchpad')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_data(args.data_dir, args.max_train_digits, args.seed)
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding=False, truncation=True)
    
    dataset = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        save_steps=args.save_steps,
        seed=args.seed,
        do_train=True,
        eval_steps=args.eval_steps,
        learning_rate=args.learning_rate
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=collator
    )

    trainer.train()

    trainer.save_model(args.output_dir)


if __name__ == '__main__':
    main()