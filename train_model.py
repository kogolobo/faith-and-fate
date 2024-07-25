from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import set_seed
from datasets import load_from_disk

def main():
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--output_dir', type=str, default='multiplication/best_model')
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=8)
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--log_steps', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--data_dir', type=str, default='multiplication/big_data/dataset')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

    dataset = load_from_disk(args.data_dir).shuffle(seed=args.seed)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding=False, truncation=True)
    
    dataset = dataset.map(tokenize_function, batched=True, num_proc=8)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        save_strategy='steps',
        save_steps=args.eval_steps,
        save_total_limit=1,
        evaluation_strategy='steps',
        eval_steps=args.eval_steps,
        logging_steps=args.log_steps,
        seed=args.seed,
        do_train=True,
        learning_rate=args.learning_rate,
        load_best_model_at_end=True
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
    tokenizer.save_pretrained(args.output_dir)

if __name__ == '__main__':
    main()