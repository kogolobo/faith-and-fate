from argparse import ArgumentParser
import torch
if torch.cuda.is_available():   
    torch.backends.cuda.matmul.allow_tf32 = True
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, BitsAndBytesConfig

from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import set_seed
from data_utils import load_data
from datasets import load_from_disk
import bitsandbytes as bnb

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer

def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

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
    parser.add_argument('--max_train_digits', type=int, default=3)
    parser.add_argument('--data_dir', type=str, default='multiplication/big_data/dataset')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    compute_dtype = torch.bfloat16
    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
    )
    torch_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=compute_dtype, quantization_config=quantization_config, attn_implementation="flash_attention_2")
    model.config.torch_dtype = compute_dtype
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    modules = find_all_linear_names(args, model)
    lora_config = LoraConfig(
        r=4,
        lora_alpha=1,
        target_modules=modules,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
    # dataset = load_data(args.data_dir, args.max_train_digits, args.seed)
    dataset = load_from_disk(args.data_dir).shuffle(seed=args.seed)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding=False, truncation=True)
    
    dataset = dataset.map(tokenize_function, batched=True)

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
        load_best_model_at_end=True,
        optim='paged_adamw_32bit'
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