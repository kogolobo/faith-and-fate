import torch
import os
import json

from tqdm import tqdm
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers.trainer_utils import set_seed
from data_utils import load_data, STOP_TOKEN, create_cot_context, create_shots
from transformers.generation.configuration_utils import GenerationConfig
from transformers import StoppingCriteria, StoppingCriteriaList

class StopTokenCriteria(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return self.stop_token_id in input_ids[0]

class CustomCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features):
        padded_features = [
            {
                'input_ids': feature.pop('input_ids'),
                'attention_mask': feature.pop('attention_mask')
            }
            for feature in features
        ]
        padded_batch = super().__call__(padded_features)
        text_batch = {
            k: [feature[k] for feature in features]
            for k in features[0].keys()
        }
        batch = {**padded_batch, **text_batch}
        return batch

def main():
    parser = ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='yuntian-deng/gpt2-explicit-cot-multiplication')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default='multiplication/generations')
    parser.add_argument('--max_train_digits', type=int, default=3)
    parser.add_argument('--do_ood', action='store_true')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=800)
    parser.add_argument('--data_dir', type=str, default='multiplication/scratchpad')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, pad_token_id=tokenizer.eos_token_id).to(args.device)
    
    collator = CustomCollatorWithPadding(tokenizer=tokenizer)
    dataset = load_data(args.data_dir, args.max_train_digits, args.seed)
    validation_data = dataset['validation']
    validation_data = validation_data.map(create_shots)
    shot_bank = validation_data['shot']
    dataset = dataset['test'].map(create_cot_context, nshots=1, shot_bank=shot_bank)
    
    def tokenize_function(examples):
        return tokenizer(examples['context'], padding=False, truncation=True)
    dataset = dataset.map(tokenize_function, batched=True)

    dataloader = DataLoader(
        dataset['test'].select(range(50)),
        batch_size=args.batch_size, 
        collate_fn=collator, 
        shuffle=False 
    )

    model.eval()
    
    stop_token_id = tokenizer.encode(STOP_TOKEN, add_special_tokens=False)[0]
    stopping_criteria = StoppingCriteriaList([StopTokenCriteria(stop_token_id)])
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        # num_beams=args.beam_size,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False
    )

    generated_answers = []
    for batch in tqdm(dataloader, desc='Generating answers'):
        with torch.no_grad():
            input_ids = batch.pop('input_ids').to(args.device)
            attention_mask = batch.pop('attention_mask').to(args.device)
            text = batch.pop('text') if 'text' in batch else None
            
            outputs = model.generate(input_ids, attention_mask=attention_mask, generation_config=generation_config, stopping_criteria=stopping_criteria)
            answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            batch['answer'] = [answers[i].replace(batch['question'][i], '').strip() for i in range(len(answers))]
            generated_answers.extend(
                [ {k: v[i] for k, v in batch.items()} for i in range(len(answers)) ]
            )

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'generated_answers.json'), 'w') as f:
        json.dump(generated_answers, f, indent=4)

    


if __name__ == '__main__':
    main()