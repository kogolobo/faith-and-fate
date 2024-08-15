import argparse
import json
import glob
import os
import pprint
from statistics import mean
from typing import Optional
from tqdm import tqdm
from generate_graph_from_scratchpad import extract_numbers
from graph_error_analysis import compute_accuracy
import re
from collections import defaultdict
import numpy as np
import seaborn as sns

def extract_answer(text: str) -> Optional[int]:
    equal_sign_match = re.search(r'=\s*((\d+[.,\s]*)+)\.?$', text)
    if equal_sign_match:
        text = equal_sign_match.group(1)

    try:
        return int(text.replace(",", "").replace(" ", "").replace(".", ""))
    except ValueError:
        return None
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=None, help="Path to the answer folder")
    parser.add_argument('--use_scratchpad', action='store_true', default=False)
    args = parser.parse_args()
    pprint.pprint(args)

    accuracies = defaultdict(list)
    overall_accuracy = []

    # Produce matrix of accuracies: (num digits, num digits) -> accuracy
    for file in glob.glob(f'{args.folder}/*.json*'):
        print(f"Processing {file}")
        with open(file, "r") as f:
            data = json.load(f)
            for idx, item in tqdm(enumerate(data), total=len(data), desc='Processing'):
                x, y = extract_numbers(item["prompt"])
                generated_answer = item["generated"]
                
                if not args.use_scratchpad:
                    clean_match = re.search(r'[^#\n]*', generated_answer)
                    clean_answer = clean_match.group(0)
                    generated_number = extract_answer(clean_answer)
                    accuracy = (x * y == generated_number)
                else:
                    generated_answer = generated_answer.split("###", maxsplit=1)[0]
                    accuracy = compute_accuracy(x, y, generated_answer)

                accuracies[(len(str(x)), len(str(y)))].append(accuracy)
                overall_accuracy.append(accuracy)

    accuracy = mean(overall_accuracy) * 100
    print(f"Overall accuracy: {accuracy}")
    # Plot a heatmap of the accuracies
    matrix = np.zeros((5, 5))
    for key, value in accuracies.items():
        matrix[key[0] - 1, key[1] - 1] = mean(value)
    matrix = matrix * 100

    print(matrix)

    mask = np.logical_not(np.tril(np.ones_like(matrix, dtype=bool)))
    heatmap = sns.heatmap(matrix, mask=mask, annot=True, square=True, cmap='coolwarm', fmt=".2f")
    heatmap.set_xticklabels(range(1, 6))
    heatmap.set_yticklabels(range(1, 6))
    heatmap.set_xlabel('Number of digits in x')
    heatmap.set_ylabel('Number of digits in y')
    heatmap.set_title(f'Accuracy heatmap (%) (overall accuracy: {accuracy:.2f} %)')
    heatmap.get_figure().savefig(os.path.join(args.folder, 'accuracy_heatmap.png'))

if __name__ == '__main__':
    main()
                    
