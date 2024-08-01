import argparse
import json
import os
import random
from functools import partial
from pathlib import Path
from typing import List
from tqdm import tqdm
from generate_graph_from_scratchpad import extract_numbers

ones = {
    0: "",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}
tens = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}

magnitudes = [
    "ones",
    "tens",
    "hundreds",
    "thousands",
    "ten-thousands",
    "hundred-thousands",
    "millions",
    "ten-millions",
    "hundred-millions",
    "billions",
]


def say_number(i):
    """
    Convert an integer in to it's word representation.

    say_number(i: integer) -> string
    """
    if i < 0:
        return _join("negative", _say_number_pos(-i))
    if i == 0:
        return "zero"
    return _say_number_pos(i)


def _say_number_pos(i):
    if i < 20:
        return ones[i]
    if i < 100:
        return _join(tens[i // 10], ones[i % 10])

    return _divide(i, 100, "hundred")


def _divide(dividend, divisor, magnitude):
    return _join(
        _say_number_pos(dividend // divisor),
        magnitude,
        _say_number_pos(dividend % divisor),
    )


def _join(*args):
    return " ".join(filter(bool, args))


def _say_magnitude(i: int):
    assert i < len(magnitudes), f"magnitude not supported: {i}, max is {len(magnitudes)}"
    return magnitudes[i]


def generate_number(i):
    assert i > 0
    return random.randint(10 ** (i - 1), 10 ** i - 1)


def digits(x: int) -> List[int]:
    n = x
    digits = []
    while n > 0:
        digits.append(n % 10)
        n //= 10

    return digits


def generate_prompt(x: int, y: int):
    digits_x = digits(x)
    digits_y = digits(y)

    question = f"What is {x} times {y}? \n\n###\n\n"
    steps = [f" Let's multiply {x} by the digit in the {_say_magnitude(0)} place of {y}, which is {digits_y[0]}.", ""]
    partial_products = []
    pp_symbols, pp_summaries = [], []
    step_count = 0
    for j, dy in enumerate(digits_y):
        carry_over = 0
        for i, dx in enumerate(digits_x):
            next_step = dx * dy + carry_over
            old_carry_over = carry_over

            if i < len(digits_x) - 1:
                residual = next_step % 10
                carry_over = next_step // 10
            else:
                residual = next_step
                carry_over = 0

            co_text = f" and carry over the {carry_over} to the next step" if carry_over > 0 else ""

            step_count += 1
            if old_carry_over == 0:
                step = (
                    f"{step_count}. Multiply {dy} by the digit in the {_say_magnitude(i)} place of {x}, which is {dx}. "
                    f"This gives {dx} x {dy} = {next_step}. "
                    f"Write down the result {residual}{co_text}."
                )
            else:
                step = (
                    f"{step_count}. Multiply {dy} by the digit in the {_say_magnitude(i)} place of {x}, which is {dx}. "
                    f"Add the carryover from the previous step to account for this. "
                    f"This gives ({dx} x {dy}) + {old_carry_over} = {next_step}. "
                    f"Write down the result {residual}{co_text}."
                )
            steps.append(step)

        step_count += 1
        pp_symbol = chr(ord("A") + j)
        pp_symbols.append(pp_symbol)
        pp = dy * x
        partial_products.append(pp)

        step = f"{step_count}. The partial product for this step is {pp_symbol}={pp} which is the concatenation of the digits we found in each step."
        steps.append(step)
        steps.append("")

        if j == 0:
            pp_summary = f"{pp_symbol}={pp} (from multiplication by {dy})"
        else:
            pp_summary = (
                f"{pp_symbol}={pp} (from multiplication by {dy} "
                f"but shifted {say_number(j)} place{'s' if j > 1 else ''} to the left, "
                f"so it becomes {pp * (10 ** j)})"
            )
        pp_summaries.append(pp_summary)

        if j < len(digits_y) - 1:
            steps.append(
                f"Now, let's multiply {x} by the digit in the {_say_magnitude(j + 1)} place of {y}, which is {digits_y[j + 1]}."
            )
            steps.append("")

    step_count += 1

    if len(pp_summaries) > 1:
        pp_summary_text = ", ".join(pp_summaries[:-1]) + " and " + pp_summaries[-1]
        pp_symbol_text = ", ".join(pp_symbols[:-1]) + " and " + pp_symbols[-1]
    else:
        pp_summary_text = pp_summaries[0]
        pp_symbol_text = pp_symbols[0]

    sum_pps_expanded = " + ".join([f"{pp} x {10 ** p}" for p, pp in enumerate(partial_products)])
    sum_pps = " + ".join([f"{pp * 10 ** p}" for p, pp in enumerate(partial_products)])
    
    prd = sum([pp * 10 ** p for p, pp in enumerate(partial_products)])
    assert prd == x * y, f"discrepancy in test case {x} x {y}: {prd} (scratchpad) != {x * y} (gold)"
    final_step = (
        f"Now, let's sum the {len(pp_symbols)} partial product{'s' if len(pp_symbols) > 1 else ''} {pp_symbol_text}, "
        f"and take into account the position of each digit: {pp_summary_text}. "
        f"The final answer is {sum_pps_expanded} = {sum_pps} = {prd}. ###"
    )

    final_prompt = "\n".join(steps) + f"\n{final_step}"
    return final_prompt, question, prd


def generate_prompt_alternative_function(x: int, y: int):
    digits_x = digits(x)
    digits_y = digits(y)

    question = f"What is {x} dax {y}? \n\n###\n\n"
    steps = [f" Let's dax {x} by the digit in the {_say_magnitude(0)} place of {y}, which is {digits_y[0]}.", ""]
    partial_products = []
    pp_symbols, pp_summaries = [], []
    step_count = 0

    for j, dy in enumerate(digits_y):
        residuals = []
        carry_over = 7  # modified version! Original: carry_over = 0
        steps.append("The initial carry over number is always 7.")
        for i, dx in enumerate(digits_x):
            next_step = dy + dx * carry_over  # modified version! Original: next_step = dy * dx + carry_over
            old_carry_over = carry_over

            if i < len(digits_x) - 1:
                residual = next_step % 10
                carry_over = next_step // 10
            else:
                residual = next_step
                carry_over = 0

            co_text = f" and carry over the {carry_over} to the next step" if carry_over > 0 else ""

            step_count += 1
            step = (
                f"{step_count}. Add {dy} to the result of multiplying the digit in the {_say_magnitude(i)} place of {x}, which is {dx}, by the carryover from the previous step. "
                f"This gives {dy} + ({dx} x {old_carry_over}) = {next_step}. "
                f"Write down the result {residual}{co_text}."
            )
            residuals.append(residual)
            steps.append(step)

        step_count += 1
        pp_symbol = chr(ord("A") + j)
        pp_symbols.append(pp_symbol)
        pp = int("".join(reversed([str(r) for r in residuals])))
        partial_products.append(pp)

        step = f"{step_count}. The partial dax for this step is {pp_symbol}={pp} which is the concatenation of the digits we found in each step."
        steps.append(step)
        steps.append("")

        if j == 0:
            pp_summary = f"{pp_symbol}={pp} (from dax by {dy})"
        else:
            pp_summary = (
                f"{pp_symbol}={pp} (from dax by {dy} "
                f"but shifted {say_number(j)} place{'s' if j > 1 else ''} to the left, "
                f"so it becomes {pp * (10 ** j)})"
            )
        pp_summaries.append(pp_summary)

        if j < len(digits_y) - 1:
            steps.append(
                f"Now, let's dax {x} by the digit in the {_say_magnitude(j + 1)} place of {y}, which is {digits_y[j + 1]}."
            )
            steps.append("")

    step_count += 1

    if len(pp_summaries) > 1:
        pp_summary_text = ", ".join(pp_summaries[:-1]) + " and " + pp_summaries[-1]
        pp_symbol_text = ", ".join(pp_symbols[:-1]) + " and " + pp_symbols[-1]
    else:
        pp_summary_text = pp_summaries[0]
        pp_symbol_text = pp_symbols[0]

    sum_pps_expanded = " + ".join([f"{pp} x {10 ** p}" for p, pp in enumerate(partial_products)])
    sum_pps = " + ".join([f"{pp * 10 ** p}" for p, pp in enumerate(partial_products)])

    prd = sum([pp * 10 ** p for p, pp in enumerate(partial_products)])
    final_step = (
        f"Now, let's sum the {len(pp_symbols)} partial product{'s' if len(pp_symbols) > 1 else ''} {pp_symbol_text}, "
        f"and take into account the position of each digit: {pp_summary_text}. "
        f"The final answer is {sum_pps_expanded} = {sum_pps} = {prd}. ###"
    )

    final_prompt = "\n".join(steps) + f"\n{final_step}"
    return final_prompt, question, prd


def sweep(start_a: int, end_a: int, start_b: int, end_b: int):
    for x in range(start_a, end_a):
        for y in range(start_b, end_b):
            yield x, y


def sample(digits_x: int, digits_y: int, size: int):
    for _ in range(size):
        yield generate_number(digits_x), generate_number(digits_y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_digit', type=int, default=4, help='maximum number of digits')
    parser.add_argument(
        "--finetune_folder", type=str, help="folder containing fine-tuning data"
    )
    parser.add_argument(
        "--use_alternative_function", action='store_true',
        help="Compute an alternative function to multiplication, defined to be computationally similar to "
             "multiplication"
    )
    parser.add_argument("--output_path", type=str, default=None, help="output path")
    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    digits = list(range(1, args.num_digit + 1))
    for k in digits:
        for p in digits[:k]:
            input_filepath = os.path.join(args.finetune_folder, f'{k}_by_{p}_digit_fine_tune.jsonl')
            assert os.path.exists(input_filepath), f"file {input_filepath} does not exist"

            output_filepath = output_path / f"scratchpad_{k}_by_{p}.jsonl"
            
            with open(input_filepath, "r") as fin:
                with open(output_filepath, "w") as fout:
                    for line in fin:
                        item = json.loads(line)
                        x, y = extract_numbers(item["prompt"])
                        item['scratchpad'], _, _ = generate_prompt(x, y)
                        fout.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    main()
