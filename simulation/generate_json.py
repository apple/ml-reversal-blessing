#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import pandas as pd
import random
import argparse


def randomly_change_numbers(input_string, k):
    # Split the input string into a list of numbers
    numbers = input_string.split()

    # Determine how many numbers to change (at least 1, up to len(numbers))
    num_changes = k  # random.randint(1, len(numbers))

    # Randomly choose indices to modify
    indices_to_change = random.sample(range(len(numbers)), num_changes)

    for index in indices_to_change:
        # Replace the selected number with a new random digit (0-9) different from the original
        original = numbers[index]
        while original in ["*", "="]:
            index = random.sample(range(len(numbers)), 1)[0]
            original = numbers[index]
        new_number = str(random.choice([i for i in range(10) if str(i) != original]))
        numbers[index] = new_number

    # Join the modified list back into a string
    return " ".join(numbers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Generate test json file based on the eval type.", add_help=False
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["multiplication", "reverse_multiplication"],
        help="multiplication, reverse_multiplication",
    )
    parser.add_argument("--num-files", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_json(path_or_buf=f"data/{args.model_type}_test.json", lines=True)

    if args.model_type == "multiplication":
        answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        for n in range(10):
            df["question"] = df["text"].apply(lambda x: x.split("=")[0] + "= ")
            # df['number1'] = df['text'].apply(lambda x: x.split("=")[1].split("*")[0].strip())
            # df['number2'] = df['text'].apply(lambda x: x.split("=")[1].split("*")[1].strip())
            df["answer"] = df["text"].apply(lambda x: x.split("=")[1])
            for i in range(3):
                df[f"choice{i}"] = df["answer"].apply(
                    lambda x: randomly_change_numbers(x, 1)
                )

            # df['question'] = df['question'] + df['number1'] + " * "
            # df['answer'] = df['number2']
            choices = []
            answers = []
            for i in range(len(df)):
                text_list = [
                    df["answer"][i].strip(),
                    df["choice0"][i],
                    df["choice1"][i],
                    df["choice2"][i],
                ]
                random.shuffle(text_list)
                choices.append({"text": text_list, "label": ["A", "B", "C", "D"]})
                answers.append(answer_map[text_list.index(df["answer"][i].strip())])
            df["choices"] = choices
            df["answerKey"] = answers
            df = df.drop(columns=["answer", "choice0", "choice1", "choice2"])
            df.to_json(f"data/test{n}.json", orient="records", lines=True)

    elif args.model_type == "reverse_multiplication":
        answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        for n in range(args.num_files):
            df["question"] = df["text"].apply(lambda x: x.split("=")[0] + "= ")
            df["answer"] = df["text"].apply(lambda x: x.split("=")[1])
            for i in range(3):
                df[f"choice{i}"] = df["answer"].apply(
                    lambda x: randomly_change_numbers(x, 1)
                )

            choices = []
            answers = []
            for i in range(len(df)):
                text_list = [
                    df["answer"][i].strip(),
                    df["choice0"][i],
                    df["choice1"][i],
                    df["choice2"][i],
                ]
                random.shuffle(text_list)
                choices.append({"text": text_list, "label": ["A", "B", "C", "D"]})
                answers.append(answer_map[text_list.index(df["answer"][i].strip())])
            df["choices"] = choices
            df["answerKey"] = answers
            df = df.drop(columns=["answer", "choice0", "choice1", "choice2"])
            df.to_json(f"data/test{n}.json", orient="records", lines=True)
