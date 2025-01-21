import os
import datasets
import numpy as np
import argparse
import json

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class Sick(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "sick"

        self.task_type = "classification"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0: "entailment",
            1: "neutral",
            2: "contradiction",
        }

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            lines.append(("sentence 1: " + datapoint["sentence_A"] + " [SEP] sentence 2: " + datapoint["sentence_B"], self.label[datapoint["label"]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('sick')

def main(args):
    dataset = Sick()
    full_data = dataset.load_dataset()

    train_data = dataset.map_hf_dataset_to_list(full_data, "train")
    dev_data = dataset.map_hf_dataset_to_list(full_data, "validation")
    test_data = dataset.map_hf_dataset_to_list(full_data, "test")

    path = "../data/sick"
    os.makedirs(path, exist_ok=True)

    def format_data(data, task_name):
        formatted = []
        options = ["contradiction", "entailment", "neutral"]
        for input_text, output in data:
            formatted.append({
                "task": task_name,
                "input": input_text,
                "output": output,
                "options": options,
            })
        return formatted

    train_json = format_data(train_data, "sick")
    dev_json = format_data(dev_data, "sick")
    test_json = format_data(test_data, "sick")

    def save_jsonl(data, path):
        with open(path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

    save_jsonl(train_json, os.path.join(path, "sick_train.jsonl"))
    save_jsonl(dev_json, os.path.join(path, "sick_dev.jsonl"))
    save_jsonl(test_json, os.path.join(path, "sick_test.jsonl"))

    # print("Data saved successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=16)
    args = parser.parse_args()
    main(args)
