# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class Superglue_CB(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "superglue-cb"

        self.task_type = "classification"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0: "entailment",
            1: "contradiction",
            2: "neutral",
        }

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            if datapoint["label"] == -1:
                continue
            lines.append(("premise: " + datapoint["premise"] + " [SEP] hypothesis: " + datapoint["hypothesis"], self.label[datapoint["label"]]))
            #lines.append(json.dumps({
            #    "input": "premise: " + datapoint["premise"] + " hypothesis: " + datapoint["hypothesis"],
            #    "output": self.label[datapoint["label"]],
            #    "options": list(self.label.values())}))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('super_glue', 'cb')

def main():
    dataset = Superglue_CB()
    full_data = dataset.load_dataset()
    print(full_data.keys())
    print(full_data["test"])
    train_data = dataset.map_hf_dataset_to_list(full_data, "train")
    # dev_data = dataset.map_hf_dataset_to_list(full_data, "validation")
    test_data = dataset.map_hf_dataset_to_list(full_data, "test")

    path = "../data/superglue-cb"
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

    train_json = format_data(train_data, "superglue-cb")
    # dev_json = format_data(dev_data, "superglue-cb")
    test_json = format_data(test_data, "superglue-cb")

    def save_jsonl(data, path):
        with open(path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

    save_jsonl(train_json, os.path.join(path, "superglue-cb_train.jsonl"))
    # save_jsonl(dev_json, os.path.join(path, "superglue-cb_dev.jsonl"))
    save_jsonl(test_json, os.path.join(path, "superglue-cb_test.jsonl"))

    print("Data saved successfully!")
if __name__ == "__main__":
    main()
