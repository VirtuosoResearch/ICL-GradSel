# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datasets
import numpy as np
import json

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class Glue_RTE(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "glue-rte"

        self.task_type = "classification"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0: "entailment",
            1: "not_entailment",
        }

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            lines.append(("sentence 1: " + datapoint["sentence1"] + " [SEP] sentence 2: " + datapoint["sentence2"], self.label[datapoint["label"]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('glue', 'rte')

def main():
    dataset = Glue_RTE()
    full_data = dataset.load_dataset()

    train_data = dataset.map_hf_dataset_to_list(full_data, "train")
    # dev_data = dataset.map_hf_dataset_to_list(full_data, "validation")
    # test_data = dataset.map_hf_dataset_to_list(full_data, "test")

    path = "../data/glue-rte"
    os.makedirs(path, exist_ok=True)

    def format_data(data, task_name):
        formatted = []
        options = ["entailment", "not_entailment"]
        for input_text, output in data:
            formatted.append({
                "task": task_name,
                "input": input_text,
                "output": output,
                "options": options,
            })
        return formatted

    train_json = format_data(train_data, "glue-rte")
    # dev_json = format_data(dev_data, "glue-rte")
    # test_json = format_data(test_data, "glue-rte")

    def save_jsonl(data, path):
        with open(path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

    save_jsonl(train_json, os.path.join(path, "glue-rte_train.jsonl"))
    # save_jsonl(dev_json, os.path.join(path, "glue-rte_dev.jsonl"))
    # save_jsonl(test_json, os.path.join(path, "glue-rte_test.jsonl"))

if __name__ == "__main__":
    main()