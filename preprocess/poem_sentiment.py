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

class PoemSentiment(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "poem_sentiment"
        self.task_type = "classification"
        self.license = "unknown"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0:"negative",
            1:"positive",
            2:"no_impact",
            #3:"mixed", # there is no `mixed` on the test set
        }

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            if datapoint["label"]==3:
                assert split_name!="test"
                continue
            lines.append((datapoint["verse_text"], self.label[datapoint["label"]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('poem_sentiment')

def main():
    dataset = PoemSentiment()
    full_data = dataset.load_dataset()

    train_data = dataset.map_hf_dataset_to_list(full_data, "train")
    dev_data = dataset.map_hf_dataset_to_list(full_data, "validation")
    test_data = dataset.map_hf_dataset_to_list(full_data, "test")

    path = "../data/poem_sentiment"
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

    train_json = format_data(train_data, "poem_sentiment")
    dev_json = format_data(dev_data, "poem_sentiment")
    test_json = format_data(test_data, "poem_sentiment")

    def save_jsonl(data, path):
        with open(path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

    save_jsonl(train_json, os.path.join(path, "poem_sentiment_train.jsonl"))
    save_jsonl(dev_json, os.path.join(path, "poem_sentiment_dev.jsonl"))
    save_jsonl(test_json, os.path.join(path, "poem_sentiment_test.jsonl"))

    # print("Data saved successfully!")

if __name__ == "__main__":
    main()
