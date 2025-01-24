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

class MedicalQuestionPairs(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "medical_questions_pairs"
        self.task_type = "classification"
        self.license = "unknown"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0: "Similar",
            1: "Dissimilar",
        }

    def get_train_test_lines(self, dataset):
        # only train set, manually split 20% data as test
        map_hf_dataset_to_list = self.get_map_hf_dataset_to_list()
        if map_hf_dataset_to_list is None:
            map_hf_dataset_to_list = self.map_hf_dataset_to_list
        lines = map_hf_dataset_to_list(dataset, "train")

        np.random.seed(42)
        np.random.shuffle(lines)

        n = len(lines)

        train_lines = lines[:int(0.8*n)]
        test_lines = lines[int(0.8*n):]

        return train_lines, test_lines


    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            lines.append(("question 1: " + datapoint["question_1"] + " [SEP] question 2: " + datapoint["question_2"], self.label[datapoint["label"]]))

        return lines

    def load_dataset(self):
        return datasets.load_dataset('medical_questions_pairs')

def main():
    dataset = MedicalQuestionPairs()
    full_data = dataset.load_dataset()
    print(full_data.keys())
    train_data = dataset.map_hf_dataset_to_list(full_data, "train")
    # dev_data = dataset.map_hf_dataset_to_list(full_data, "validation")
    # test_data = dataset.map_hf_dataset_to_list(full_data, "test")

    path = "../data/medical_questions_pairs"
    os.makedirs(path, exist_ok=True)

    def format_data(data, task_name):
        formatted = []
        options = ["Similar", "Dissimilar"]
        for input_text, output in data:
            formatted.append({
                "task": task_name,
                "input": input_text,
                "output": output,
                "options": options,
            })
        return formatted

    train_json = format_data(train_data, "medical_questions_pairs")
    # dev_json = format_data(dev_data, "medical_questions_pairs")
    # test_json = format_data(test_data, "medical_questions_pairs")

    def save_jsonl(data, path):
        with open(path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

    save_jsonl(train_json, os.path.join(path, "medical_questions_pairs_test.jsonl"))
    # save_jsonl(dev_json, os.path.join(path, "medical_questions_pairs_dev.jsonl"))
    # save_jsonl(test_json, os.path.join(path, "medical_questions_pairs_test.jsonl"))

    print("Data saved successfully!")
if __name__ == "__main__":
    main()
