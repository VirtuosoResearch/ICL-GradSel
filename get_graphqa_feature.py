import json
import torch
from transformers import GPT2Tokenizer, GPT2Model
from torch.utils.data import DataLoader, Dataset
import argparse
from tqdm import tqdm
from datasets import load_dataset

def transform_graphqa(example):
    return {
        "task": "graphqa_cycle_check",
        "input": example.get("input", ""),
        "output": example.get("output", ""),
        "options": ["Yes, there is a cycle.", "No, there is no cycle."]
    }

class convert_format:

    def __call__(self, examples):
        examples["input"] = examples["question"][:]
        examples["output"] = examples["answer"][:]
        return examples
    
class ICLDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, "r") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item["input"]
        label = item.get("output", None) 
        # print("input: ",input_text, " label : ", label)
        return {"input": input_text, "label": label}

def extract_features_lasttoken(model, tokenizer, dataloader, device, output_file, split):
    features = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_texts = str(batch["input"])

            inputs = tokenizer(
                input_texts, return_tensors="pt", padding=True, truncation=True, max_length=128
            )
            inputs = {key: val.to(device) for key, val in inputs.items()}
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
            attention_mask = inputs["attention_mask"]  # (batch_size, seq_len)

            for i in range(hidden_states.size(0)):
                last_non_pad_idx = attention_mask[i].nonzero(as_tuple=True)[0].max().item()
                last_hidden_state = hidden_states[i, last_non_pad_idx, :] 

                features.append(last_hidden_state.cpu().numpy().tolist())
                # print(len(last_hidden_state.cpu().numpy().tolist()))

    print(len(features))
    with open(output_file, "w") as f:
        json.dump(features, f)

    print(f"Features saved to {output_file}")


def main(args):

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    model = GPT2Model.from_pretrained("gpt2-large")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenizer.pad_token = tokenizer.eos_token

    task_name = "cycle_check"
    prompt_style = "zero_shot"
    # Split the dataset into train and validation
    task_file_dir = "data/graphqa/{}_{}_er_train.json".format(task_name, prompt_style)
    train_dataset = load_dataset("json", data_files=task_file_dir)['train']
    
    # fileter out the examples by the text encoder
    column_names = train_dataset.column_names
    # convert the input and output format
    train_dataset = train_dataset.map(convert_format(), batched=True, remove_columns=column_names)
    print(train_dataset.features)

    task_file_dir = "data/graphqa/{}_{}_er_valid.json".format(task_name, prompt_style)
    eval_dataset = load_dataset("json", data_files=task_file_dir)['train']
    # fileter out the examples by the text encoder
    column_names = eval_dataset.column_names
    # convert the input and output format
    eval_dataset = eval_dataset.map(convert_format(), batched=True, remove_columns=column_names)
    
    task_file_dir = "data/graphqa/{}_{}_er_test.json".format(task_name, prompt_style)
    predict_dataset = load_dataset("json", data_files=task_file_dir)['train']
    # fileter out the examples by the text encoder
    column_names = predict_dataset.column_names
    # convert the input and output format
    predict_dataset = predict_dataset.map(convert_format(), batched=True, remove_columns=column_names)

    train_data = train_dataset.map(transform_graphqa)
    test_data = eval_dataset.map(transform_graphqa)
    val_data = predict_dataset.map(transform_graphqa)

    train_output_file = f"./features/graphqa_cycle_check_train_features.json"
    dataloader = DataLoader(train_data, batch_size=1, shuffle=False)
    extract_features_lasttoken(model, tokenizer, dataloader, device, train_output_file, "train")

    test_output_file = f"./features/graphqa_cycle_check_test_features.json"
    dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    extract_features_lasttoken(model, tokenizer, dataloader, device, test_output_file, "test")

    val_output_file = f"./features/graphqa_cycle_check_val_features.json"
    dataloader = DataLoader(val_data, batch_size=1, shuffle=False)
    extract_features_lasttoken(model, tokenizer, dataloader, device, val_output_file, "val")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="superglue-cb", type=str)
    args = parser.parse_args()
    main(args)