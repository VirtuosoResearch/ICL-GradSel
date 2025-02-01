import json
import torch
from transformers import GPT2Tokenizer, GPT2Model
from torch.utils.data import DataLoader, Dataset
import argparse
from tqdm import tqdm

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
            input_texts = str(batch["input"]) + (str(batch["label"]) if split=="test" else "")

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

    test_file_path = f"./data/{args.task}/{args.task}_test.jsonl"
    test_output_file = f"./features/{args.task}_test_features.json"
    dataset = ICLDataset(test_file_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    extract_features_lasttoken(model, tokenizer, dataloader, device, test_output_file, "test")

    val_file_path = f"./data/{args.task}/{args.task}_dev.jsonl"
    val_output_file = f"./features/{args.task}_val_features.json"
    dataset = ICLDataset(val_file_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    extract_features_lasttoken(model, tokenizer, dataloader, device, val_output_file, "val")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="superglue-cb", type=str)
    args = parser.parse_args()
    main(args)