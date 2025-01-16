import json
import torch
from transformers import GPT2Tokenizer, GPT2Model
from torch.utils.data import DataLoader, Dataset


class SuperGlueCBDataset(Dataset):
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


def extract_features_avg(model, tokenizer, dataloader, device, output_file):
    features = []

    with torch.no_grad():
        for batch in dataloader:
            input_texts = batch["input"]
            labels = batch["label"]

            inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {key: val.to(device) for key, val in inputs.items()}

            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state.mean(dim=1)

            for feature, label in zip(hidden_states.cpu().numpy(), labels):
                # print(len(feature.tolist()))
                features.append(feature.tolist())
        print(len(features))

    with open(output_file, "w") as f:
        json.dump(features, f)

    print(f"Features saved to {output_file}")

def extract_features_avg_nonpad(model, tokenizer, dataloader, device, output_file):
    features = []

    with torch.no_grad():
        for batch in dataloader:
            input_texts = batch["input"]

            inputs = tokenizer(
                input_texts, return_tensors="pt", padding=True, truncation=True, max_length=128
            )
            inputs = {key: val.to(device) for key, val in inputs.items()}

            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
            attention_mask = inputs["attention_mask"]  # (batch_size, seq_len)

            for i in range(hidden_states.size(0)):

                non_pad_tokens = attention_mask[i].nonzero(as_tuple=True)[0]  # indices of non-pad tokens
                valid_hidden_states = hidden_states[i, non_pad_tokens, :]  # get non-pad embeddings

                feature = valid_hidden_states.mean(dim=0) 
                features.append(feature.cpu().numpy().tolist())
                # print(len(feature.cpu().numpy().tolist()))
    print(len(features))

    with open(output_file, "w") as f:
        json.dump(features, f)

    print(f"Features saved to {output_file}")

def extract_features_lasttoken(model, tokenizer, dataloader, device, output_file):
    features = []

    with torch.no_grad():
        for batch in dataloader:
            input_texts = batch["input"]

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


def main():
    file_path = "./data/superglue-cb/superglue-cb_test.jsonl"
    output_file = "./features/superglue-cb_features.json"

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    model = GPT2Model.from_pretrained("gpt2-large")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenizer.pad_token = tokenizer.eos_token

    dataset = SuperGlueCBDataset(file_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: {
        "input": [d["input"] for d in x],
        "label": [d["label"] for d in x]
    })

    extract_features_lasttoken(model, tokenizer, dataloader, device, output_file)



if __name__ == "__main__":
    main()
