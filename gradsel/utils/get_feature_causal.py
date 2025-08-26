import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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
        return {"input": input_text, "label": label}

def extract_features_lasttoken(model, tokenizer, dataloader, device, output_file, split):
    features = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_texts = [batch["input"][i] + (batch["label"][i] if batch["label"][i] is not None and split == "test" else "") 
                           for i in range(len(batch["input"]))]
            
            inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {key: val.to(device) for key, val in inputs.items()}

            outputs = model(**inputs, output_hidden_states=True)

            if outputs.hidden_states is None:
                raise ValueError("Model did not return hidden states. Ensure `output_hidden_states=True`.")

            hidden_states = outputs.hidden_states[-1]
            attention_mask = inputs["attention_mask"]  # (batch_size, seq_len)

            for i in range(hidden_states.size(0)):
                last_non_pad_idx = attention_mask[i].nonzero(as_tuple=True)[0].max().item()
                last_hidden_state = hidden_states[i, last_non_pad_idx, :] 
                features.append(last_hidden_state.cpu().numpy().tolist())

    with open(output_file, "w") as f:
        json.dump(features, f)

    print(f"Features saved to {output_file}")

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    
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
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B", type=str)
    args = parser.parse_args()
    main(args)