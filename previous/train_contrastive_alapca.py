import argparse
import logging
import os
import json
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from adapters import AutoAdapterModel, DoubleSeqBnConfig

logging.basicConfig(level=logging.INFO)
torch.set_float32_matmul_precision("high")

class CustomQAData(Dataset):
    """
    Example dataset that handles your JSON structure:
    This example:
     1. Creates (input, label) pairs from Neg, Pos, and also from Question, Answer.
     2. Tokenizes each pair.
    """

    def __init__(self, data_file, tokenizer, max_length=256, mode="train"):
        super().__init__()
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.samples = []

        # Load data
        with open(self.data_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Build (input, output) pairs. This is just an example:
        # You can be more selective: for training, you might only want Pos/Neg pairs;
        # for QA, you might only want Question/Answer pairs, etc.
        for item in data:
            # Negative sample
            neg_input = item["Neg"]["input"]
            neg_output = item["Neg"]["output"]
            self.samples.append((neg_input, neg_output))

            # Positive sample
            pos_input = item["Pos"]["input"]
            pos_output = item["Pos"]["output"]
            self.samples.append((pos_input, pos_output))

            # QA sample
            qa_input = item["Question"]
            qa_output = item["Answer"]
            self.samples.append((qa_input, qa_output))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text_input, text_output = self.samples[idx]

        # Tokenize
        # For a causal LM (GPT-style), you often just concatenate prompt+answer in one sequence.
        # For seq2seq, you have input_ids -> encoder, labels -> decoder.
        # Adjust as needed for your specific model type.
        enc = self.tokenizer(
            text_input,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        with self.tokenizer.as_target_tokenizer():
            dec = self.tokenizer(
                text_output,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

        # Flatten to make each key a 1D tensor (rather than [1, seq_len])
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": dec["input_ids"].squeeze(0),
        }

        return item


def train_step(args, model, loss_fn, batch, optimizer):
    model.train()
    optimizer.zero_grad()

    outputs = model(
        input_ids = batch["input_ids"],
        attention_mask = batch["attemtion_mask"],
        labels = batch["labels"]
    )

    loss = outputs.loss
    loss.backward()
    optimizer.step()

    return loss.item()

def test_step(model, data_loader, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    correct_predictions = 0

    for batch in tqdm(data_loader, desc="Testing"):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        total_loss += loss.item()


        logits = outputs.logits  # (batch_size, seq_len, vocab_size)
        predictions = torch.argmax(logits, dim=-1)  # (batch_size, seq_len)

        labels = batch["labels"]
        mask = labels != -100
        correct_predictions += (predictions[mask] == labels[mask]).sum().item()
        total_samples += mask.sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    return avg_loss, accuracy


def main(args):
    print("arguments".upper().center(80, "-"))
    print(args)
    print("-" * 80)
    device = torch.device(f"cuda:{args.devices[0]}")
    print("device: ", device)

    model_key = args.model_key.replace("/", "-")
    if "gpt" in args.model_key or "Llama" in model_key or "bloomz" in model_key or "gemma" in model_key or "Mistral" in model_key:
        hf_key = args.model_key.replace("_", "-")
        tokenizer = AutoTokenizer.from_pretrained(hf_key)
        tokenizer.padding_side = 'right'
        model = AutoModelForCausalLM.from_pretrained(hf_key)
        model_type = "decoder"
        append_eos = True
    elif "flan" in model_key:
        hf_key = "google/{}".format(model_key.replace("_", "-"))
        model = AutoModelForSeq2SeqLM.from_pretrained(hf_key)
        tokenizer = AutoTokenizer.from_pretrained(hf_key, model_max_length=512)
        model_type = "encoder_decoder"
        append_eos = False  # T5 tokenizers already append eos
    else:
        raise NotImplementedError(args.model_key)

    if args.train_adapter:
        model = AutoAdapterModel.from_pretrained(hf_key)

        reduction = args.reduction_factor
        bottleneck_config = DoubleSeqBnConfig(
            mh_adapter=True,
            output_adapter=True,
            reduction_factor=reduction,
            non_linearity="relu"
        )

        model.add_adapter(adapter_name="seq_bn", config=bottleneck_config)

        model.to(device)

        for name, param in model.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False

        model.set_active_adapters("seq_bn")

        print("-" * 20, "Bottleneck_Adapter", "-" * 20)
        trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params_count = sum(p.numel() for p in model.parameters())

        print(f"Trainable parameters: {trainable_params_count} || All parameters: {all_params_count} || ratio: {trainable_params_count / all_params_count}")
        
        print("model.device: ", model.device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ############################################################################
    #                       Load dataset & split                                #
    ############################################################################
    full_dataset = CustomQAData(
        data_file="./data.json",
        tokenizer=tokenizer,
        max_length=args.max_length,
        mode="train"
    )
    total_len = len(full_dataset)
    val_len = int(0.2 * total_len)  # 80% train, 20% valid
    train_len = total_len - val_len
    train_data, val_data = random_split(full_dataset, [train_len, val_len])

    logging.info(f"Full dataset size: {total_len}")
    logging.info(f"Train set size: {train_len}")
    logging.info(f"Val set size: {val_len}")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)


    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for batch in pbar:
                batch = {k: v.to(device) for k, v in batch.items()}

                loss_val = train_step(args, model, loss_fn, batch, optimizer)
                total_train_loss += loss_val
                pbar.set_postfix({"loss": loss_val})

        avg_train_loss = total_train_loss / len(train_loader)

        # Validate
        val_loss, val_accuracy = test_step(model, val_loader, device)
        logging.info(
            f"[Epoch {epoch+1}] train_loss={avg_train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}"
        )

        # Save best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    # If desired, load the best model state back
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    logging.info("Training complete.")


    test_loss, test_accuracy = test_step(model, val_loader, device)
    logging.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", type=str, default="google/flan-t5-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    main(args)
