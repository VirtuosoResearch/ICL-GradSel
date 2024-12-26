import json
import torch
import argparse
from Alcon_QA import train_data, test_data
from torch.utils.data import DataLoader

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from evaluate import load
from adapters import AutoAdapterModel, DoubleSeqBnConfig

def main(args):
    print("arguments".upper().center(80, "-"))
    print(args)
    print("-" * 80)
    device = torch.device(f"cuda:{args.device}")
    print("device: ", device)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    hf_key = args.model_key.replace("_", "-")
    tokenizer = AutoTokenizer.from_pretrained(hf_key)
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(hf_key)
    model_type = "decoder"
    append_eos = True

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id


    if args.train_adapter:
        model = AutoAdapterModel.from_pretrained(args.model_key)

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rouge = load("rouge")

    best_loss = float("inf")
    best_model_path = "./alcon_best_model.pt"
    max_seq_length = 256

    for epoch in range(args.epochs):
        print(f"Epoch{epoch + 1}/{args.epochs}".center(80, "-"))
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_dataloader):
            inputs = tokenizer(batch["Question"], padding=True, truncation=True, max_length=max_seq_length, return_tensors="pt").to(device)
            labels = tokenizer(batch["Answer"], padding=True, truncation=True, max_length=max_seq_length, return_tensors="pt").input_ids.to(device)

            labels = torch.nn.functional.pad(labels, (0, max_seq_length - labels.size(1)), value=tokenizer.pad_token_id)
            optimizer.zero_grad()
            outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=max_seq_length*2)

            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            rouge_scores = rouge.compute(predictions=decoded_preds, references=decoded_labels)
            rouge_loss = 1 - rouge_scores['rougeL']

            print(f"rouge_loss: {rouge_loss}")
            torch.autograd.backward(torch.tensor(rouge_loss, requires_grad=True))
            optimizer.step()

            running_loss += rouge_loss

        running_loss /= len(train_dataloader)
        if running_loss < best_loss:
            best_loss = running_loss
            best_model = model.state_dict()
            torch.save(best_model, best_model_path)
        print(f"Training loss: {running_loss}")

    best_model = torch.load(best_model_path)
    model.load_state_dict(best_model)
    model.eval()

    predictions = []
    references = []
    for batch in tqdm(test_dataloader):
        inputs = tokenizer(batch["Question"], padding=True, truncation=True, max_length=max_seq_length, return_tensors="pt").to(device)
        labels = batch["Answer"]

        with torch.no_grad():
            outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=max_seq_length)

        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_labels = [label.strip() for label in labels]

        predictions.extend(decoded_preds)
        references.extend(decoded_labels)

    results = rouge.compute(predictions=predictions, references=references)
    print("ROUGE scores:", results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--train_adapter", action="store_true")
    parser.add_argument("--reduction_factor", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()
    main(args)
