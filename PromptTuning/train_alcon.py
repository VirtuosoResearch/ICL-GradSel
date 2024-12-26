import json
import torch
import argparse
from Alcon_QA import train_data, test_data
from torch.utils.data import DataLoader

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

from adapters import AutoAdapterModel, DoubleSeqBnConfig



def main(args):
    print("arguments".upper().center(80, "-"))
    print(args)
    print("-" * 80)
    device = torch.device(f"cuda:{args.device}")
    print("device: ", device)


    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    hf_key = args.model_key.replace("_", "-")
    tokenizer = AutoTokenizer.from_pretrained(hf_key)
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(hf_key)
    model_type = "decoder"
    append_eos = True
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    best_loss = float("inf")
    best_model_path = "./alcon_best_model.pt"
    max_seq_length = 256
    for epoch in range(args.epochs):
        print(f"Epoch{epoch+1}/{args.epochs}".center(80,"-"))
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_dataloader):
            inputs = tokenizer(batch["Question"], padding=True, truncation=True, max_length=max_seq_length, return_tensors="pt").to(device)
            labels = tokenizer(batch["Answer"], padding=True, truncation=True, max_length=max_seq_length, return_tensors="pt").input_ids.to(device)

            labels = torch.nn.functional.pad(labels, (0, max_seq_length - labels.size(1)), value=tokenizer.pad_token_id)
            optimizer.zero_grad()
            outputs = model(inputs.input_ids, attention_mask=inputs.attention_mask)
            logits = outputs.logits

            logits = logits[:, :max_seq_length, :]
            
            if logits.size(1) < max_seq_length:
                padding_size = max_seq_length - logits.size(1)
                logits = torch.nn.functional.pad(logits, (0, 0, 0, padding_size), value=tokenizer.pad_token_id)

            loss = criterion(logits.contiguous().view(-1, logits.size(-1)), labels.contiguous().view(-1))
            loss.backward()
            optimizer.step()

            running_loss+=loss.item()
            # print(f"loss.item(): {loss.item()}")
        running_loss/=len(train_dataloader)
        if running_loss < best_loss:
            best_loss = running_loss
            best_model = model.state_dict()
            torch.save(best_model, best_model_path)
        print(f"Trainning loss: {running_loss}")

    best_model = torch.load(best_model_path)
    model.load_state_dict(best_model)
    model.eval()

    test_loss = 0
    for batch in tqdm(test_dataloader):
        inputs = tokenizer(batch["Question"], padding=True, truncation=True, max_length=max_seq_length, return_tensors="pt").to(device)
        labels = tokenizer(batch["Answer"], padding=True, truncation=True, max_length=max_seq_length, return_tensors="pt").input_ids.to(device)
        labels = torch.nn.functional.pad(labels, (0, max_seq_length - labels.size(1)), value=tokenizer.pad_token_id)

        outputs = model(inputs.input_ids, attention_mask=inputs.attention_mask)
        logits = outputs.logits
        logits = logits[:, :max_seq_length, :]
        
        if logits.size(1) < max_seq_length:
            padding_size = max_seq_length - logits.size(1)
            logits = torch.nn.functional.pad(logits, (0, 0, 0, padding_size), value=tokenizer.pad_token_id)
        loss = criterion(logits.contiguous().view(-1, logits.size(-1)), labels.contiguous().view(-1))
        test_loss += loss.item()
    test_loss/=len(test_dataloader)
    print(f"test_loss: {test_loss}")

if __name__=="__main__":
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