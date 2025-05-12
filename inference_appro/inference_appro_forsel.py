from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch
import json
import numpy as np
from tqdm import tqdm
import argparse
from thop import profile
import random

is_flops = True

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def get_embedding(model, input_ids, model_name):
    if "opt" in model_name:
        return model.model.decoder.embed_tokens(input_ids)
    return model.model.embed_tokens(input_ids)


def compute_loss(model, input_ids, attention_mask, label_token_id):
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    last_token_idx = attention_mask.sum(dim=1).item() - 2
    flops =0
    if is_flops:
        flops, param = profile(model, inputs=(input_ids,))
    return -logits[0, last_token_idx, label_token_id], flops


def estimate_loss_first_order(anchor_loss, anchor_grad, anchor_embed, dp_embed):
    delta = dp_embed - anchor_embed
    return anchor_loss + torch.sum(anchor_grad * delta).item()


def forward_selection_real(model, tokenizer, test_data, val_data, device, k, model_name):
    selected_indices = []
    prompt_prefix = ""
    val_losses = []
    total_flops =0
    for _ in tqdm(range(k), desc="Real FS steps"):
        best_loss, best_idx = float("inf"), -1
        for i, candidate in enumerate(tqdm(test_data, desc="Evaluating candidates", leave=False)):
            if i in selected_indices:
                continue
            prompt = prompt_prefix + f"Input: {candidate['input']} Output: {candidate['output']}\n"
            total_loss = 0.0
            for val_dp in tqdm(val_data, desc="Validating (real)", leave=False):
                text = prompt + f"Input: {val_dp['input']} Output: "
                inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                label_token_id = tokenizer(val_dp["output"], return_tensors="pt").input_ids[0][1].to(device)

                with torch.no_grad():
                    loss, flops = compute_loss(model, input_ids, attention_mask, label_token_id)
                    total_loss += loss.item()
                total_flops+= flops
            avg_loss = total_loss / len(val_data)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_idx = i
        selected_indices.append(best_idx)
        prompt_prefix += f"Input: {test_data[best_idx]['input']} Output: {test_data[best_idx]['output']}\n"

        round_loss = []
        for val_dp in tqdm(val_data, desc="Recording real loss", leave=False):
            text = prompt_prefix + f"Input: {val_dp['input']} Output: "
            inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            label_token_id = tokenizer(val_dp["output"], return_tensors="pt").input_ids[0][1].to(device)

            with torch.no_grad():
                loss, flops = compute_loss(model, input_ids, attention_mask, label_token_id)
                total_flops+= flops
            round_loss.append(loss.item())
        if _ ==k-1:
            val_losses.append(round_loss)
    return val_losses, selected_indices, total_flops

def compute_loss_embedding(model, embedding, attention_mask, label_token_id, input_ids):
    logits = model(inputs_embeds=embedding, attention_mask=attention_mask).logits
    last_token_idx = attention_mask.sum(dim=1).item() - 2
    flops =0
    if is_flops:
        flops, param = profile(model, inputs=(input_ids,))
    return -logits[0, last_token_idx, label_token_id], flops

def forward_selection_approx(model, tokenizer, test_data, val_data, device, k, selected_indices, model_name):
    selected_indices = []
    prompt_prefix = ""
    val_losses = []
    total_flops = 0

    for step in tqdm(range(k), desc="Approx FS steps"):
        candidate_indices = [i for i in range(len(test_data)) if i not in selected_indices]
        anchor_idx = random.choice(candidate_indices)
        anchor = test_data[anchor_idx]
        anchor_prompt = f"Input: {anchor['input']} Output: {anchor['output']}\n"
        
        val_dp = val_data[0]
        text = anchor_prompt + f"Input: {val_dp['input']} Output: "
        inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        label_token_id = tokenizer(val_dp["output"], return_tensors="pt").input_ids[0][1].to(device)

        emb = get_embedding(model, input_ids, model_name).detach()
        emb.requires_grad = True
        anchor_loss, flops = compute_loss_embedding(model, emb, attention_mask, label_token_id, input_ids)
        total_flops += flops
        anchor_grad = torch.autograd.grad(anchor_loss, emb)[0].detach()
        anchor_emb = emb.detach()

        best_loss, best_idx = float("inf"), -1
        for i in tqdm(candidate_indices, desc="Evaluating candidates", leave=False):
            candidate = test_data[i]
            prompt = prompt_prefix + f"Input: {candidate['input']} Output: {candidate['output']}\n"
            total_loss = 0.0
            for val_dp in val_data:
                text = prompt + f"Input: {val_dp['input']} Output: "
                inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
                input_ids = inputs["input_ids"].to(device)
                dp_emb = get_embedding(model, input_ids, model_name).detach()
                approx_loss = estimate_loss_first_order(anchor_loss.item(), anchor_grad, anchor_emb, dp_emb)
                total_loss += approx_loss
            avg_loss = total_loss / len(val_data)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_idx = i

        selected_indices.append(best_idx)
        prompt_prefix += f"Input: {test_data[best_idx]['input']} Output: {test_data[best_idx]['output']}\n"

        round_loss = []
        for val_dp in val_data:
            text = prompt_prefix + f"Input: {val_dp['input']} Output: "
            inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(device)
            dp_emb = get_embedding(model, input_ids, model_name).detach()
            approx_loss = estimate_loss_first_order(anchor_loss.item(), anchor_grad, anchor_emb, dp_emb)
            round_loss.append(approx_loss)

        if step == k - 1:
            val_losses.append(round_loss)

    return val_losses, total_flops



def compute_mse(real_losses, approx_losses):
    real = np.array(real_losses)
    approx = np.array(approx_losses)
    denom = np.maximum(np.abs(real), np.abs(approx)) + 1e-8 
    relative_error = np.abs(real - approx) / denom
    return np.mean(relative_error ** 2)


def main(args):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        llm_int8_threshold=6.0
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        quantization_config=bnb_config,
        device_map= {"": args.device}
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model=model.to(device)

    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.padding_side=="left":
        tokenizer.padding_side = 'right'
    
    print("tokenizer.padding_side: ",tokenizer.padding_side)

    model.resize_token_embeddings(len(tokenizer))
    tokenizer.pad_token = tokenizer.eos_token

    test_data = load_jsonl(f"./data/{args.task}/{args.task}_test.jsonl")
    val_data = load_jsonl(f"./data/{args.task}/{args.task}_dev.jsonl")
    test_data = test_data[:20]
    val_data = val_data[:15]
    print("Running real forward selection")
    model_name = args.model
    real_losses, selected_indices, flops1 = forward_selection_real(model, tokenizer, test_data, val_data, device, args.k, model_name)

    print("Running approximate forward selection")
    approx_losses, flops2 = forward_selection_approx(model, tokenizer, test_data, val_data, device, args.k, selected_indices, model_name)

    mse = compute_mse(real_losses, approx_losses)
    print("MSE between real and approximate forward selection losses:", mse)

    print(f"flops1: {flops1 / 1e9:.2f}, flops2: {flops2 / 1e9:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-llm-7b-chat")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--task", type=str, default="coin_flip")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--is_flops", default=False, action="store_true")
    args = parser.parse_args()
    main(args)
