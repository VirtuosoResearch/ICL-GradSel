import torch
import numpy as np
from tqdm import tqdm
import random
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import argparse
import json


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def sample_unique_combinations(all_indices, k, num_combinations, seed=42):
    random.seed(seed)
    seen = set()
    combinations = []
    max_trials = num_combinations * 10
    trials = 0
    while len(combinations) < num_combinations and trials < max_trials:
        comb = tuple(sorted(random.sample(all_indices, k)))
        if comb not in seen:
            seen.add(comb)
            combinations.append(comb)
        trials += 1
    return combinations


def get_embedding(model, input_ids, model_name):
    if "opt" in model_name:
        return model.model.decoder.embed_tokens(input_ids)
    return model.model.embed_tokens(input_ids)


def compute_loss(model, input_ids, attention_mask, label_token_id):
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    last_token_idx = attention_mask.sum(dim=1).item() - 2
    return -logits[0, last_token_idx, label_token_id], 0


def compute_loss_embedding(model, embedding, attention_mask, label_token_id, input_ids):
    logits = model(inputs_embeds=embedding, attention_mask=attention_mask).logits
    last_token_idx = attention_mask.sum(dim=1).item() - 2
    return -logits[0, last_token_idx, label_token_id], 0


def estimate_loss_first_order(anchor_loss, anchor_grad, anchor_embed, dp_embed):
    delta = dp_embed - anchor_embed
    return anchor_loss + torch.sum(anchor_grad * delta).item()


def score_points_from_combinations(loss_list, combinations, k):
    point_scores = defaultdict(list)
    for losses, comb in zip(loss_list, combinations):
        comb_loss = np.mean(losses)
        for idx in comb:
            point_scores[idx].append(comb_loss)

    avg_scores = [(idx, np.mean(scores)) for idx, scores in point_scores.items()]
    avg_scores.sort(key=lambda x: x[1])  # sort by ascending loss
    selected_indices = [idx for idx, _ in avg_scores[:k]]
    return selected_indices


def compute_val_loss_from_prompt(selected_indices, test_data, val_data, model, tokenizer, device, model_name):
    prompt_prefix = "".join([
        f"Input: {test_data[idx]['input']} Output: {test_data[idx]['output']}\n" for idx in selected_indices
    ])

    round_loss = []
    for val_dp in val_data:
        text = prompt_prefix + f"Input: {val_dp['input']} Output:"
        inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        label_token_id = tokenizer(val_dp["output"], return_tensors="pt").input_ids[0][1].to(device)

        with torch.no_grad():
            loss, _ = compute_loss(model, input_ids, attention_mask, label_token_id)
        round_loss.append(loss.item())

    return round_loss


def random_ensemble_real(model, tokenizer, test_data, val_data, device, k, model_name, num_combinations=50):
    all_indices = list(range(len(test_data)))
    combinations = sample_unique_combinations(all_indices, k, num_combinations)
    combination_losses = []

    for comb in tqdm(combinations, desc="Real inference for random ensemble"):
        prompt = "".join([
            f"Input: {test_data[idx]['input']} Output: {test_data[idx]['output']}\n" for idx in comb
        ])
        round_loss = []
        for val_dp in val_data:
            text = prompt + f"Input: {val_dp['input']} Output: "
            inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            label_token_id = tokenizer(val_dp["output"], return_tensors="pt").input_ids[0][1].to(device)

            with torch.no_grad():
                loss, _ = compute_loss(model, input_ids, attention_mask, label_token_id)
            round_loss.append(loss.item())
        combination_losses.append(round_loss)

    selected_indices = score_points_from_combinations(combination_losses, combinations, k)
    final_loss = compute_val_loss_from_prompt(selected_indices, test_data, val_data, model, tokenizer, device, model_name)

    return final_loss, selected_indices


def random_ensemble_approx(model, tokenizer, test_data, val_data, device, k, model_name, num_combinations=50):
    all_indices = list(range(len(test_data)))
    combinations = sample_unique_combinations(all_indices, k, num_combinations)
    anchor_combs = random.sample(combinations, 1)
    anchor_info = {}
    approx_losses_all = []

    for anchor in anchor_combs:
        anchor_prompt = "".join([
            f"Input: {test_data[idx]['input']} Output: {test_data[idx]['output']}\n" for idx in anchor
        ])
        val_info = []
        for val_dp in val_data:
            text = anchor_prompt + f"Input: {val_dp['input']} Output: "
            inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            label_token_id = tokenizer(val_dp["output"], return_tensors="pt").input_ids[0][1].to(device)

            emb = get_embedding(model, input_ids, model_name).detach()
            emb.requires_grad = True
            anchor_loss, _ = compute_loss_embedding(model, emb, attention_mask, label_token_id, input_ids)
            anchor_grad = torch.autograd.grad(anchor_loss, emb)[0].detach()
            anchor_emb = emb.detach()

            val_info.append({
                "anchor_loss": anchor_loss.item(),
                "anchor_grad": anchor_grad,
                "anchor_emb": anchor_emb,
            })

        anchor_info[anchor] = (anchor_prompt, val_info)

    for comb in tqdm(combinations, desc="Taylor estimation for random ensemble"):
        if comb in anchor_combs:
            continue

        best_anchor = max(anchor_combs, key=lambda a: len(set(a) & set(comb)))
        anchor_prompt, anchor_val_info = anchor_info[best_anchor]

        target_prompt = "".join([
            f"Input: {test_data[idx]['input']} Output: {test_data[idx]['output']}\n" for idx in comb
        ])
        round_loss = []
        for j, val_dp in enumerate(val_data):
            text = target_prompt + f"Input: {val_dp['input']} Output: "
            inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(device)
            dp_emb = get_embedding(model, input_ids, model_name).detach()

            anchor_loss_j = anchor_val_info[j]["anchor_loss"]
            anchor_grad_j = anchor_val_info[j]["anchor_grad"]
            anchor_emb_j = anchor_val_info[j]["anchor_emb"]

            approx_loss = estimate_loss_first_order(anchor_loss_j, anchor_grad_j, anchor_emb_j, dp_emb)
            round_loss.append(approx_loss)
        approx_losses_all.append(round_loss)

    selected_indices = score_points_from_combinations(approx_losses_all, combinations, k)
    final_loss = compute_val_loss_from_prompt(selected_indices, test_data, val_data, model, tokenizer, device, model_name)

    return final_loss, selected_indices


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
        device_map={"": args.device}
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.padding_side == "left":
        tokenizer.padding_side = "right"
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    test_data = load_jsonl(f"./data/{args.task}/{args.task}_test.jsonl")[:50]
    val_data = load_jsonl(f"./data/{args.task}/{args.task}_dev.jsonl")[:15]
    model_name = args.model

    print("Running real random ensemble")
    real_loss, _ = random_ensemble_real(model, tokenizer, test_data, val_data, device, args.k, model_name)

    print("Running approximate random ensemble")
    approx_loss, _ = random_ensemble_approx(model, tokenizer, test_data, val_data, device, args.k, model_name)

    mse = compute_mse(real_loss, approx_loss)
    print("MSE between real and approx ensemble loss:", mse)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-llm-7b-chat")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--task", type=str, default="coin_flip")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    main(args)
