import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2LMHeadModel, GPT2Tokenizer, OPTForCausalLM
from utils.data import load_data
from tqdm import tqdm
import argparse
import random

# "meta-llama/Llama-3.2-1B"
def main(args):
    dataset_name = args.task
    model_name = args.model
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    if "gpt2" in model_name:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
    elif "opt" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = OPTForCausalLM.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    model=model.to(device, dtype=torch.bfloat16)

    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

    test_data = load_data(None, "test", 3, seed=42, config_split="test",
                        datasets=[dataset_name], is_null=False)

    if len(test_data)>1000: test_data = test_data[:1000]

    init = ""
    random.seed(args.seed)
    random_numbers = random.sample(range(len(test_data)), args.k)
    for i in random_numbers:
        init+="Input: " + test_data[i]["input"]+" Output: "+test_data[i]["output"]+"\n"


    anchor_dp = test_data[0]

    anchor_losses = {}
    anchor_gradients = {}

    for option in anchor_dp["options"]:
        input_tokens =  init+"Input: " + anchor_dp["input"] + " Label:"
        
        tokens_input = tokenizer(input_tokens, return_tensors="pt", padding="max_length", truncation=True, max_length=100*(args.k+1)).input_ids.to(device)
        tokens_output = tokenizer(option, return_tensors="pt").input_ids[0][-1].to(device)

        # last_token_idx = tokens_input.ne(tokenizer.pad_token_id).sum(dim=1) - 1
        # print("last_token_idx : ", last_token_idx)

        with torch.no_grad():
            if "gpt2" in model_name: embedding_input = model.transformer.wte(tokens_input)
            elif "opt" in model_name: embedding_input = model.model.decoder.embed_tokens(tokens_input)
            else: embedding_input = model.model.embed_tokens(tokens_input)
        
        anchor_embedding_input = embedding_input.clone().detach()

        embedding_input.requires_grad = True

        print(f"embedding_input shape: {embedding_input.shape}")
        print(f"Max index in inputs_embeds: {embedding_input.max()}")

        embedding_input = embedding_input.to(torch.float32)  
        output_logits = model(inputs_embeds=embedding_input).logits
        last_token_idx = tokens_input.shape[1] - 1
        log_probs = F.log_softmax(output_logits[0, last_token_idx, :], dim=-1)

        loss = -log_probs[tokens_output]
        loss.backward()

        anchor_losses[option] = loss.item()
        anchor_gradients[option] = embedding_input.grad.clone().detach()

    dp_label = []
    dp_loss_all = []
    dp_loss, dp_gradients = {}, {}
    for dp in tqdm(test_data[1:]):
        for option in dp["options"]:
            input_tokens =init+ "Input: " + dp["input"] + " Label:"
            
            tokens_input = tokenizer(input_tokens, return_tensors="pt", padding="max_length", truncation=True, max_length=100*(args.k+1)).input_ids.to(device)
            tokens_output = tokenizer(option, return_tensors="pt").input_ids[0][-1].to(device)

            # last_token_idx = tokens_input.ne(tokenizer.pad_token_id).sum(dim=1) - 1
            # print("last_token_idx : ", last_token_idx)

            with torch.no_grad():
                if "gpt2" in model_name: embedding_input = model.transformer.wte(tokens_input)
                elif "opt" in model_name: embedding_input = model.model.decoder.embed_tokens(tokens_input)
                else: embedding_input = model.model.embed_tokens(tokens_input)
            
            embedding_input.requires_grad = True

            output_logits = model(inputs_embeds=embedding_input).logits
            last_token_idx = tokens_input.shape[1] - 1
            log_probs = F.log_softmax(output_logits[0, last_token_idx, :], dim=-1)

            loss = -log_probs[tokens_output]
            loss.backward()

            dp_loss[option] = loss.item()
            dp_gradients[option] = embedding_input.grad.clone().detach()
            predicted_dp_idx = np.argmin(dp_loss)

            dp_loss_all.append(dp_loss)
            dp_label.append(dp["options"][predicted_dp_idx])


    correct_predictions = 0
    total_samples = len(test_data) - 1 

    current_error = 0.0
    error_list = [] 

    for idx, dp in tqdm(enumerate(test_data[1:])):
        option_losses = []

        for option in dp["options"]:
            input_tokens = init+"Input: " + dp["input"] + " Label:"

            tokens_input = tokenizer(input_tokens, return_tensors="pt", padding="max_length", truncation=True, max_length=100*(args.k+1)).input_ids.to(device)

            with torch.no_grad():
                if "gpt2" in model_name: 
                    embedding_dp_option = model.transformer.wte(tokens_input)
                elif "opt" in model_name: 
                    embedding_dp_option = model.model.decoder.embed_tokens(tokens_input)
                else: 
                    embedding_dp_option = model.model.embed_tokens(tokens_input)

            delta_P = embedding_dp_option - anchor_embedding_input
            taylor_correction = torch.sum(anchor_gradients[option] * delta_P).item()
            estimated_loss = anchor_losses[option] + taylor_correction

            option_losses.append(estimated_loss)

        predicted_option_idx = np.argmin(option_losses)
        predicted_label = dp["options"][predicted_option_idx]

        inference_loss = dp_loss_all[idx]

        sample_errors = []
        for jdx, label in enumerate(dp["options"]):
            error = np.fabs(inference_loss[label] - option_losses[jdx]) / max(inference_loss[label], option_losses[jdx])
            current_error += error
            sample_errors.append(error)

        error_list.append(np.mean(sample_errors)) 
        if predicted_label == dp_label[idx]:
            correct_predictions += 1

    total_samples = len(test_data) - 1

    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    current_error = current_error / total_samples / len(test_data[0]["options"])

    error_variance = np.var(error_list) if error_list else 0

    print("len(test_data[0][\"options\"]) : ",len(test_data[0]["options"]))
    print("error : ", current_error)
    print("error variance : ", error_variance)
    print("accuracy : ", accuracy)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="superglue-cb", type=str)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B", type=str)
    parser.add_argument("--k", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    main(args)