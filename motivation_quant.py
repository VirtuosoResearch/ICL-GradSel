import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2LMHeadModel, GPT2Tokenizer, OPTForCausalLM
from utils.data import load_data
from tqdm import tqdm
import argparse
import random
from transformers import BitsAndBytesConfig

# "meta-llama/Llama-3.2-1B"
def main(args):
    dataset_name = args.task
    model_name = args.model
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        llm_int8_threshold=6.0
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config=bnb_config,
        device_map= {"": args.device}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model=model.to(device)

    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    if "CodeLlama" in model_name:
        tokenizer.padding_side = 'right'
    
    print("tokenizer.padding_side: ",tokenizer.padding_side)

    model.resize_token_embeddings(len(tokenizer))

    test_data = load_data(None, "test", 3, seed=42, config_split="test",
                        datasets=[dataset_name], is_null=False)

    if len(test_data)>1000: test_data = test_data[:1000]
    
    instructions = f"Here are {len(test_data[0]['options'])} options: "
    for option in test_data[0]["options"]: instructions+=option+", "
    instructions+="You should choose one of them to answer after 'Output: '. \n"
    init = instructions

    random.seed(args.seed)
    random_numbers = random.sample(range(len(test_data)), args.k)
    if args.k>0:
        init+= f"Here are {args.k} samples for your reference. \n"
    for i in random_numbers:
        init+="Input: " + test_data[i]["input"]+" Output: "+test_data[i]["output"]+"\n"

    init+="Here is the query to answer: \n"

    anchor_dp = test_data[0]

    anchor_losses = {}
    anchor_gradients = {}

    for option in anchor_dp["options"]:
        input =  init+"Input: " + anchor_dp["input"] + " Output: "
        input_tokens = tokenizer(input, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_length*(args.k+1))
        input_ids = input_tokens["input_ids"].to(device)
        attention_mask = input_tokens["attention_mask"].to(device)
        tokens_output = tokenizer(option, return_tensors="pt").input_ids[0][1].to(device)
        print("input_tokens.keys(): ",input_tokens.keys())
        print("input_token: ",tokenizer.decode(input_ids[0]))
        print("output_token: ", tokens_output)
        print("output_text: ",tokenizer.decode(tokens_output))

        # last_token_idx = tokens_input.ne(tokenizer.pad_token_id).sum(dim=1) - 1
        # print("last_token_idx : ", last_token_idx)

        with torch.no_grad():
            if "gpt2" in model_name: embedding_input = model.transformer.wte(input_ids)
            elif "opt" in model_name: embedding_input = model.model.decoder.embed_tokens(input_ids)
            else: embedding_input = model.model.embed_tokens(input_ids)
        
        anchor_embedding_input = embedding_input.clone().detach()

        embedding_input.requires_grad = True

        print(f"embedding_input shape: {embedding_input.shape}")
        # print(f"Max index in inputs_embeds: {embedding_input.max()}")

        embedding_input = embedding_input.to(model.dtype)  
        output_logits = model(inputs_embeds=embedding_input, attention_mask=attention_mask).logits
        last_token_idx = attention_mask.sum(dim=1).item()-1
        print("last_token_idx: ",last_token_idx)
        print("output: ", tokenizer.decode(torch.argmax(output_logits[0,last_token_idx,:])))
        # log_probs = F.log_softmax(output_logits[0, last_token_idx, :], dim=-1)
        # loss = -log_probs[tokens_output]
        # loss.backward()
        selected_logit = -output_logits[0, last_token_idx, tokens_output.item()]
        gradient = torch.autograd.grad(selected_logit, embedding_input, retain_graph=False, create_graph=False)[0]

        anchor_losses[option] = selected_logit.item()
        anchor_gradients[option] = gradient.detach()

    dp_label = []
    dp_loss_all = []
    dp_loss, dp_gradients = {}, {}
    # exit()
    for dp in tqdm(test_data[1:]):
        for option in dp["options"]:
            input_tokens =init+ "Input: " + dp["input"] + " Output: "
            
            tokens_input = tokenizer(input_tokens, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_length*(args.k+1))
            input_ids = tokens_input["input_ids"].to(device)
            attention_mask = tokens_input["attention_mask"].to(device)
            tokens_output = tokenizer(option, return_tensors="pt").input_ids[0][1].to(device)

            # last_token_idx = tokens_input.ne(tokenizer.pad_token_id).sum(dim=1) - 1
            # print("last_token_idx : ", last_token_idx)

            with torch.no_grad():
                if "gpt2" in model_name: embedding_input = model.transformer.wte(input_ids)
                elif "opt" in model_name: embedding_input = model.model.decoder.embed_tokens(input_ids)
                else: embedding_input = model.model.embed_tokens(input_ids)
            
            embedding_input = embedding_input.to(model.dtype)
            embedding_input.requires_grad = True

            output_logits = model(inputs_embeds=embedding_input, attention_mask=attention_mask).logits
            last_token_idx = attention_mask.sum(dim=1).item()-1
            # log_probs = F.log_softmax(output_logits[0, last_token_idx, :], dim=-1)
            # loss = -log_probs[tokens_output]
            # loss.backward()
            selected_logit = -output_logits[0, last_token_idx, tokens_output.item()]
            gradient = torch.autograd.grad(selected_logit, embedding_input, retain_graph=False, create_graph=False)[0]

            dp_loss[option] = selected_logit.item()
            dp_gradients[option] = gradient.detach()
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

            tokens_input = tokenizer(input_tokens, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_length*(args.k+1))
            input_ids = tokens_input["input_ids"].to(device)
            attention_mask = tokens_input["attention_mask"].to(device)

            with torch.no_grad():
                if "gpt2" in model_name: 
                    embedding_dp_option = model.transformer.wte(input_ids)
                elif "opt" in model_name: 
                    embedding_dp_option = model.model.decoder.embed_tokens(input_ids)
                else: 
                    embedding_dp_option = model.model.embed_tokens(input_ids)

            delta_P = embedding_dp_option - anchor_embedding_input
            # delta_P_norm = torch.norm(delta_P).item()
            # dp_norm = torch.norm(embedding_dp_option).item()
            # anchor_norm = torch.norm(anchor_embedding_input).item()
            # print("mean_norm: ", delta_P_norm)
            # print("norm_rate: ",delta_P_norm/max(dp_norm, anchor_norm))

            taylor_correction = torch.sum(anchor_gradients[option] * delta_P).item()
            estimated_loss = anchor_losses[option] + taylor_correction
            # print(f"anchor_gradients.size: {anchor_gradients[option].shape}; delta_P.size: {delta_P.shape}")
            option_losses.append(estimated_loss)

        predicted_option_idx = np.argmin(option_losses)
        predicted_label = dp["options"][predicted_option_idx]

        inference_loss = dp_loss_all[idx]

        sample_errors = []
        for jdx, label in enumerate(dp["options"]):
            # print(f"inference_loss[label] : {inference_loss[label]}, option_losses[jdx] : {option_losses[jdx]}")
            error = np.fabs(np.fabs(inference_loss[label]) - np.fabs(option_losses[jdx])) / max(np.fabs(inference_loss[label]), np.fabs(option_losses[jdx]))
            current_error += error
            sample_errors.append(error)

        error_list.append(np.mean(sample_errors)) 
        if predicted_label == dp_label[idx]:
            correct_predictions += 1

    total_samples = len(test_data) - 1

    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    current_error = current_error / total_samples / len(test_data[0]["options"])

    error_variance = np.var(error_list) if error_list else 0

    print("error : ", current_error)
    print("error variance : ", error_variance)
    print("accuracy : ", accuracy)
    # print(torch.cuda.memory_summary(device=args.device, abbreviated=False))



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="sst2", type=str)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--model", default="meta-llama/Llama-2-13b-hf", type=str)
    parser.add_argument("--k", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    main(args)
