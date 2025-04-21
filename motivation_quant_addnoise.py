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
from transformers import AutoConfig
from torch.nn.attention import SDPBackend, sdpa_kernel



def estimate_loss_second_order(anchor_loss, anchor_gradient, anchor_input, dp_input, option):
    delta_P = dp_input - anchor_input.detach()
    delta_P = delta_P.to(anchor_input.device)

    # delta_P.requires_grad = False
    Hvs = torch.autograd.grad(
        outputs=anchor_gradient, 
        inputs=anchor_input,
        grad_outputs=delta_P, 
        retain_graph=True,
        create_graph=False,
        allow_unused=False
    )[0]

    # taylor_1st = torch.sum(anchor_gradient * delta_P)
    taylor_1st = torch.dot(anchor_gradient.view(-1), delta_P.view(-1))
    if Hvs is None:
        print(f"[Warning] Hessian-vector product failed at option: {option}")
        taylor_2nd = torch.tensor(0.0, device=anchor_input.device)
    else:
        # taylor_2nd = 0.5 * torch.sum(delta_P * Hvs)
        taylor_2nd = 0.5 * torch.dot(delta_P.view(-1), Hvs.view(-1))
    
    # print("anchor_loss: ",anchor_loss, "taylor_1st: ", taylor_1st.item(), "taylor_2nd: ",taylor_2nd.item())

    estimated_loss_1 = anchor_loss + taylor_1st.item()
    estimated_loss_2 = anchor_loss + taylor_1st.item() + taylor_2nd.item()

    del delta_P, taylor_1st, taylor_2nd, Hvs
    torch.cuda.empty_cache()

    return estimated_loss_1, estimated_loss_2

def perturbation_estimation_eval(anchor_losses, anchor_gradients, anchor_embedding, model, tokenizer, option_texts, attention_mask, max_relative_dist=0.05, trials=100):
    device = next(model.parameters()).device

    errors1, errors2 = [],[]
    for option in option_texts:
        original_embed = anchor_embedding[option]
        norm_orig = torch.norm(original_embed)
        losses_estimated1 = []
        losses_estimated2 = []
        losses_true = []

        for _ in tqdm(range(trials)):
            noise = torch.randn_like(original_embed).to(device)
            noise = noise / torch.norm(noise) * norm_orig * max_relative_dist
            perturbed_embed = original_embed + noise
            estimated_loss_1, estimated_loss_2 = estimate_loss_second_order(
                anchor_loss=anchor_losses[option],
                anchor_gradient=anchor_gradients[option],
                anchor_input=original_embed,
                dp_input=perturbed_embed,
                option=option
            )

            # attention_mask = torch.ones(perturbed_embed.shape[:-1], dtype=torch.long).to(device)

            perturbed_embed = perturbed_embed.to(model.dtype)
            output_logits = model(inputs_embeds=perturbed_embed, attention_mask=attention_mask).logits
            last_token_idx = attention_mask.sum(dim=1).item() - 1

            tokens_output = tokenizer(option, return_tensors="pt").input_ids[0][1].to(device)
            true_loss = -output_logits[0, last_token_idx, tokens_output.item()].item()

            error = np.fabs(np.fabs(true_loss) - np.fabs(estimated_loss_1)) / max(np.fabs(true_loss), np.fabs(estimated_loss_1))
            errors1.append(error)
            losses_estimated1.append(estimated_loss_1)
            losses_true.append(true_loss)

            error = np.fabs(np.fabs(true_loss) - np.fabs(estimated_loss_2)) / max(np.fabs(true_loss), np.fabs(estimated_loss_2))
            errors2.append(error)
            losses_estimated2.append(estimated_loss_2)
            losses_true.append(true_loss)

    mean_error1 = np.mean(errors1)
    var_error1 = np.var(errors1)
    mean_error2 = np.mean(errors2)
    var_error2 = np.var(errors2)

    print(f"1st Average Relative Error: {mean_error1}")
    print(f"1st Varience: {var_error1}")
    print(f"2nd Average Relative Error: {mean_error2}")
    print(f"2nd Varience: {var_error2}")



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
        device_map= {"": args.device},
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
    dev_data = load_data(None, "dev", 3, seed=42, config_split="dev",
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
        init+="Input: " + dev_data[i]["input"]+" Output: "+dev_data[i]["output"]+"\n"

    init+="Here is the query to answer: \n"

    with sdpa_kernel(SDPBackend.MATH):

        anchor_dp = test_data[0]

        anchor_losses = {}
        anchor_gradients = {}
        anchor_embedding = {}
        atmk = None
        for option in anchor_dp["options"]:
            input =  init+"Input: " + anchor_dp["input"] + " Output: "
            input_tokens = tokenizer(input, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_length*(args.k+1))
            input_ids = input_tokens["input_ids"].to(device)
            attention_mask = input_tokens["attention_mask"].to(device)
            atmk = attention_mask.clone()
            tokens_output = tokenizer(option, return_tensors="pt").input_ids[0][1].to(device)
            print("input_tokens.keys(): ",input_tokens.keys())
            print("input_token: ",tokenizer.decode(input_ids[0]))
            print("output_token: ", tokens_output)
            print("output_text: ",tokenizer.decode(tokens_output))


            with torch.no_grad():
                if "gpt2" in model_name: embedding_input = model.transformer.wte(input_ids)
                elif "opt" in model_name: embedding_input = model.model.decoder.embed_tokens(input_ids)
                else: embedding_input = model.model.embed_tokens(input_ids)
            
            # anchor_embedding_input = embedding_input.clone().detach()
            embedding_input.requires_grad = True

            print(f"embedding_input shape: {embedding_input.shape}")

            embedding_input = embedding_input.to(model.dtype)  
            output_logits = model(inputs_embeds=embedding_input, attention_mask=attention_mask).logits
            last_token_idx = attention_mask.sum(dim=1).item()-1
            print("last_token_idx: ",last_token_idx)
            print("input_last_token: ", tokenizer.decode(input_ids[0, last_token_idx]))
            print("output: ", tokenizer.decode(torch.argmax(output_logits[0,last_token_idx,:])))
            # log_probs = F.log_softmax(output_logits[0, last_token_idx, :], dim=-1)
            # loss = -log_probs[tokens_output]
            # loss.backward()
            selected_logit = -output_logits[0, last_token_idx, tokens_output.item()]
            gradient = torch.autograd.grad(selected_logit, embedding_input, retain_graph=True, create_graph=True)[0]
            
            anchor_embedding[option] = embedding_input
            anchor_losses[option] = selected_logit.item()
            anchor_gradients[option] = gradient

        perturbation_estimation_eval(
            anchor_losses=anchor_losses, 
            anchor_gradients=anchor_gradients, 
            anchor_embedding=anchor_embedding,
            model=model,
            tokenizer=tokenizer,
            option_texts=test_data[0]["options"],
            attention_mask=atmk,
            max_relative_dist=args.dist
        )


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="sst2", type=str)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--model", default="meta-llama/Llama-2-13b-hf", type=str)
    parser.add_argument("--k", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--dist", default=0.05, type=float)
    args = parser.parse_args()
    # import os
    # os.environ["FLASH_ATTENTION_2_DISABLED"] = "1"
    main(args)
