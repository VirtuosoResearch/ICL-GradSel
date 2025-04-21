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
from sklearn.decomposition import PCA

pca_model = None

def train_pca_on_embeddings(embedding_list, n_components=128):
    matrix = np.concatenate(embedding_list, axis=0)
    print(f"[PCA] Training PCA on shape: {matrix.shape}")
    pca = PCA(n_components=n_components)
    pca.fit(matrix)
    return pca

def apply_pca_denoise(embedding_input, pca):
    if pca is None:
        return embedding_input

    emb = embedding_input[0].detach().cpu().numpy()
    proj = pca.transform(emb)
    denoised = pca.inverse_transform(proj)
    emb_denoised = torch.tensor(denoised, dtype=embedding_input.dtype, device=embedding_input.device)
    return emb_denoised.unsqueeze(0)


def estimate_loss_second_order(anchor_loss, anchor_gradient, anchor_input, dp_input, option):
    delta_P = dp_input - anchor_input.detach()
    delta_P = delta_P.to(anchor_input.device)

    delta_P.requires_grad = False
    Hvs = torch.autograd.grad(
        outputs=anchor_gradient, 
        inputs=anchor_input,
        grad_outputs=delta_P, 
        retain_graph=True,
        create_graph=False,
        allow_unused=False
    )[0]

    taylor_1st = torch.dot(anchor_gradient.view(-1), delta_P.view(-1))
    if Hvs is None:
        print(f"[Warning] Hessian-vector product failed at option: {option}")
        taylor_2nd = torch.tensor(0.0, device=anchor_input.device)
    else:
        taylor_2nd = 0.5 * torch.dot(delta_P.view(-1), Hvs.view(-1))

    estimated_loss = anchor_loss + taylor_1st.item() + taylor_2nd.item()

    del delta_P, taylor_1st, taylor_2nd, Hvs
    torch.cuda.empty_cache()

    return estimated_loss


def main(args):
    global pca_model

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
        device_map={"": args.device},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model.to(device)

    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    if "CodeLlama" in model_name:
        tokenizer.padding_side = 'right'

    model.resize_token_embeddings(len(tokenizer))

    test_data = load_data(None, "test", 3, seed=42, config_split="test", datasets=[dataset_name], is_null=False)
    dev_data = load_data(None, "dev", 3, seed=42, config_split="dev", datasets=[dataset_name], is_null=False)

    if len(test_data) > 1000:
        test_data = test_data[:1000]

    instructions = f"Here are {len(test_data[0]['options'])} options: " + ", ".join(test_data[0]["options"]) + ".\nYou should choose one of them to answer after 'Output: '.\n"
    init = instructions

    random.seed(args.seed)
    random_numbers = random.sample(range(len(test_data)), args.k)
    if args.k > 0:
        init += f"Here are {args.k} samples for your reference.\n"
    for i in random_numbers:
        init += "Input: " + dev_data[i]["input"] + " Output: " + dev_data[i]["output"] + "\n"
    init += "Here is the query to answer:\n"

    embedding_collection = []

    for dp in tqdm(test_data[1:], desc="Collecting embeddings for PCA"):
        for option in dp["options"]:
            input_text = init + "Input: " + dp["input"] + " Output: "
            input_tokens = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_length * (args.k + 1))
            input_ids = input_tokens["input_ids"].to(device)

            with torch.no_grad():
                if "gpt2" in model_name: 
                    embedding_input = model.transformer.wte(input_ids)
                elif "opt" in model_name: 
                    embedding_input = model.model.decoder.embed_tokens(input_ids)
                else: 
                    embedding_input = model.model.embed_tokens(input_ids)

            embedding_collection.append(embedding_input[0].cpu().numpy())

    pca_model = train_pca_on_embeddings(embedding_collection, n_components=128)

    with sdpa_kernel(SDPBackend.MATH):
        anchor_dp = test_data[0]
        anchor_losses = {}
        anchor_gradients = {}
        anchor_embedding = {}

        for option in anchor_dp["options"]:
            input_text = init + "Input: " + anchor_dp["input"] + " Output: "
            input_tokens = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_length * (args.k + 1))
            input_ids = input_tokens["input_ids"].to(device)
            attention_mask = input_tokens["attention_mask"].to(device)
            tokens_output = tokenizer(option, return_tensors="pt").input_ids[0][1].to(device)

            with torch.no_grad():
                if "gpt2" in model_name: 
                    embedding_input = model.transformer.wte(input_ids)
                elif "opt" in model_name: 
                    embedding_input = model.model.decoder.embed_tokens(input_ids)
                else: 
                    embedding_input = model.model.embed_tokens(input_ids)

            embedding_input = apply_pca_denoise(embedding_input, pca_model)
            embedding_input.requires_grad = True
            embedding_input = embedding_input.to(model.dtype)

            output_logits = model(inputs_embeds=embedding_input, attention_mask=attention_mask).logits
            last_token_idx = attention_mask.sum(dim=1).item() - 1
            selected_logit = -output_logits[0, last_token_idx, tokens_output.item()]
            gradient = torch.autograd.grad(selected_logit, embedding_input, retain_graph=True, create_graph=True)[0]

            anchor_embedding[option] = embedding_input
            anchor_losses[option] = selected_logit.item()
            anchor_gradients[option] = gradient

        correct_predictions = 0
        total_samples = len(test_data) - 1
        error_list = []
        distances = []

        for dp_idx, dp in tqdm(enumerate(test_data[1:]), total=total_samples):
            option_losses = []
            dp_losses = {}

            for option in dp["options"]:
                input_text = init + "Input: " + dp["input"] + " Output: "
                input_tokens = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_length * (args.k + 1))
                input_ids = input_tokens["input_ids"].to(device)
                attention_mask = input_tokens["attention_mask"].to(device)
                tokens_output = tokenizer(option, return_tensors="pt").input_ids[0][1].to(device)

                with torch.no_grad():
                    if "gpt2" in model_name: 
                        embedding_input = model.transformer.wte(input_ids)
                    elif "opt" in model_name: 
                        embedding_input = model.model.decoder.embed_tokens(input_ids)
                    else: 
                        embedding_input = model.model.embed_tokens(input_ids)

                embedding_input = apply_pca_denoise(embedding_input, pca_model)
                embedding_input = embedding_input.to(model.dtype)

                output_logits = model(inputs_embeds=embedding_input, attention_mask=attention_mask).logits
                last_token_idx = attention_mask.sum(dim=1).item() - 1
                inference_loss = -output_logits[0, last_token_idx, tokens_output.item()].item()
                dp_losses[option] = inference_loss

                estimated_loss = estimate_loss_second_order(
                    anchor_losses[option],
                    anchor_gradients[option],
                    anchor_embedding[option],
                    embedding_input,
                    option
                )
                option_losses.append(estimated_loss)

                delta_P = embedding_input.detach() - anchor_embedding[option].detach()
                distances.append(
                    torch.norm(delta_P).item() / max(
                        torch.norm(embedding_input.detach()).item(),
                        torch.norm(anchor_embedding[option].detach()).item()
                    )
                )

            predicted_idx = int(np.argmin(option_losses))
            predicted_label = dp["options"][predicted_idx]
            true_idx = int(np.argmin([dp_losses[o] for o in dp["options"]]))
            true_label = dp["options"][true_idx]

            sample_errors = []
            for j, option in enumerate(dp["options"]):
                true_l = abs(dp_losses[option])
                est_l = abs(option_losses[j])
                rel_err = abs(true_l - est_l) / max(true_l, est_l)
                sample_errors.append(rel_err)

            error_list.append(np.mean(sample_errors))
            if predicted_label == true_label:
                correct_predictions += 1

        print("accuracy:", correct_predictions / total_samples)
        print("avg estimation error:", np.mean(error_list))
        print("error variance:", np.var(error_list))
        print("relevant distance:", np.mean(distances))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="sst2", type=str)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--model", default="meta-llama/Llama-2-13b-hf", type=str)
    parser.add_argument("--k", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    main(args)


# import torch
# import torch.nn.functional as F
# import numpy as np
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers import GPT2LMHeadModel, GPT2Tokenizer, OPTForCausalLM
# from utils.data import load_data
# from tqdm import tqdm
# import argparse
# import random
# from transformers import BitsAndBytesConfig
# from transformers import AutoConfig
# from torch.nn.attention import SDPBackend, sdpa_kernel
# from sklearn.decomposition import PCA

# pca_model = None

# def train_pca_on_embeddings(model, tokenizer, data_list, device, model_name, max_length=128, sample_size=200, n_components=128):
#     all_embeddings = []
#     for i, dp in enumerate(data_list[:sample_size]):
#         input_text = "Input: " + dp["input"] + " Output: " + dp["output"]
#         input_tokens = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
#         input_ids = input_tokens["input_ids"].to(device)

#         with torch.no_grad():
#             if "gpt2" in model_name: 
#                 embedding = model.transformer.wte(input_ids)
#             elif "opt" in model_name: 
#                 embedding = model.model.decoder.embed_tokens(input_ids)
#             else: 
#                 embedding = model.model.embed_tokens(input_ids)

#         all_embeddings.append(embedding[0].cpu().numpy())

#     matrix = np.concatenate(all_embeddings, axis=0)
#     print(f"[PCA] Training PCA on shape: {matrix.shape}")
#     pca = PCA(n_components=n_components)
#     pca.fit(matrix)
#     return pca

# def apply_pca_denoise(embedding_input, pca):
#     if pca is None:
#         return embedding_input

#     emb = embedding_input[0].detach().cpu().numpy()
#     proj = pca.transform(emb)
#     denoised = pca.inverse_transform(proj)
#     emb_denoised = torch.tensor(denoised, dtype=embedding_input.dtype, device=embedding_input.device)
#     return emb_denoised.unsqueeze(0)


# def estimate_loss_second_order(anchor_loss, anchor_gradient, anchor_input, dp_input, option):
#     delta_P = dp_input - anchor_input.detach()
#     delta_P = delta_P.to(anchor_input.device)

#     delta_P.requires_grad = False
#     Hvs = torch.autograd.grad(
#         outputs=anchor_gradient, 
#         inputs=anchor_input,
#         grad_outputs=delta_P, 
#         retain_graph=True,
#         create_graph=False,
#         allow_unused=False
#     )[0]

#     taylor_1st = torch.dot(anchor_gradient.view(-1), delta_P.view(-1))
#     if Hvs is None:
#         print(f"[Warning] Hessian-vector product failed at option: {option}")
#         taylor_2nd = torch.tensor(0.0, device=anchor_input.device)
#     else:
#         taylor_2nd = 0.5 * torch.dot(delta_P.view(-1), Hvs.view(-1))

#     estimated_loss = anchor_loss + taylor_1st.item() + taylor_2nd.item()

#     del delta_P, taylor_1st, taylor_2nd, Hvs
#     torch.cuda.empty_cache()

#     return estimated_loss


# def main(args):
#     global pca_model

#     dataset_name = args.task
#     model_name = args.model
#     device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_quant_type="nf4",
#         llm_int8_threshold=6.0
#     )

#     model = AutoModelForCausalLM.from_pretrained(
#         model_name, 
#         quantization_config=bnb_config,
#         device_map={"": args.device},
#     )
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = model.to(device)

#     model.eval()
#     tokenizer.pad_token = tokenizer.eos_token
#     if "CodeLlama" in model_name:
#         tokenizer.padding_side = 'right'

#     model.resize_token_embeddings(len(tokenizer))

#     test_data = load_data(None, "test", 3, seed=42, config_split="test", datasets=[dataset_name], is_null=False)
#     dev_data = load_data(None, "dev", 3, seed=42, config_split="dev", datasets=[dataset_name], is_null=False)

#     if len(test_data) > 1000:
#         test_data = test_data[:1000]

#     pca_model = train_pca_on_embeddings(model, tokenizer, test_data, device, model_name, max_length=args.max_length, sample_size=200, n_components=128)

#     instructions = f"Here are {len(test_data[0]['options'])} options: " + ", ".join(test_data[0]["options"]) + ".\nYou should choose one of them to answer after 'Output: '.\n"
#     init = instructions

#     random.seed(args.seed)
#     random_numbers = random.sample(range(len(test_data)), args.k)
#     if args.k > 0:
#         init += f"Here are {args.k} samples for your reference.\n"
#     for i in random_numbers:
#         init += "Input: " + dev_data[i]["input"] + " Output: " + dev_data[i]["output"] + "\n"
#     init += "Here is the query to answer:\n"

#     with sdpa_kernel(SDPBackend.MATH):
#         anchor_dp = test_data[0]
#         anchor_losses = {}
#         anchor_gradients = {}
#         anchor_embedding = {}

#         for option in anchor_dp["options"]:
#             input_text = init + "Input: " + anchor_dp["input"] + " Output: "
#             input_tokens = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_length * (args.k + 1))
#             input_ids = input_tokens["input_ids"].to(device)
#             attention_mask = input_tokens["attention_mask"].to(device)
#             tokens_output = tokenizer(option, return_tensors="pt").input_ids[0][1].to(device)

#             with torch.no_grad():
#                 if "gpt2" in model_name: 
#                     embedding_input = model.transformer.wte(input_ids)
#                 elif "opt" in model_name: 
#                     embedding_input = model.model.decoder.embed_tokens(input_ids)
#                 else: 
#                     embedding_input = model.model.embed_tokens(input_ids)

#             embedding_input = apply_pca_denoise(embedding_input, pca_model)
#             embedding_input.requires_grad = True
#             embedding_input = embedding_input.to(model.dtype)

#             output_logits = model(inputs_embeds=embedding_input, attention_mask=attention_mask).logits
#             last_token_idx = attention_mask.sum(dim=1).item() - 1
#             selected_logit = -output_logits[0, last_token_idx, tokens_output.item()]
#             gradient = torch.autograd.grad(selected_logit, embedding_input, retain_graph=True, create_graph=True)[0]

#             anchor_embedding[option] = embedding_input
#             anchor_losses[option] = selected_logit.item()
#             anchor_gradients[option] = gradient

#         dp_label = []
#         dp_loss_all = []
#         dp_loss = {}

#         for dp in tqdm(test_data[1:]):
#             for option in dp["options"]:
#                 input_text = init + "Input: " + dp["input"] + " Output: "
#                 input_tokens = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_length * (args.k + 1))
#                 input_ids = input_tokens["input_ids"].to(device)
#                 attention_mask = input_tokens["attention_mask"].to(device)
#                 tokens_output = tokenizer(option, return_tensors="pt").input_ids[0][1].to(device)

#                 with torch.no_grad():
#                     if "gpt2" in model_name: 
#                         embedding_input = model.transformer.wte(input_ids)
#                     elif "opt" in model_name: 
#                         embedding_input = model.model.decoder.embed_tokens(input_ids)
#                     else: 
#                         embedding_input = model.model.embed_tokens(input_ids)

#                 embedding_input = apply_pca_denoise(embedding_input, pca_model)
#                 embedding_input.requires_grad = True
#                 embedding_input = embedding_input.to(model.dtype)
#                 output_logits = model(inputs_embeds=embedding_input, attention_mask=attention_mask).logits
#                 last_token_idx = attention_mask.sum(dim=1).item() - 1
#                 selected_logit = -output_logits[0, last_token_idx, tokens_output.item()]
#                 dp_loss[option] = selected_logit.item()

#             predicted_dp_idx = np.argmin([dp_loss[o] for o in dp["options"]])
#             dp_loss_all.append(dp_loss.copy())
#             dp_label.append(dp["options"][predicted_dp_idx])

#         correct_predictions = 0
#         total_samples = len(test_data) - 1 
#         current_error = 0.0
#         error_list = [] 
#         distances = []

#         for idx, dp in tqdm(enumerate(test_data[1:]), total=len(test_data)-1):
#             option_losses = []
#             for option in dp["options"]:
#                 input_text = init + "Input: " + dp["input"] + " Output: "
#                 input_tokens = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_length * (args.k + 1))
#                 input_ids = input_tokens["input_ids"].to(device)
#                 attention_mask = input_tokens["attention_mask"].to(device)

#                 with torch.no_grad():
#                     if "gpt2" in model_name: 
#                         embedding_dp_option = model.transformer.wte(input_ids)
#                     elif "opt" in model_name: 
#                         embedding_dp_option = model.model.decoder.embed_tokens(input_ids)
#                     else: 
#                         embedding_dp_option = model.model.embed_tokens(input_ids)

#                 embedding_dp_option = apply_pca_denoise(embedding_dp_option, pca_model)
#                 estimated_loss = estimate_loss_second_order(
#                     anchor_loss=anchor_losses[option],
#                     anchor_gradient=anchor_gradients[option],
#                     anchor_input=anchor_embedding[option],
#                     dp_input=embedding_dp_option,
#                     option=option
#                 )
#                 option_losses.append(estimated_loss)

#                 delta_P = embedding_dp_option.detach() - anchor_embedding[option].detach()
#                 distances.append(
#                     torch.norm(delta_P).item() / max(
#                         torch.norm(embedding_dp_option.detach()).item(),
#                         torch.norm(anchor_embedding[option].detach()).item()
#                     )
#                 )

#             predicted_option_idx = np.argmin(option_losses)
#             predicted_label = dp["options"][predicted_option_idx]
#             inference_loss = dp_loss_all[idx]
#             sample_errors = []
#             for jdx, label in enumerate(dp["options"]):
#                 error = np.fabs(np.fabs(inference_loss[label]) - np.fabs(option_losses[jdx])) / max(np.fabs(inference_loss[label]), np.fabs(option_losses[jdx]))
#                 current_error += error
#                 sample_errors.append(error)
#             error_list.append(np.mean(sample_errors)) 
#             if predicted_label == dp_label[idx]:
#                 correct_predictions += 1

#         accuracy = correct_predictions / total_samples if total_samples > 0 else 0
#         current_error = current_error / total_samples / len(test_data[0]["options"])
#         error_variance = np.var(error_list) if error_list else 0

#         print("error : ", current_error)
#         print("error variance : ", error_variance)
#         print("accuracy : ", accuracy)
#         print("relevant distance: ", np.mean(distances))


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--task", default="sst2", type=str)
#     parser.add_argument("--device", default=0, type=int)
#     parser.add_argument("--max_length", default=128, type=int)
#     parser.add_argument("--model", default="meta-llama/Llama-2-13b-hf", type=str)
#     parser.add_argument("--k", default=0, type=int)
#     parser.add_argument("--seed", default=0, type=int)
#     args = parser.parse_args()
#     main(args)
