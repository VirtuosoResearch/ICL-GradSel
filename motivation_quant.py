import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.data import load_data
from tqdm import tqdm
import argparse
import random
from transformers import BitsAndBytesConfig


def compute_log_odds(logits, target_id, option_ids):
    selected_logits = logits[option_ids]
    probs = F.softmax(selected_logits, dim=0)
    idx_in_subset = option_ids.index(target_id)
    p = probs[idx_in_subset]
    return torch.log(p)
    # return torch.log(p / (1 - p + 1e-8))


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
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

    test_data = load_data(None, "test", 3, seed=42, config_split="test",
                          datasets=[args.task], is_null=False)

    if len(test_data) > 1000:
        test_data = test_data[:1000]

    init = ""
    random.seed(args.seed)
    for i in random.sample(range(len(test_data)), args.k):
        init += f"Input: {test_data[i]['input']} Label: {test_data[i]['output']}\n"

    anchor_dp = test_data[0]
    anchor_logodds = {}
    anchor_gradients = {}

    option_token_ids = [
        tokenizer(option, return_tensors="pt").input_ids[0][1].item()
        for option in anchor_dp["options"]
    ]

    for option in anchor_dp["options"]:
        input_str = init + "Input: " + anchor_dp["input"] + " Label:"
        input = tokenizer(input_str, return_tensors="pt", padding="max_length",
                          truncation=True, max_length=args.max_length * (args.k + 1)).to(device)
        input_ids = input["input_ids"]
        attention_mask = input["attention_mask"]
        target_id = tokenizer(option, return_tensors="pt").input_ids[0][1].item()

        print("input: ", tokenizer.decode(input_ids[0]))
        print("target_true_id: ", target_id)
        print("target: ",tokenizer.decode(target_id))

        with torch.no_grad():
            if "gpt2" in args.model:
                embedding_input = model.transformer.wte(input_ids)
            elif "opt" in args.model:
                embedding_input = model.model.decoder.embed_tokens(input_ids)
            else:
                embedding_input = model.model.embed_tokens(input_ids)

        anchor_embedding_input = embedding_input.clone().detach()
        embedding_input = embedding_input.clone().detach().requires_grad_(True).to(model.dtype)

        logits = model(inputs_embeds=embedding_input, attention_mask=attention_mask).logits
        last_index = attention_mask[0].sum().item() - 1
        logits_at_last_token = logits[0, int(last_index), :]

        print("logits_at_last_token: ",tokenizer.decode(torch.argmax(logits_at_last_token)))

        f_theta = compute_log_odds(logits_at_last_token, target_id, option_token_ids)
        f_theta.backward()

        anchor_logodds[option] = f_theta.item()
        anchor_gradients[option] = embedding_input.grad.clone().detach()

    errors = []
    correct = 0
    total = 0

    for dp in tqdm(test_data[1:], desc="Evaluating Taylor Estimation"):
        true_logodds = {}
        estimated_logodds = {}

        for option in dp["options"]:
            input_str = init + "Input: " + dp["input"] + " Label:"
            target_id = tokenizer(option, return_tensors="pt").input_ids[0][1].item()

            input = tokenizer(input_str, return_tensors="pt", padding="max_length", truncation=True,
                              max_length=args.max_length * (args.k + 1)).to(device)
            input_ids = input["input_ids"]
            attention_mask = input["attention_mask"]

            with torch.no_grad():
                if "gpt2" in args.model:
                    embedding_input = model.transformer.wte(input_ids)
                elif "opt" in args.model:
                    embedding_input = model.model.decoder.embed_tokens(input_ids)
                else:
                    embedding_input = model.model.embed_tokens(input_ids)

            embedding_input = embedding_input.to(model.dtype).requires_grad_(True)
            logits = model(inputs_embeds=embedding_input, attention_mask=attention_mask).logits

            last_index = attention_mask[0].sum().item() - 1
            logits_at_last_token = logits[0, int(last_index), :]

            f_theta = compute_log_odds(logits_at_last_token, target_id, option_token_ids)
            true_logodds[option] = f_theta.item()

            delta = embedding_input - anchor_embedding_input
            grad = anchor_gradients[option]
            taylor = torch.sum(grad * delta).item()

            estimated_logodds[option] = anchor_logodds[option] + taylor

        options = dp["options"]
        pred_true = min(options, key=lambda o: true_logodds[o])
        pred_est = min(options, key=lambda o: estimated_logodds[o])

        if pred_true == pred_est:
            correct += 1
        total += 1

        per_option_errors = [
            abs(true_logodds[o] - estimated_logodds[o]) / max(abs(true_logodds[o]), abs(estimated_logodds[o]), 1e-6)
            for o in options
        ]
        errors.append(np.mean(per_option_errors))

    print(f"Taylor Approximation Accuracy: {correct / total:.4f}")
    print(f"Average Relative Error: {np.mean(errors):.4f}")
    print(f"Error Variance: {np.var(errors):.4f}")


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
# from utils.data import load_data
# from tqdm import tqdm
# import argparse
# import random
# from transformers import BitsAndBytesConfig


# # def compute_log_odds(logits, target_id):
# #     probs = F.softmax(logits, dim=-1)
# #     print("probs: ", probs)
# #     print("target_id: ",target_id)
# #     p = probs[target_id]
# #     print("p: ",p)
# #     return torch.log(p / (1 - p))

# def compute_log_odds(logits, target_id, option_ids):
#     selected_logits = logits[option_ids]
#     probs = F.softmax(selected_logits, dim=0)
#     idx_in_subset = option_ids.index(target_id)
#     p = probs[idx_in_subset]
#     return torch.log(p / (1 - p + 1e-8))


# def main(args):
#     device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_quant_type="nf4",
#         llm_int8_threshold=6.0
#     )

#     model = AutoModelForCausalLM.from_pretrained(
#         args.model,
#         quantization_config=bnb_config,
#         device_map={"": args.device}
#     )
#     tokenizer = AutoTokenizer.from_pretrained(args.model)
#     model.eval()
#     tokenizer.pad_token = tokenizer.eos_token
#     model.resize_token_embeddings(len(tokenizer))

#     test_data = load_data(None, "test", 3, seed=42, config_split="test",
#                           datasets=[args.task], is_null=False)

#     if len(test_data) > 1000:
#         test_data = test_data[:1000]

#     init = ""
#     random.seed(args.seed)
#     for i in random.sample(range(len(test_data)), args.k):
#         init += f"Input: {test_data[i]['input']} Label: {test_data[i]['output']}\n"

#     anchor_dp = test_data[0]
#     anchor_logodds = {}
#     anchor_gradients = {}

#     option_token_ids = [
#         tokenizer(option, return_tensors="pt").input_ids[0][1].item()
#         for option in anchor_dp["options"]
#     ]

#     for option in anchor_dp["options"]:
#         input_str = init + "Input: " + anchor_dp["input"] + " Label:"
#         input = tokenizer(input_str, return_tensors="pt", padding=True,
#                               truncation=True, max_length=args.max_length * (args.k + 1))
#         input_ids = input["input_ids"].to(device)
#         attention_mask = input["attention_mask"].to(device)
#         target_id = tokenizer(option, return_tensors="pt").input_ids[0][1]
#         print("input: ", tokenizer.decode(input_ids[0]))
#         print("target_true_id: ", target_id)
#         print("target: ",tokenizer.decode(target_id))

#         with torch.no_grad():
#             if "gpt2" in args.model:
#                 embedding_input = model.transformer.wte(input_ids)
#             elif "opt" in args.model:
#                 embedding_input = model.model.decoder.embed_tokens(input_ids)
#             else:
#                 embedding_input = model.model.embed_tokens(input_ids)

#         anchor_embedding_input = embedding_input.clone().detach()
#         embedding_input = embedding_input.clone().detach().requires_grad_(True).to(model.dtype)

#         # logits = model(inputs_embeds=embedding_input, attention_mask= attention_mask).logits[0, -1, :]
#         logits = model(inputs_embeds=embedding_input, attention_mask=attention_mask).logits
#         last_index = (attention_mask.sum(dim=1) - 1).item()
#         logits_at_last_token = logits[0, last_index, :]

#         print("logits_at_last_token: ",tokenizer.decode(torch.argmax(logits_at_last_token)))
#         f_theta = compute_log_odds(logits_at_last_token, target_id, option_token_ids)
#         f_theta.backward()

#         anchor_logodds[option] = f_theta.item()
#         anchor_gradients[option] = embedding_input.grad.clone().detach()

#     errors = []
#     correct = 0
#     total = 0

#     for dp in tqdm(test_data[1:], desc="Evaluating Taylor Estimation"):
#         true_logodds = {}
#         estimated_logodds = {}
#         delta_embedding = {}

#         for option in dp["options"]:
#             input_str = init + "Input: " + dp["input"] + " Label:"
#             target_id = tokenizer(option, return_tensors="pt").input_ids[0][1].item()

#             input = tokenizer(input_str, return_tensors="pt", padding=True, truncation=True,
#                             max_length=args.max_length * (args.k + 1)).to(device)
#             input_ids = input["input_ids"]
#             attention_mask = input["attention_mask"]

#             with torch.no_grad():
#                 if "gpt2" in args.model:
#                     embedding_input = model.transformer.wte(input_ids)
#                 elif "opt" in args.model:
#                     embedding_input = model.model.decoder.embed_tokens(input_ids)
#                 else:
#                     embedding_input = model.model.embed_tokens(input_ids)

#             embedding_input = embedding_input.to(model.dtype).requires_grad_(True)

#             print("embedding_input.shape: ",embedding_input.shape)
#             last_index = attention_mask[0].sum().item() - 1
#             logits = model(inputs_embeds=embedding_input, attention_mask=attention_mask).logits
#             logits_at_last_token = logits[0, int(last_index), :]

#             f_theta = compute_log_odds(logits_at_last_token, target_id, option_token_ids)
#             true_logodds[option] = f_theta.item()

#             delta = embedding_input - anchor_embedding_input
#             taylor = torch.sum(anchor_gradients[option] * delta).item()
#             estimated_logodds[option] = anchor_logodds[option] + taylor

#         options = dp["options"]
#         pred_true = min(options, key=lambda o: true_logodds[o])
#         pred_est = min(options, key=lambda o: estimated_logodds[o])

#         if pred_true == pred_est:
#             correct += 1
#         total += 1

#         per_option_errors = [
#             abs(true_logodds[o] - estimated_logodds[o]) / max(abs(true_logodds[o]), abs(estimated_logodds[o]), 1e-6)
#             for o in options
#         ]
#         errors.append(np.mean(per_option_errors))


#     # for dp in tqdm(test_data[1:], desc="Evaluating Taylor Estimation"):
#     #     true_logodds = {}
#     #     estimated_logodds = {}
#     #     delta_embedding = {}

#     #     input_strs = [init + "Input: " + dp["input"] + " Label:" for _ in dp["options"]]
#     #     input_token = tokenizer(input_strs, return_tensors="pt", padding=True,
#     #                               truncation=True, max_length=args.max_length * (args.k + 1))
#     #     input_ids_all = input_token["input_ids"].to(device)
#     #     attention_mask = input_token["attention_mask"].to(device)

#     #     for idx, option in enumerate(dp["options"]):
#     #         target_id = tokenizer(option, return_tensors="pt").input_ids[0][1].item()

#     #         with torch.no_grad():
#     #             if "gpt2" in args.model:
#     #                 embedding_input = model.transformer.wte(input_ids_all[idx].unsqueeze(0))
#     #             elif "opt" in args.model:
#     #                 embedding_input = model.model.decoder.embed_tokens(input_ids_all[idx].unsqueeze(0))
#     #             else:
#     #                 embedding_input = model.model.embed_tokens(input_ids_all[idx].unsqueeze(0))

#     #         embedding_input = embedding_input.to(model.dtype).requires_grad_(True)

#     #         print("attention_mask.shape: ",attention_mask.shape)
#     #         last_index = (attention_mask.sum(dim=1)-1).item()
#     #         logits = model(inputs_embeds=embedding_input).logits[0, last_index, :]
#     #         f_theta = compute_log_odds(logits, target_id, option_token_ids)

#     #         true_logodds[option] = f_theta.item()

#     #         delta = embedding_input - anchor_embedding_input
#     #         taylor = torch.sum(anchor_gradients[option] * delta).item()
#     #         estimated_logodds[option] = anchor_logodds[option] + taylor

#     #     options = dp["options"]
#     #     pred_true = min(options, key=lambda o: true_logodds[o])
#     #     pred_est = min(options, key=lambda o: estimated_logodds[o])

#     #     if pred_true == pred_est:
#     #         correct += 1
#     #     total += 1

#     #     per_option_errors = [
#     #         abs(true_logodds[o] - estimated_logodds[o]) / max(abs(true_logodds[o]), abs(estimated_logodds[o]), 1e-6)
#     #         for o in options
#     #     ]
#     #     print(per_option_errors)
#     #     errors.append(np.mean(per_option_errors))

#     print(f"\nTaylor Approximation Accuracy: {correct / total:.4f}")
#     print(f"Average Relative Error: {np.mean(errors):.4f}")
#     print(f"Error Variance: {np.var(errors):.4f}")


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

