# %%

import torch
import json
import logging
import numpy as np
from transformers import GPT2Tokenizer, AutoTokenizer
from metaicl.data import MetaICLData
from metaicl.model import MetaICLModel
from metaicl.data import prepro_sentence_pair_single
from utils.data import load_data

class args:
    do_zeroshot = True
    use_demonstrations = True
    use_calibration = False
    unseen_domain_only = False
    log_file = None
    task = None
    dataset = "glue-rte"
    k = 3
    unlabeled_k = 2
    seed = "42"
    device = 0
    test_batch_size = 4
    global_step = None
    checkpoint = None
    out_dir = "out/gpt2-large"
    split = "test"
    is_null = False
    method = "direct"
    gpt2 = "gpt2-large"
    m = 4


logger = logging.getLogger(__name__)
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
if args.gpt2.startswith("gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2)
elif "Llama" in args.gpt2:
    tokenizer = AutoTokenizer.from_pretrained(args.gpt2)
else:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

add_newlines = not args.gpt2.startswith("gpt2")
checkpoint = None
metaicl_model = MetaICLModel(logger=logger, out_dir= args.out_dir, device_num=args.device)
metaicl_model.load(checkpoint, gpt2=args.gpt2)

max_length_per_example, max_length = 128, 256
if args.use_demonstrations:
    max_length = min(max_length * args.k, 1024)

metaicl_data = MetaICLData(logger, tokenizer, args.method, args.use_demonstrations, args.k,
                            max_length, max_length_per_example)

results = []
errors = []
seeds = args.seed.split(",")
config_split = "unseen_domain_test" if args.unseen_domain_only else "test"

seed = seeds[0]
test_data = load_data(args.task, args.split, args.k, seed=seed, config_split=config_split,
            datasets=None if args.dataset is None else args.dataset.split(","), is_null=args.is_null)
task = args.dataset

with open(f"config/tasks/{task}.json", "r") as f:
    config = json.load(f)
with open(f"./features/{task}_features.json", "r") as f:
    test_features = json.load(f)


def run_a_forward_pass(input_tokens, output_tokens, tokenizer):
    encoded = prepro_sentence_pair_single(
                input_tokens, output_tokens, max_length=1024, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
                allow_truncation=args.use_demonstrations
        )
    input_ids = torch.LongTensor([encoded[0]])
    attention_mask = torch.LongTensor([encoded[1]])
    token_type_ids = torch.LongTensor([encoded[2]])

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    token_type_ids = token_type_ids.to(device)
    results = metaicl_model.run_model(input_ids, attention_mask, token_type_ids)
    return input_ids, results.cpu().detach().item()

dp_idx = 0
dp = test_data[dp_idx]
option_tokens = [
    tokenizer(option)["input_ids"] for option in dp['options']
]
options = [dp["options"].index(dp["output"])]

dp = test_data[dp_idx]
input_tokens = tokenizer("Input: " + dp["input"] + " " + "Label: ")["input_ids"]
output_tokens = tokenizer(dp["output"])["input_ids"]
dp_feature = test_features[dp_idx]
random_k_neighbors = metaicl_data._select_random_k_neighbors(
    dp_feature, test_features, test_data, args.k, dp_idx
)
metaicl_model.model.eval()
metaicl_model.model.to(device)
demonstrations = []
topk_data, topk_indices, __ = metaicl_data._select_top_k_neighbors(
    dp_feature, test_features, test_data, args.unlabeled_k, dp_idx
)
for i in topk_indices:
    unlabel_dp = test_data[i]
    tmp_str = "Input: " + unlabel_dp["input"] + " " + "Label: " + "\n"
    demonstrations += tokenizer(tmp_str)["input_ids"]

for i, neighbor_dp in enumerate(random_k_neighbors):
    tmp_str =  "Input: " + neighbor_dp["input"] + " " + "Label: " + neighbor_dp["output"] + "\n"
    demonstrations += tokenizer(tmp_str)["input_ids"]

one_trial_losses = []
for option_token in option_tokens:
    input_ids, results = run_a_forward_pass(demonstrations + input_tokens, option_token, tokenizer)
    one_trial_losses.append(results)

label_id = np.argmin(one_trial_losses)
label = dp["options"][label_id]
print(label)