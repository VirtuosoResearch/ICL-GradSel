# %%
import os
import argparse
import pickle as pkl
import random
import torch
import math
import json
import string
import logging
import numpy as np

from tqdm import tqdm
from collections import Counter, defaultdict

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import GPT2Tokenizer, AutoTokenizer

from metaicl.data import MetaICLData
from metaicl.model import MetaICLModel

from utils.data import load_data

class args:
    do_zeroshot = True
    use_demonstrations = True
    use_calibration = False
    unseen_domain_only = False
    log_file = None

    task = None
    dataset = "poem_sentiment"
    k = 3
    seed = "42"
    device = 0

    test_batch_size = 4
    global_step = None
    checkpoint = None
    use_random_english_words = False

    out_dir = "out/gpt2-large"

    split = "test"
    is_null = False
    method = "direct"
    gpt2 = "gpt2-large"

    topk = True
    randomk = False
    supcon = False
    unlabeled = False
    multidata = False
    m = 4

handlers = [logging.StreamHandler()]
if args.log_file is not None:
    handlers.append(logging.FileHandler(args.log_file))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=handlers)
logger = logging.getLogger(__name__)
logger.info(args)

device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
if args.gpt2.startswith("gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2)
elif "Llama" in args.gpt2:
    tokenizer = AutoTokenizer.from_pretrained(args.gpt2)
else:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
logger.info("tokenizer.vocab_size : %d",tokenizer.vocab_size)
logger.info(f"PAD token id: {tokenizer.pad_token_id}")
logger.info(f"UNK token id: {tokenizer.unk_token_id}")
logger.info(f"BOS token id: {tokenizer.bos_token_id}")
logger.info(f"EOS token id: {tokenizer.eos_token_id}")


add_newlines = not args.gpt2.startswith("gpt2")
checkpoint = None
metaicl_model = MetaICLModel(logger=logger, out_dir= args.out_dir, device_num=args.device)
metaicl_model.load(checkpoint, gpt2=args.gpt2)


if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

# setup hyperparams for data

max_length_per_example = 128
max_length = 256
if args.use_demonstrations:
    orig_max_length = max_length
    max_length = min(max_length * args.k, 1024)

logger.info("batch_size=%d\tmax_length=%d\tmax_length_per_example=%d" % (
    args.test_batch_size, max_length, max_length_per_example))

metaicl_data = MetaICLData(logger, tokenizer, args.method, args.use_demonstrations, args.k,
                            max_length, max_length_per_example)

results = []
errors = []
seeds = args.seed.split(",")
config_split = "unseen_domain_test" if args.unseen_domain_only else "test"

seed = seeds[0]
test_data = load_data(args.task, args.split, args.k, seed=seed, config_split=config_split,
            datasets=None if args.dataset is None else args.dataset.split(","), is_null=args.is_null)

test_counter = Counter()

for dp in test_data:
    test_counter[dp["task"]] += 1

for k, v in test_counter.items():
    logger.info("[Test] %s\t%d" % (k, v))

logger.info("%s on %s (%d test)" % (args.method, args.task, len(test_counter)))

# %%
task = test_task = args.dataset
curr_test_data = test_data 

config_file = "config/tasks/{}.json".format(test_task)
assert os.path.exists(config_file), config_file
with open(config_file, "r") as f:
    config = json.load(f)

is_classification = config["task_type"]=="classification"
if is_classification:
    options = curr_test_data[0]["options"]
    assert np.all([d["options"]==options for d in curr_test_data])

if args.do_zeroshot:
    split_name = args.split
    if args.is_null:
        split_name += "-null"
    cache_path = os.path.join(args.out_dir,
                                "{}-{}-{}{}{}{}{}{}{}{}{}.pkl".format(
                                    task,
                                    split_name,
                                    metaicl_data.method,
                                    "-topk" if args.topk else "",
                                    "-randomk" if args.randomk else "",
                                    "-supcon" if args.supcon else "",
                                    "-unlabeled" if args.unlabeled else "",
                                    "-k={}".format(args.k) if args.use_demonstrations else "",
                                    "-s={}".format(seed) if args.use_demonstrations or args.use_random_english_words else "",
                                    "" if add_newlines else "-no-newlines",
                                    "-m={}".format(args.m) if args.supcon else ""))

# %%
new_test_data = []

for dp in test_data:
    assert type(dp) == dict, ("Each example should be a dictionary", dp)
    assert "input" in dp and "options" in dp and type(dp["options"]) == list, \
        ("Test example should contain input and options in a list format", dp)
    if "output" not in dp:
        dp["output"] = dp["options"][0]  
    new_test_data.append(dp.copy())

test_texts = [dp["input"] + " " + dp["output"] for dp in new_test_data]

task = test_data[0]["task"]
features_path = f"./features/{task}_features.json"
with open(features_path, "r") as file:
    test_features = json.load(file)

# %%
from metaicl.data import prepro_sentence_pair_single
from scipy.spatial.distance import cosine

def select_top_k_neighbors(test_sample_embedding, test_embeddings, test_data, k, dp_idx):
    similarities = []
    for idx, dp in enumerate(test_embeddings):

        if idx == dp_idx:
            similarities.append(-1.0)
            continue
        similarity = 1 - cosine(test_sample_embedding, dp)
        similarities.append(similarity)

    top_k_indices = np.argsort(similarities)[-k:][::-1]
    return [test_data[i] for i in top_k_indices], top_k_indices , similarities

def _select_random_k_neighbors(test_sample_embedding, test_embeddings, test_data, k, dp_idx):
        
        length = len(test_data)
        candidates = [i for i in range(length) if i!= dp_idx]
        random_indices = random.sample(candidates, k)

        return [test_data[i] for i in random_indices]

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

# %%
accuracies = []
args.k = 3 # select three labeled
unlabeled_k = 3 
num_trials = 1

for dp_idx in range(len(test_data)):
    dp = test_data[dp_idx]
    input_tokens = tokenizer("Input: " + dp["input"] + " " + "Label: ")["input_ids"]
    output_tokens = tokenizer(dp["output"])["input_ids"]
    logger.info(dp['output'])
    logger.info(dp['options'])

    dp_feature = test_features[dp_idx]

    random_k_neighbors = _select_random_k_neighbors(
        dp_feature, test_features, test_data, args.k, dp_idx
    )

    metaicl_model.model.eval()
    metaicl_model.model.to(device)


    losses = []
    demonstrations = []
    # add unlabled examples 
    # candidates = [i for i in range(len(test_data)) if i!=dp_idx]
    # random.seed(trial)
    topk_data, topk_indices, __ = select_top_k_neighbors(
        dp_feature, test_features, test_data, unlabeled_k, dp_idx
    )
    for i in topk_indices:
        unlabel_dp = test_data[i]
        tmp_str = "Input: " + unlabel_dp["input"] + " " + "Label: " + "\n"
        demonstrations += tokenizer(tmp_str)["input_ids"]
    
    # add labled examples
    for i, neighbor_dp in enumerate(random_k_neighbors):
        tmp_str =  "Input: " + neighbor_dp["input"] + " " + "Label: " + neighbor_dp["output"] + "\n"
        demonstrations += tokenizer(tmp_str)["input_ids"]
    one_trial_losses = []
    for option_token in option_tokens:
        input_ids, results = run_a_forward_pass(demonstrations +    input_tokens, option_token, tokenizer)
        one_trial_losses.append(results)
    
    # if (trial+1) % 10 == 0:
    logger.info(one_trial_losses)
    min_loss = 1e6; max_accuracy = 0
    label = np.argmin(one_trial_losses)
    accuracy = int(dp["options"][label] == dp['output'])
    min_loss = min(min_loss, np.min(one_trial_losses))
    max_accuracy = max(max_accuracy, accuracy)



    logger.info("min_loss: %f, max_accuracy: %f", min_loss, max_accuracy)
    accuracies.append(max_accuracy)
    if (dp_idx + 1) % 10 == 0:
        logger.info("Avg. accuracy: {}".format(np.mean(accuracies)))

logger.info("Avg. accuracy: {}".format(np.mean(accuracies)))

# # %%
# losses = np.array(losses)

# prediction_data = [ ]

# for idx in range(100):
#     prediction_data.append({
#         "indices": idx,
#         "options": dp["options"]
#     })

# def do_predict(data, losses, verbose=False):

#     assert len(losses) == 100 
#     predictions = []
#     logger.info("len(data):", len(data))
#     for idx, dp in enumerate(data):
#         logger.info(f"Processing input {idx + 1}/{len(data)}")
#         logger.info("dp[\"options\"]:", dp["options"])

#         curr_label_losses = [
#             np.mean([losses[trial_idx] for trial_idx in indices])
#             for indices in dp["indices"]
#         ]

#         prediction_idx = sorted(enumerate(curr_label_losses), key=lambda x: x[1])[0][0]
#         prediction = dp["options"][prediction_idx]
#         predictions.append(prediction.strip())

#     return predictions

# predictions = do_predict(prediction_data, losses=losses, verbose=True)
# logger.info("Final Predictions for test_data[0]:", predictions)

# logger.info((losses < before_loss).sum())
# logger.info(min(losses))
# logger.info((before_loss-min(losses) )/before_loss)

# k = 2 Loss: 6.0350
# k = 4 Loss: 4.6970
# k = 8 Loss: 4.9098
# k = 16

# 0: 17 0.06885907971558475
# 1: 56 0.10330490071564696
# 2: 1 0.002612681340769363
# 3: 48 0.08768596101212324
