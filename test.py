# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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

def main(logger, args):
    assert (args.dataset is not None and args.task is None) or (args.dataset is None and args.task is not None)

    if args.gpt2.startswith("gpt2"):
        tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2)
    elif "Llama" in args.gpt2:
        tokenizer = AutoTokenizer.from_pretrained(args.gpt2)
    else:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    add_newlines = True
    if "Llama" in args.gpt2:
        special_tokens = {
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "bos_token": "<bos>",
            "eos_token": "<eos>"
        }

        tokenizer.add_special_tokens(special_tokens)
    ### checkpoint ...
    if not args.do_zeroshot:
        if args.checkpoint is not None:
            checkpoint = args.checkpoint
            assert args.global_step is None
        else:
            assert args.global_step is not None
            checkpoint = os.path.join(args.out_dir, "model-{}.pt".format(args.global_step))
        assert os.path.exists(checkpoint)
    else:
        add_newlines = not args.gpt2.startswith("gpt2")
        if False: #args.gpt2=="gpt-j-6B":
            # we are using the HF veresion where GPT-J-6B checkpoint is not officially registered
            # so need to download the model checkpoint and specify checkpoint
            assert args.checkpoint is not None and os.path.exists(args.checkpoint)
            args.gpt2 = args.checkpoint
        checkpoint = None
    
    metaicl_model = MetaICLModel(args.device, logger, args.out_dir)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # setup hyperparams for data

    max_length_per_example = 128
    max_length = 256
    if args.use_demonstrations:
        orig_max_length = max_length
        if args.do_zeroshot:
            max_length = min(max_length_per_example * args.k, 1024)
            # max_length = max_length * args.k
        else:
            max_length = min(max_length * args.k, 1024)

    logger.info("batch_size=%d\tmax_length=%d\tmax_length_per_example=%d" % (
        args.test_batch_size, max_length, max_length_per_example))

    metaicl_data = MetaICLData(args.device ,logger, tokenizer, args.method,args.use_demonstrations, args.k,
                               max_length, max_length_per_example)
    # metaicl_data.to(device)
    results = []
    errors = []
    seeds = args.seed.split(",")
    config_split = "unseen_domain_test" if args.unseen_domain_only else "test"

    for seed in seeds:

        test_data = load_data(args.task, "test", args.k, seed=seed, config_split=config_split,
                             datasets=None if args.dataset is None else args.dataset.split(","), is_null=args.is_null)
        val_data = load_data(args.task, "dev", args.k, seed=seed, config_split=config_split,
                             datasets=None if args.dataset is None else args.dataset.split(","), is_null=args.is_null)

        print("*"*20)
        print(f"args.split : {args.split}")

        test_task = test_data[0]["task"]

        config_file = "config/tasks/{}.json".format(test_task)
        assert os.path.exists(config_file), config_file
        with open(config_file, "r") as f:
            config = json.load(f)
        is_classification = config["task_type"]=="classification"
        if is_classification:
            options = test_data[0]["options"]
            assert np.all([d["options"]==options for d in test_data])
        result = run(logger, test_task, metaicl_data, metaicl_model,
                     test_data, val_data, seed, checkpoint, is_classification, add_newlines, tokenizer)
        if result is None:
            errors.append("%s/%s" % (test_task, seed))
        else:
            results.append(result)

    if args.is_null:
        return

    logger.info("Macro-F1 of %s over %d target tasks: %.1f" % (args.task, len(results) // len(seeds), 100*np.mean(results)))

    if len(errors)>0:
        logger.info("You had errors with datasets:", ",".join(errors))
        logger.info("Please see the error messages")


def run(logger, task, metaicl_data, metaicl_model, test_data, val_data, seed,
        checkpoint, is_classification, add_newlines, tokenizer):

    if args.do_zeroshot:
        split_name = args.split
        if args.is_null:
            split_name += "-null"
        cache_path = os.path.join(args.out_dir,
                                  "{}-{}-{}{}{}{}{}{}{}{}{}{}{}{}{}.pkl".format(
                                      task,
                                      split_name,
                                      metaicl_data.method,
                                      "-topk" if args.topk else "",
                                      "-randomk" if args.randomk else "",
                                      "-supcon" if args.supcon else "",
                                      "-ground" if args.ground else "",
                                      "-unlabeled" if args.unlabeled else "",
                                      "-ranens" if args.ranens else "",
                                      "-forsel" if args.forsel else "",
                                      "-estim" if args.estim else "",
                                      "-k={}".format(args.k) if args.use_demonstrations else "",
                                      "-s={}".format(seed) if args.use_demonstrations or args.use_random_english_words else "",
                                      "" if add_newlines else "-no-newlines",
                                      "-m={}".format(args.m) if args.supcon or args.ranens or args.forsel or args.unlabeled or args.ground else ""))

    datapath = "./data/alldata.jsonl"
    if args.topk:
        metaicl_data.tensorize_topk(test_data, val_data, options=None, add_newlines=add_newlines)
    elif args.randomk:
        metaicl_data.tensorize_randomk(test_data, val_data, options=None,  add_newlines=add_newlines)
    elif args.supcon:
        metaicl_data.tensorize_supcon(test_data, val_data, args.m, options=None,  add_newlines=add_newlines)
    elif args.ground:
        metaicl_data.tensorize_ground(args.gpt2, test_data, val_data, options=None,  add_newlines=add_newlines)
    elif args.unlabeled:
        metaicl_data.tensorize_unlabeled(args.gpt2, test_data, val_data,args.m, options=None,  add_newlines=add_newlines)
    elif args.multidata:
        metaicl_data.tensorize_multidata(test_data, val_data, datapath, args.m, options=None, add_newlines=add_newlines)
    elif args.ranens:
        metaicl_data.tensorize_ranens(test_data, val_data, args.m, args.seed, options=None, add_newlines=add_newlines)
    elif args.forsel:
        metaicl_data.tensorize_forsel(test_data, val_data, args.m, args.seed, options=None, add_newlines=add_newlines)
    elif args.estim:
        metaicl_data.tensorize_estimate(args.gpt2, test_data, val_data, args.is_quant, options=None,  add_newlines=add_newlines)

    metaicl_data.print_tensorized_example()
    logger.info(cache_path)
    prediction_path = cache_path.replace(".pkl", ".txt")
    if args.use_calibration:
        prediction_path = prediction_path.replace(".txt", "-calibrated.txt")


    # if os.path.exists(cache_path):
    #     with open(cache_path, "rb") as f:
    #         losses = pkl.load(f)
    # else:
    if metaicl_model.is_none():
        metaicl_model.load(checkpoint, gpt2=args.gpt2, is_quant=args.is_quant)
        metaicl_model.cuda()
        metaicl_model.eval()

    if "Llama" in args.gpt2:
        metaicl_model.resize(tokenizer)

    losses = metaicl_model.do_inference(metaicl_data, args.test_batch_size)
        
    with open(cache_path, "wb") as f:
        pkl.dump(losses, f)

    logger.info(f"len(losses): {len(losses)}; len(metaicl_data): {len(metaicl_data)}")
    # assert len(losses)==len(metaicl_data)

    if args.is_null:
        return None

    if args.use_calibration:
        assert args.do_zeroshot
        bias_path = cache_path.replace("/"+task+"-"+args.split, "/"+task+"-"+args.split+"-null")
        assert os.path.exists(bias_path), bias_path
        with open(bias_path, "rb") as f:
            bias_losses = pkl.load(f)

        losses = np.array(losses)
        bias_losses = np.array(bias_losses)
        assert losses.shape == bias_losses.shape
        losses -= bias_losses

    predictions = metaicl_model.do_predict(metaicl_data, losses=losses)
    groundtruths = [dp["output"] for dp in val_data]
    perf = metaicl_data.evaluate(predictions, groundtruths, is_classification)
    logger.info("Accuracy=%s" % perf)

    with open(prediction_path, "w") as f:
        for prediction in predictions:
            f.write(prediction)
            f.write("\n")

    return perf

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--do_zeroshot", default=False, action="store_true")
    parser.add_argument("--use_demonstrations", default=False, action="store_true")
    parser.add_argument("--use_calibration", default=False, action="store_true")
    parser.add_argument("--unseen_domain_only", default=False, action="store_true")

    parser.add_argument("--log_file", default=None, type=str)

    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--seed", type=str, default="100")

    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--global_step", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--use_random_english_words", default=False, action="store_true")

    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--is_null", default=False, action="store_true")
    parser.add_argument("--method", type=str, default="direct", choices=["direct", "channel"])
    parser.add_argument("--gpt2", type=str, default="gpt2-large")

    parser.add_argument("--topk",default=False, action="store_true")
    parser.add_argument("--randomk", default=False, action="store_true")
    parser.add_argument("--supcon", default=False, action="store_true")
    parser.add_argument("--ground", default=False, action="store_true")
    parser.add_argument("--unlabeled", default=False, action="store_true")
    parser.add_argument("--multidata", default=False, action="store_true")
    parser.add_argument("--ranens", default=False, action="store_true")
    parser.add_argument("--forsel", default=False, action="store_true")
    parser.add_argument("--estim", default=False, action="store_true")
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--is_quant", default=False, action="store_true")
    args = parser.parse_args()

    handlers = [logging.StreamHandler()]
    if args.log_file is not None:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=handlers)
    logger = logging.getLogger(__name__)
    logger.info(args)

    main(logger, args)
