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
from transformers import modelTokenizer, AutoTokenizer

from gradsel.data import gradselData
from gradsel.model import gradselModel

from gradsel.utils.data import load_data

def main(logger, args):
    assert (args.dataset is not None and args.task is None) or (args.dataset is None and args.task is not None)

    if args.model.startswith("model"):
        tokenizer = modelTokenizer.from_pretrained(args.model)
    elif "Llama" or "deepseek" or "Qwen" in args.model:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    else:
        tokenizer = AutoTokenizer.from_pretrained("model")

    if tokenizer.padding_side=="left":
        tokenizer.padding_side = "right"
    add_newlines = True
    if "Llama" or "Qwen" in args.model:
        special_tokens = {
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "bos_token": "<bos>",
            "eos_token": "<eos>"
        }
        tokenizer.add_special_tokens(special_tokens)
        
    if not args.do_zeroshot:
        if args.checkpoint is not None:
            checkpoint = args.checkpoint
            assert args.global_step is None
        else:
            assert args.global_step is not None
            checkpoint = os.path.join(args.out_dir, "model-{}.pt".format(args.global_step))
        assert os.path.exists(checkpoint)
    else:
        add_newlines = not args.model.startswith("model")
        checkpoint = None
    
    gradsel_model = gradselModel(args.device, logger, args.out_dir,model=args.model)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    max_length_per_example = 128
    max_length = 128
    if args.use_demonstrations:
        orig_max_length = max_length
        if args.do_zeroshot:
            max_length = args.max_length * args.k
        else:
            max_length = min(max_length * args.k, 1024)

    if args.k==0: max_length = args.max_length
    logger.info("batch_size=%d\tmax_length=%d\tmax_length_per_example=%d" % (
        args.test_batch_size, max_length, max_length_per_example))

    gradsel_data = gradselData(args.device ,logger, tokenizer, args.method,args.use_demonstrations, args.k,
                               max_length=max_length, seed=args.seed, is_flops=args.is_flops)

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

        result = run(logger, test_task, gradsel_data, gradsel_model,
                     test_data, val_data, seed, checkpoint, is_classification, add_newlines, tokenizer)
        if result is None:
            errors.append("%s/%s" % (test_task, seed))
        else:
            results.append(result)

    if args.is_null:
        return
    if len(errors)>0:
        logger.info("You had errors with datasets:", ",".join(errors))
        logger.info("Please see the error messages")

def run(logger, task, gradsel_data, gradsel_model, test_data, val_data, seed,
        checkpoint, is_classification, add_newlines, tokenizer):

    if args.do_zeroshot:
        split_name = args.split
        if args.is_null:
            split_name += "-null"
        cache_path = os.path.join(args.out_dir,
                                  "{}-{}-{}{}{}{}{}{}{}{}{}{}{}{}{}.pkl".format(
                                      task,
                                      split_name,
                                      gradsel_data.method,
                                      "-topk" if args.topk else "",
                                      "-randomk" if args.randomk else "",
                                      "-ground" if args.ground else "",
                                      "-ranens" if args.ranens else "",
                                      "-forsel" if args.forsel else "",
                                      "-estim" if args.estim else "",
                                      "-k={}".format(args.k) if args.use_demonstrations else "",
                                      "-s={}".format(seed) if args.use_demonstrations or args.use_random_english_words else "",
                                      "" if add_newlines else "-no-newlines",
                                      "-m={}".format(args.m) if args.supcon or args.ranens or args.forsel or args.unlabeled or args.ground else ""))

    datapath = "./data/alldata.jsonl"
    train_split = 0.4
    train_data = val_data[int(len(val_data)*train_split):]
    val_data = val_data[:int(len(val_data)*train_split)]
    if args.topk:
        gradsel_data.tensorize_topk(test_data, val_data, options=None, add_newlines=add_newlines)
    elif args.randomk:
        gradsel_data.tensorize_randomk(test_data, val_data, options=None,  add_newlines=add_newlines)
    elif args.bm25:
        gradsel_data.tensorize_bm25(test_data, val_data, options=None,  add_newlines=add_newlines)
    elif args.ground:
        gradsel_data.tensorize_ground(args.model, test_data, val_data, options=None,  add_newlines=add_newlines)
    elif args.groundestim:
        gradsel_data.tensorize_ground(args.model, test_data, val_data, estimate=True, options=None,  add_newlines=add_newlines)
    elif args.forsel:
        gradsel_data.tensorize_estimate(args.model, test_data, val_data, args.is_quant, method="forsel", true_step=args.true_step, options=None,  add_newlines=add_newlines)
    elif args.ranens:
        gradsel_data.tensorize_estimate(args.model, test_data, val_data, args.is_quant, method="ranens", num_anchors=args.num_anchors, options=None,  add_newlines=add_newlines)

    gradsel_data.print_tensorized_example()
    logger.info(cache_path)
    prediction_path = cache_path.replace(".pkl", ".txt")
    if args.use_calibration:
        prediction_path = prediction_path.replace(".txt", "-calibrated.txt")

    if gradsel_model.is_none():
        gradsel_model.load(checkpoint, model=args.model, is_quant=args.is_quant)
        gradsel_model.cuda()
        gradsel_model.eval()
    if "Llama" in args.model:
        gradsel_model.resize(tokenizer)

    losses = []
    losses, flops = gradsel_model.do_inference(gradsel_data, args.test_batch_size, is_flops=args.is_flops)
    print(f"args.is_flops: {args.is_flops}, flops: {flops}")
    with open(cache_path, "wb") as f:
        pkl.dump(losses, f)

    logger.info(f"len(losses): {len(losses)}; len(gradsel_data): {len(gradsel_data)}")

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
    logger.info(f"Total_FLOPS: {(gradsel_data.total_flops+flops) / 1e9:.2f} GFLOPs")
    predictions = gradsel_model.do_predict(gradsel_data, losses=losses)
    groundtruths = [dp["output"] for dp in val_data]
    perf = gradsel_data.evaluate(predictions, groundtruths, is_classification)
    logger.info("Accuracy= %s", perf)

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
    parser.add_argument("--seed", type=str, default="0")

    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--global_step", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--use_random_english_words", default=False, action="store_true")

    parser.add_argument("--out_dir", type=str, default="out/model-large")

    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--is_null", default=False, action="store_true")
    parser.add_argument("--method", type=str, default="direct", choices=["direct", "channel"])
    parser.add_argument("--model", type=str, default="model-large")

    parser.add_argument("--topk",default=False, action="store_true")
    parser.add_argument("--randomk", default=False, action="store_true")
    parser.add_argument("--ground", default=False, action="store_true")
    parser.add_argument("--ranens", default=False, action="store_true")
    parser.add_argument("--forsel", default=False, action="store_true")
    parser.add_argument("--estim", default=False, action="store_true")
    parser.add_argument("--bm25", default=False, action="store_true")
    parser.add_argument("--groundestim", default=False, action="store_true")
    
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--is_quant", default=False, action="store_true")
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--is_flops", default=False, action="store_true")
    parser.add_argument("--num_anchors", default=5, type=int)
    parser.add_argument("--true_step", default=0, type=int)
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
