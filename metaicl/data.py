# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import csv
import json
import logging
import string
import numpy as np
import pickle as pkl
import math
import torch
import random
from itertools import combinations
import warnings
import torch.nn.functional as F
warnings.filterwarnings("ignore", category=UserWarning)
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2LMHeadModel, GPT2Tokenizer, OPTForCausalLM
from tqdm import tqdm
from thop import profile

from rank_bm25 import BM25Okapi
logging.getLogger("thop").setLevel(logging.WARNING)

# import sys
# import os
# sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
# from forward_path import Forward

# import nltk
# from nltk.corpus import wordnet
import random

# from data_augment import DataAugment

from collections import defaultdict
from functools import partial
from multiprocessing import Pool

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from sklearn.linear_model import LinearRegression


from utils.data import load_data
from metaicl.model import MetaICLModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


class OpenLLMEvaluator:
    def __init__(self, model_name="deepseek-ai/deepseek-llm-7b-chat"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.model.generation_config = GenerationConfig.from_pretrained(model_name)
        self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id

    def query(self, question: str, examples: list = []) -> str:
        messages = []
        for inp, out in examples:
            messages.append({"role": "user", "content": inp})
            messages.append({"role": "assistant", "content": out})
        messages.append({"role": "user", "content": question})

        input_tensor = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(input_tensor, max_new_tokens=100)
        result = self.tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
        return result.strip()


from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

class OpenLLMEvaluator:
    def __init__(self, model_name="deepseek-ai/deepseek-llm-7b-chat"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.model.generation_config = GenerationConfig.from_pretrained(model_name)
        self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id

    def query(self, question: str, examples: list = [], is_flops=False) -> str:
        messages = []
        messages_text = ""
        for inp, out in examples:
            messages_text+=inp+out
            messages.append({"role": "user", "content": inp})
            messages.append({"role": "assistant", "content": out})
        messages.append({"role": "user", "content": question})

        input_tensor = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        input_ids = self.tokenizer(messages_text)["input_ids"]
        flops=0
        if is_flops:
            flops, params = profile(self.model, inputs=(input_ids,))

        outputs = self.model.generate(input_tensor, max_new_tokens=100)
        result = self.tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
        return result.strip(), flops

def _get_embedding_loss(model, tokenizer, input_texts, pad_to_length, is_flops=False):
    model = model.model
    tokenizer.padding_side = "right"

    inputs = tokenizer(input_texts, padding="max_length", return_tensors='pt', truncation=True, max_length=pad_to_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        embedding = model.model.embed_tokens(inputs['input_ids'])
    embedding.requires_grad = True 
    embedding = embedding.to(model.dtype)

    outputs = model(inputs_embeds=embedding)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = inputs["input_ids"][..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=tokenizer.pad_token_id)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(shift_labels.size())

    lens = (inputs["input_ids"] != tokenizer.pad_token_id).sum(-1)
    ce_loss = loss.sum(-1) / lens

    ce_loss.backward()
    embedding_grad = embedding.grad

    effective_embedding_grad = embedding_grad[:, :-1, :]

    flops=0
    if is_flops:
        flops, params = profile(model, inputs=(inputs['input_ids'],))

    return ce_loss, effective_embedding_grad, flops

def _get_embedding_loss_(model, tokenizer, input_texts, pad_to_length):
    model = model.model
    tokenizer.padding_side = "right"
    inputs = tokenizer(input_texts, padding=True, return_tensors='pt', truncation=True, max_length=pad_to_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        embedding = model.model.embed_tokens(inputs['input_ids'])
    embedding.requires_grad = True 
    embedding = embedding.to(model.dtype)

    outputs = model(inputs_embeds=embedding)

    #outputs = model(**inputs)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = inputs["input_ids"][..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=tokenizer.pad_token_id)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(
        shift_labels.size())

    lens = (inputs["input_ids"] != tokenizer.pad_token_id).sum(-1)

    #ce_loss = loss.sum(-1).cpu().detach().numpy() / lens
    #ce_loss = loss.sum(-1).cpu().detach().numpy()
    ce_loss = loss.sum(-1) / lens

    ce_loss.backward()
    embedding_grad = embedding.grad
    return ce_loss, embedding_grad

class MetaICLData(object):

    def __init__(self, device=0, logger=None, tokenizer=None, method="channel", use_demonstrations=True, k=16,
                 max_length=1024, max_length_per_example=256,
                 do_tensorize=False, tensorize_dir=None, seed=0, n_process=None, n_gpu=None, local_rank=-1, is_flops=False):

        self.logger = logger
        self.tokenizer = tokenizer
        self.method = method
        self.use_demonstrations = use_demonstrations
        self.k = k
        self.max_length = max_length
        self.max_length_per_example = max_length_per_example

        self.do_tensorize = do_tensorize
        self.tensorize_dir = tensorize_dir
        self.n_process = n_process
        self.n_gpu = n_gpu
        self.local_rank = local_rank

        self.tensorized_inputs = None
        self.metadata = None
        self.device = device
        self.is_null = False
        self.is_flops = is_flops
        self.seed =seed
        self.total_flops=0


        #print(tokenizer)

        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def __len__(self):
        if self.tensorized_inputs is None:
            return 0
        return len(self.tensorized_inputs["input_ids"])

    def __str__(self):
        text = "[MetaICL Data]: method=%d, "
        if self.use_demonstrations:
            text += "%d demonstrations\n" % self.k
        else:
            text += "no demonstrations\n"
        if self.metadata is None:
            text += "Currently not containing any examples"
        else:
            text += "Currently containing %d examples with %d tensors to be fed in\n" % (len(self.metadata), len(self))
            text += "\n"
            text += self.print_tensorized_example(return_string=True)
        return ("="*50) + "\n" + text + "\n" + ("="*50)

    def forward(self, gpt2, metaicl_model, demonstrations, dp, task, return_loss = False):
        logger = logging.getLogger(__name__)
        device = torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")
        tokenizer = self.tokenizer

        #max_length_per_example, max_length = 128, 128
        max_length_per_example = self.max_length_per_example
        max_length = self.max_length
        if self.use_demonstrations:
            max_length = min(max_length * self.k, 1024)

        def run_a_forward_pass(input_tokens, output_tokens, tokenizer):
            encoded = prepro_sentence_pair_single(
                        input_tokens, output_tokens, max_length=1024, tokenizer=self.tokenizer, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
                        allow_truncation=self.use_demonstrations
                )
            input_ids = torch.LongTensor([encoded[0]])
            attention_mask = torch.LongTensor([encoded[1]])
            token_type_ids = torch.LongTensor([encoded[2]])
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            results, _ = metaicl_model.run_model(input_ids, attention_mask, token_type_ids)

            if self.is_flops:
                self.logger.info(f"len(input_ids): {input_ids.size()}")
                flops, params = profile(metaicl_model.model, inputs=(input_ids,))
            else: flops =0

            return input_ids, results.cpu().detach().item(), flops

        option_tokens = [tokenizer(option)["input_ids"] for option in dp['options']]
        input_tokens = tokenizer(dp["input"] + " ")["input_ids"]
        metaicl_model.model.eval()
        # metaicl_model.model.to(device)

        one_trial_losses = []

        total_flops = 0
        for option_token in option_tokens:
            input_ids, results, flops = run_a_forward_pass(demonstrations + input_tokens, option_token, tokenizer)
            total_flops+=flops
            # if self.is_flops: self.logger.info(f"----- flops : {flops / 1e9:.2f} GFLOPs")
            one_trial_losses.append(results)

        idx = dp["options"].index(dp["output"])
        label_id = np.argmin(one_trial_losses)
        label = dp["options"][label_id]
        if return_loss:
            return one_trial_losses[idx], total_flops
        return label_id, label, total_flops

    def random_ensemble(self, gpt2, metaicl_model, test_data, dev_data, num_combinations=100, k=8, seed=42, num_anchors=1):
        random.seed(seed)
        device = torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")

        self.logger.info(f"number of anchors: {num_anchors}")
        all_indices = list(range(len(test_data)))

        # Efficient sampling
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

        sampled_combinations = sample_unique_combinations(all_indices, k, num_combinations, seed=seed)

        # Step 1: Choose 5 anchor combinations and precompute
        anchor_combs = random.sample(sampled_combinations, num_anchors)
        anchor_info = {}

        total_flops = 0
        for anchor in tqdm(anchor_combs):
            anchor_prompt = "".join([
                f"Input: {test_data[idx]['input']} Label: {test_data[idx]['output']}\n" for idx in anchor
            ])
            base_losses, base_gradients, _, flops = zip(*[
                self.forward_estim(gpt2, metaicl_model, anchor_prompt, dp, dp["task"], return_loss=True)
                for dp in dev_data
            ])
            total_flops += sum(flops)

            loss_tensor = torch.tensor(base_losses, device=device)
            grad_tensor = torch.stack([torch.stack(g, dim=0) for g in base_gradients], dim=0)  # [len(dev), num_labels, D]
            anchor_info[anchor] = (anchor_prompt, loss_tensor, grad_tensor)

        # Step 2: For each non-anchor combination, estimate with closest anchor
        accuracy_results = []
        for comb in tqdm(sampled_combinations, total=len(sampled_combinations)):
            if comb in anchor_combs:
                continue

            # Find anchor with most overlap
            best_anchor = max(anchor_combs, key=lambda a: len(set(a) & set(comb)))
            anchor_prompt, base_loss_tensor, grad_tensor = anchor_info[best_anchor]

            target_prompt = "".join([
                f"Input: {test_data[idx]['input']} Label: {test_data[idx]['output']}\n" for idx in comb
            ])

            correct = 0
            for dp_idx, dp in enumerate(dev_data):
                dev_str = f"Input: {dp['input']} Label:"
                delta_P = self.compute_embedding_difference_(
                    gpt2, metaicl_model, anchor_prompt + dev_str, target_prompt + dev_str
                )

                taylor_losses = []
                for j in range(len(base_loss_tensor[dp_idx])):
                    correction = torch.sum(grad_tensor[dp_idx][j] * delta_P).item()
                    approx_loss = base_loss_tensor[dp_idx][j].item() + correction
                    taylor_losses.append(approx_loss)

                pred_id = np.argmin(taylor_losses)
                pred = dp["options"][pred_id]
                if pred == dp["output"]:
                    correct += 1

            acc = correct / len(dev_data)
            accuracy_results.append((comb, acc))

        # Step 3: Score each sample by average accuracy of all combinations it is in
        point_scores = defaultdict(list)
        for comb, acc in accuracy_results:
            for idx in comb:
                point_scores[idx].append(acc)

        avg_scores = []
        for idx, scores in point_scores.items():
            avg_scores.append((idx, sum(scores) / len(scores)))

        avg_scores.sort(key=lambda x: -x[1])
        final_indices = [idx for idx, _ in avg_scores[:k]]
        selected_data = [test_data[i] for i in final_indices]

        return selected_data, accuracy_results[0][1], total_flops


    def tensorize_bm25(self, _test_data, _val_data, options=None, add_newlines=True):
        if options is not None:
            for i, dp in enumerate(_test_data):
                _test_data[i] = {"input": dp, "options": options}
            for i, dp in enumerate(_val_data):
                _val_data[i] = {"input": dp, "options": options}

        val_data, test_data = [], []
        for dp in _test_data:
            if "output" not in dp:
                dp["output"] = dp["options"][0]
            test_data.append(dp.copy())
        for dp in _val_data:
            if "output" not in dp:
                dp["output"] = dp["options"][0]
            val_data.append(dp.copy())

        task = _test_data[0]["task"]
        input_ids, attention_mask, token_type_ids = [], [], []
        metadata = []

        test_inputs = [dp["input"].split() for dp in test_data]
        bm25 = BM25Okapi(test_inputs)

        instructions = f"Here are {len(test_data[0]['options'])} options: "
        for option in test_data[0]["options"]:
            instructions += option + ", "
        instructions += f"You should choose one of them to answer at the end. \nHere are {self.k} samples for your reference. \n"
        init_tokens = self.tokenizer(instructions)["input_ids"][1:]

        for dp_idx, dp in enumerate(val_data):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)

            if self.use_demonstrations:
                # query_terms = dp["input"].split()
                # scores = bm25.get_scores(query_terms)
                # scores[dp_idx] = -1e9

                # topk_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.k]
                # top_k_neighbors = [test_data[i] for i in topk_indices]
                top_k_neighbors, _, __ = self._select_top_k_neighbors_bm25(dp["input"], test_data, self.k)
                demonstrations = init_tokens
                for i, neighbor_dp in enumerate(top_k_neighbors):
                    input_, output_ = self._prepro_each_datapoint(
                        neighbor_dp, is_first=i == 0, for_demonstrations=True, add_newlines=add_newlines)
                    demonstrations += input_[1:] + output_[1:]

            indices = [[i] for i in range(len(input_ids), len(input_ids) + len(inputs))]

            metadata.append({"indices": indices, "answer": answer, "options": dp["options"]})
            for inputs_, outputs_ in zip(inputs, outputs):
                if self.use_demonstrations:
                    inputs_ = demonstrations + inputs_[1:]
                encoded = prepro_sentence_pair_single(
                    inputs_, [outputs_[2]], self.max_length, self.tokenizer,
                    self.tokenizer.bos_token_id, self.tokenizer.eos_token_id,
                    allow_truncation=self.use_demonstrations)
                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])

        self.tensorized_inputs = dict(
            input_ids=torch.LongTensor(input_ids),
            attention_mask=torch.LongTensor(attention_mask),
            token_type_ids=torch.LongTensor(token_type_ids))
        self.metadata = metadata


    def evaluate_accuracy(self, gpt2, metaicl_model, demonstrations, dev_data, task):
        correct = 0; total = len(dev_data)
        input_str = ""
        for item in demonstrations:
            input_str = input_str + item["input"] + " "+ "Label: "+item["output"]+"\n"
        input_token = self.tokenizer(input_str)["input_ids"]
        total_flops=0
        for idx, dp in enumerate(dev_data):
            _, label, flops = self.forward(gpt2, metaicl_model, input_token, dp, task)
            if self.is_flops: self.logger.info(f"----- flops : {flops / 1e9:.2f} GFLOPs")
            total_flops+=flops
            if label == dp["output"]: correct += 1
        return correct / total if total > 0 else 0 , total_flops

    def evaluate_loss(self, gpt2, metaicl_model, demonstrations, dev_data, task):
        total = len(dev_data)
        input_str = ""
        all_loss = 0.0
        for item in demonstrations:
            input_str = input_str + item["input"] + " "+ "Label: "+item["output"]+"\n"
        input_token = self.tokenizer(input_str)["input_ids"]
        total_flops =0
        for idx, dp in enumerate(dev_data):
            loss, flops = self.forward(gpt2, metaicl_model, input_token, dp, task, return_loss=True)
            total_flops+=flops
            if self.is_flops: self.logger.info(f"----- flops : {flops / 1e9:.2f} GFLOPs")
            all_loss+=loss
        return all_loss, total_flops

    def greedy_select_subset(self, gpt2, test_data, dev_data, subset_size=10):
        selected_indices, best_demonstrations = [], []
        best_demonstrations = []

        add_newlines = False
        checkpoint = None
        metaicl_model = MetaICLModel(logger=self.logger, out_dir= "./cache", device_num=self.device)
        metaicl_model.load(checkpoint, gpt2=gpt2)
        metaicl_model.resize()

        while len(selected_indices) < self.k:
            base_index = next(i for i in range(len(test_data)) if i not in selected_indices)
            best_candidate = base_index
            best_accuracy = self.evaluate_accuracy(gpt2, metaicl_model, best_demonstrations+[test_data[base_index]], dev_data, test_data[0]["task"])
            for i in range(len(test_data)):
                if (i in selected_indices) or i==base_index: continue
                candidate_demonstrations = best_demonstrations + [test_data[i]]
                candidate_accuracy = self.evaluate_accuracy(gpt2, metaicl_model, candidate_demonstrations, dev_data, test_data[0]["task"])
                
                self.logger.info(f"----------------candidate_accuracy : {candidate_accuracy}----------------")
                if candidate_accuracy > best_accuracy:
                    best_candidate = i
                    best_accuracy = candidate_accuracy

            selected_indices.append(best_candidate)
            self.logger.info("**"*20)
            self.logger.info(f"selected_indices: {selected_indices}; best_candidate : {best_candidate}")
            best_demonstrations.append(test_data[best_candidate])
        return best_demonstrations, best_accuracy

    def greedy_select_condition(self, gpt2, metaicl_model, test_data, dev_data, subset_size=10):
            loss_index_pairs = []
            total_flops = 0

            for i in range(len(test_data)):
                demonstrations = [test_data[i]]
                loss, flops = self.evaluate_loss(gpt2, metaicl_model, demonstrations, dev_data, test_data[0]["task"])
                self.logger.info(f"Index {i} - Loss: {loss}")
                total_flops += flops
                loss_index_pairs.append((loss, i))

            topk = sorted(loss_index_pairs, key=lambda x: x[0])[:self.k]
            selected_indices = [i for _, i in topk]
            best_demonstrations = [test_data[i] for i in selected_indices]
            best_loss = sum([l for l, _ in topk]) / self.k

            self.logger.info("== Selected Indices ==")
            self.logger.info(selected_indices)

            return best_demonstrations, best_loss, total_flops

    def greedy_select_condition_estim(self, gpt2, metaicl_model, test_data, dev_data):
        def build_text(prefix_text, base_sample, query_sample, op):
            return prefix_text + base_sample["input"] + base_sample["output"] + query_sample["input"] + op + "\n"
        
        self.options = test_data[0]["options"]
        prompt_text = ""
        device = torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")
        max_token_len = max(
            len(self.tokenizer(build_text("", ex, dev_data[0], self.options[0]), return_tensors="pt")["input_ids"][0])
            for ex in test_data
        )

        approx_loss_list = [0.0 for _ in range(len(test_data))]
        total_flops=0
        for query in dev_data:
            loss_option_dict, gradient_option_dict = {}, {}

            for op in self.options:
                base_text = build_text(prompt_text, test_data[0], query, op)
                loss_op, embedding_grad_op, flops = _get_embedding_loss(
                    model=metaicl_model, tokenizer=self.tokenizer, input_texts=[base_text], pad_to_length=max_token_len, is_flops=self.is_flops
                )
                loss_option_dict[op] = loss_op
                gradient_option_dict[op] = embedding_grad_op
                total_flops += flops

            for i, candidate_sample in enumerate(test_data):
                candidate_approx_loss_dict = {}
                for op in self.options:
                    candidate_text = build_text(prompt_text, candidate_sample, query, op)
                    delta_P = self.compute_embedding_difference(
                        gpt2, metaicl_model, base_str=base_text, candidate_str=candidate_text, pad_to_length=max_token_len
                    )
                    taylor_correction = torch.sum(gradient_option_dict[op] * delta_P).item()
                    taylor_approx_loss = loss_option_dict[op] + taylor_correction
                    candidate_approx_loss_dict[op] = taylor_approx_loss

                approx_loss_list[i] += candidate_approx_loss_dict[query["output"]].cpu().item()
        
        for idx, loss in enumerate(approx_loss_list):
            self.logger.info(f"Index {idx} - Loss: {loss}")
        topk_indices = sorted(range(len(test_data)), key=lambda i: approx_loss_list[i])[:self.k]

        self.logger.info(f"selected indices: {topk_indices}")

        best_demonstrations = [test_data[i] for i in topk_indices]

        # self.logger.info(f"Selected indices (top-k by loss): {topk_indices}")
        return best_demonstrations, None, total_flops


    def tensorize_ground(self, gpt2, _test_data, _val_data, estimate=False, options=None, add_newlines=True):
        print("options: ", options)
        if options is not None:
            print("len(_test_data) : ", len(_test_data))
            print(_test_data[0])
            for i, dp in enumerate(_test_data):
                assert "options" not in dp,print(i,dp)
                _test_data[i] = {"input": dp, "options": options}
            for i, dp in enumerate(_val_data):
                assert "options" not in dp
                _val_data[i] = {"input": dp, "options": options}
        print("len(_test_data) : ",len(_test_data))
        print("len(_val_data) : ", len(_val_data))

        val_data, dev_data, test_data = [], [], []
        for dp in _test_data:
            if "output" not in dp: dp["output"] = dp["options"][0]
            test_data.append(dp.copy())
        for idx, dp in enumerate(_val_data):
            if "output" not in dp: dp["output"] = dp["options"][0]
            if idx<= 15: dev_data.append(dp.copy())
            val_data.append(dp.copy())
        task = _test_data[0]["task"]
        with open(f"./features/{task}_test_features.json", "r") as file: test_features = json.load(file)
        with open(f"./features/{task}_val_features.json", "r") as file: val_features = json.load(file)

        add_newlines = False
        checkpoint = None
        metaicl_model = MetaICLModel(logger=self.logger, out_dir= "./cache", device_num=self.device)
        metaicl_model.load(checkpoint, gpt2=gpt2)
        if "Llama" in gpt2:
            metaicl_model.resize(self.tokenizer)

        input_ids, attention_mask, token_type_ids = [], [], []
        metadata = []
        total_flops = 0
        for dp_idx, dp in tqdm(enumerate(val_data), total=len(val_data)):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)
            if self.use_demonstrations:
                test_text = dp["input"]
                dp_feature = val_features[dp_idx]

                samples, top_indices, _ = self._select_top_k_neighbors(dp_feature, test_features, test_data, k=20,dp_idx=-1)

                if estimate==False:
                    ground, _, flops = self.greedy_select_condition(gpt2=gpt2, metaicl_model=metaicl_model,test_data=samples, dev_data=dev_data, subset_size=self.k)
                else:
                    ground, _, flops = self.greedy_select_condition_estim(gpt2=gpt2, metaicl_model=metaicl_model,test_data=samples, dev_data=dev_data)

                total_flops+=flops
                # def greedy_select_subset(self, test_data, dp_data, subset_size=10):
                demonstrations = []
                for i, neighbor_dp in enumerate(ground):
                    input_, output_ = self._prepro_each_datapoint(
                        neighbor_dp, is_first=i == 0, for_demonstrations=True, add_newlines=add_newlines)
                    demonstrations += input_ + output_

            indices = [[i] for i in range(len(input_ids), len(input_ids) + len(inputs))]

            metadata.append({"indices": indices, "answer": answer, "options": dp["options"]})

            for inputs_, outputs_ in zip(inputs, outputs):
                if self.use_demonstrations:
                    inputs_ = demonstrations + inputs_
                encoded = prepro_sentence_pair_single(
                    inputs_, outputs_, self.max_length, self.tokenizer, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id,
                    allow_truncation=self.use_demonstrations
                )
                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])
        
        if self.is_flops: 
            self.total_flops=total_flops
            # self.logger.info(f"Total_FLOPS: {total_flops / 1e9:.2f} GFLOPs")

        self.tensorized_inputs = dict(input_ids=torch.LongTensor(input_ids),
                                      attention_mask=torch.LongTensor(attention_mask),
                                      token_type_ids=torch.LongTensor(token_type_ids))
        self.metadata = metadata

    def _select_top_k_neighbors_bm25(self, query_text, candidate_data, k):
        corpus = [dp["input"] for dp in candidate_data]
        tokenized_corpus = [doc.split() for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query_text.split()
        scores = bm25.get_scores(tokenized_query)
        topk_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [candidate_data[i] for i in topk_indices], topk_indices, scores

    def tensorize_uncertainty_rank(self, _test_data, _val_data, options=None, add_newlines=True):
        if options is not None:
            for i, dp in enumerate(_test_data):
                _test_data[i] = {"input": dp, "options": options}
            for i, dp in enumerate(_val_data):
                _val_data[i] = {"input": dp, "options": options}

        val_data, dev_data, test_data = [], [], []
        for dp in _test_data:
            if "output" not in dp:
                dp["output"] = dp["options"][0]
            test_data.append(dp.copy())
        for idx, dp in enumerate(_val_data):
            if "output" not in dp:
                dp["output"] = dp["options"][0]
            if idx <= 16:
                dev_data.append(dp.copy())
            val_data.append(dp.copy())

        task = _test_data[0]["task"]
        with open(f"./features/{task}_test_features.json", "r") as file:
            test_features = json.load(file)
        with open(f"./features/{task}_val_features.json", "r") as file:
            val_features = json.load(file)

        llm = OpenLLMEvaluator()
        input_ids, attention_mask, token_type_ids = [], [], []
        metadata = []

        sample_scores = {}
        for idx, dev_dp in tqdm(enumerate(dev_data), total=len(dev_data)):
            # candidates, _, _ = self._select_top_k_neighbors_bm25(dev_dp["input"], test_data, k=5)

            dp_feature = val_features[idx]
            candidates, _, __ = self._select_top_k_neighbors(
                dp_feature, test_features, test_data, 10, idx
            )
            rewards = []
            for j in range(len(candidates)):
                prompt_examples = [(x["input"], x["output"]) for x in candidates[:j+1]]
                pred, flops = llm.query(dev_dp["input"], prompt_examples)
                self.total_flops+=flops
                reward = int(pred.strip().lower() == dev_dp["output"].strip().lower()) * 2 - 1
                rewards.append(reward)
            for j, r in enumerate(rewards):
                idx = j
                cand = candidates[idx]
                key = cand["input"] + "||" + cand["output"]
                sample_scores[key] = sample_scores.get(key, 0) + r

        for dp_idx, dp in tqdm(enumerate(val_data), total=len(val_data)):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)

            if self.use_demonstrations:
                # candidates, _, _ = self._select_top_k_neighbors_bm25(dp["input"], test_data, k=5)
                dp_feature = val_features[dp_idx]
                candidates, _, __ = self._select_top_k_neighbors(
                    dp_feature, test_features, test_data, 10, dp_idx
                )
                
                # for cand in candidates:
                #     key = cand["input"] + "||" + cand["output"]
                #     cand["score"] = sample_scores.get(key, -999)
                # sorted_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
                # selected = sorted_candidates[:self.k]
                selected = candidates
                demonstrations = []
                for i, neighbor_dp in enumerate(selected):
                    input_, output_ = self._prepro_each_datapoint(
                        neighbor_dp, is_first=i == 0, for_demonstrations=True, add_newlines=add_newlines)
                    demonstrations += input_ + output_

            indices = [[i] for i in range(len(input_ids), len(input_ids) + len(inputs))]
            metadata.append({"indices": indices, "answer": answer, "options": dp["options"]})

            for inputs_, outputs_ in zip(inputs, outputs):
                if self.use_demonstrations:
                    inputs_ = demonstrations + inputs_
                encoded = prepro_sentence_pair_single(
                    inputs_, outputs_, self.max_length, self.tokenizer,
                    self.tokenizer.bos_token_id, self.tokenizer.eos_token_id,
                    allow_truncation=self.use_demonstrations
                )
                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])

        self.tensorized_inputs = dict(
            input_ids=torch.LongTensor(input_ids),
            attention_mask=torch.LongTensor(attention_mask),
            token_type_ids=torch.LongTensor(token_type_ids)
        )
        self.metadata = metadata

    def compute_loss_and_gradient(self, gpt2, metaicl_model, tokenizer, input_tokens, output_tokens, device):

        tokenizer.pad_token = tokenizer.eos_token
        input_ids = tokenizer(input_tokens, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length).input_ids.to(device)
        output_ids = tokenizer(output_tokens, return_tensors="pt")["input_ids"][0][-1].to(device)

        with torch.no_grad():
            if "gpt" in gpt2:
                embedding = metaicl_model.model.transformer.wte(input_ids)
            elif "opt" in gpt2: embedding = metaicl_model.model.model.decoder.embed_tokens(input_ids)
            else:
                embedding = metaicl_model.model.model.embed_tokens(input_ids)
        embedding.requires_grad = True 
        embedding = embedding.to(metaicl_model.model.dtype)

        output_logits = metaicl_model.model(inputs_embeds=embedding).logits
        last_token_idx = input_ids.shape[1] - 1 
        log_probs = F.log_softmax(output_logits[0, last_token_idx, :], dim=-1) 

        target_token = output_ids
        loss = -log_probs[target_token]
        
        flops = 0
        if self.is_flops:
            flops, params = profile(metaicl_model.model, inputs=(input_ids,))
            self.logger.info(f"----- flops : {flops / 1e9:.2f} GFLOPs")
        
        loss.backward()
        return loss.item(), embedding.grad, flops

    def compute_loss_and_gradient_op(self, gpt2, metaicl_model, tokenizer, input_tokens, output_tokens, device):

        tokenizer.pad_token = tokenizer.eos_token

        input_ids = tokenizer(input_tokens, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length).input_ids.to(device)
        output_ids = tokenizer(output_tokens, return_tensors="pt")["input_ids"][0][-1].to(device)

        with torch.no_grad():
            if "gpt" in gpt2:
                embedding = metaicl_model.model.transformer.wte(input_ids)
            elif "opt" in gpt2: embedding = metaicl_model.model.model.decoder.embed_tokens(input_ids)
            else:
                embedding = metaicl_model.model.model.embed_tokens(input_ids)
        embedding.requires_grad = True 
        embedding = embedding.to(metaicl_model.model.dtype)

        output_logits = metaicl_model.model(inputs_embeds=embedding).logits
        last_token_idx = input_ids.shape[1] - 1 
        log_probs = F.log_softmax(output_logits[0, last_token_idx, :], dim=-1) 

        target_token = output_ids
        loss = -log_probs[target_token]
        
        if self.is_flops: 
            flops, params = profile(metaicl_model.model, inputs=(input_ids,))
            self.logger.info(f"----- flops : {flops / 1e9:.2f} GFLOPs")
        
        loss.backward()
        return loss.item(), embedding.grad, flops

    def forward_estim(self, gpt2, metaicl_model, demonstrations, dp, task, return_loss=False):

        logger = logging.getLogger(__name__)
        device = torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")
        tokenizer = self.tokenizer

        option_tokens = dp['options']
        input_tokens = demonstrations + dp["input"] + " Label:"

        losses = []
        gradients = []
        total_flops =0
        for option in option_tokens:
            loss, grad, flops = self.compute_loss_and_gradient(gpt2, metaicl_model, tokenizer, input_tokens, option, device)
            # compute_loss_and_gradient(self, gpt2, model, tokenizer, input_tokens, output_tokens, device):
            losses.append(loss)
            gradients.append(grad)
            total_flops+=flops

        label_id = np.argmin(losses)
        label = dp["options"][label_id]

        if return_loss:
            return losses, gradients, label_id, total_flops
        return label_id, label, total_flops
    
    def forward_estim_op(self, gpt2, metaicl_model, input_text, return_loss=False):

        logger = logging.getLogger(__name__)
        device = torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")
        tokenizer = self.tokenizer

        losses = []
        gradients = []
        total_flops =0
        for option in option_tokens:
            loss, grad, flops = self.compute_loss_and_gradient_op(gpt2, metaicl_model, tokenizer, input_tokens, option, device)
            # compute_loss_and_gradient(self, gpt2, model, tokenizer, input_tokens, output_tokens, device):
            losses.append(loss)
            gradients.append(grad)
            total_flops+=flops

        label_id = np.argmin(losses)
        label = dp["options"][label_id]

        if return_loss:
            return losses, gradients, label_id, total_flops
        return label_id, label, total_flops

    def compute_embedding_difference(self, gpt2, metaicl_model, base_str, candidate_str, pad_to_length):
        device = torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")
        tokenizer = self.tokenizer

        input_tokens_1 = tokenizer(base_str, return_tensors="pt", padding="max_length", truncation=True, max_length=pad_to_length)["input_ids"].to(device)
        input_tokens_2 = tokenizer(candidate_str, return_tensors="pt", padding="max_length", truncation=True, max_length=pad_to_length)["input_ids"].to(device)

        with torch.no_grad():
            if "gpt" in gpt2:
                embedding_1 = metaicl_model.model.transformer.wte(input_tokens_1)
                embedding_2 = metaicl_model.model.transformer.wte(input_tokens_2)
            elif "opt" in gpt2:
                embedding_1 = metaicl_model.model.model.decoder.embed_tokens(input_tokens_1)
                embedding_2 = metaicl_model.model.model.decoder.embed_tokens(input_tokens_2)
            else:
                embedding_1 = metaicl_model.model.model.embed_tokens(input_tokens_1)
                embedding_2 = metaicl_model.model.model.embed_tokens(input_tokens_2)
            
        embedding_1 = embedding_1.to(metaicl_model.model.dtype)
        embedding_2 = embedding_2.to(metaicl_model.model.dtype)

        delta_P = embedding_2 - embedding_1.detach()

        delta_P_effective = delta_P[:, :-1, :]
        return delta_P_effective


    def compute_embedding_difference_(self, gpt2, metaicl_model, base_str, candidate_str):
        device = torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")
        #input_tokens_1 = self.tokenizer(base_str, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)["input_ids"].to(device)
        #input_tokens_2 = self.tokenizer(candidate_str, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)["input_ids"].to(device)
        input_tokens_1 = self.tokenizer(base_str, return_tensors="pt", truncation=True, max_length=self.max_length)["input_ids"].to(device)
        input_tokens_2 = self.tokenizer(candidate_str, return_tensors="pt", truncation=True, max_length=self.max_length)["input_ids"].to(device)

        tokenizer = self.tokenizer
        #max_len = max(input_tokens_1.size(1), input_tokens_2.size(1))
        #max_len = pad_to_length
        max_len = self.max_length
        input_tokens_1 = torch.nn.functional.pad(input_tokens_1, (0, max_len - input_tokens_1.size(1)), value=tokenizer.pad_token_id)
        input_tokens_2 = torch.nn.functional.pad(input_tokens_2, (0, max_len - input_tokens_2.size(1)), value=tokenizer.pad_token_id)


        with torch.no_grad():
            if "gpt" in gpt2:
                embedding_1 = metaicl_model.model.transformer.wte(input_tokens_1)
                embedding_2 = metaicl_model.model.transformer.wte(input_tokens_2)
            elif "opt" in gpt2:
                embedding_1 = metaicl_model.model.model.decoder.embed_tokens(input_tokens_1)
                embedding_2 = metaicl_model.model.model.decoder.embed_tokens(input_tokens_2)
            else:
                embedding_1 = metaicl_model.model.model.embed_tokens(input_tokens_1)
                embedding_2 = metaicl_model.model.model.embed_tokens(input_tokens_2)
        
        embedding_1 = embedding_1.to(metaicl_model.model.dtype)
        embedding_2 = embedding_2.to(metaicl_model.model.dtype)

        delta_P = embedding_2 - embedding_1.detach()


        return delta_P

    def greedy_select_subset2(self, gpt2, metaicl_model, test_data, dev_data):
        selected_indices, best_demonstrations = [], []
        best_input_str = ""
        best_accuracy = 0.0
        device = torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")

        total_flops = 0

        while len(selected_indices) < self.k:
            
            base_index = next(i for i in range(len(test_data)) if i not in selected_indices)
            base_test_example = test_data[base_index]
            # print(f"test_data[base_index] : {test_data[base_index]}")
            base_str = "Input: " + test_data[base_index]["input"] + " Label: " + test_data[base_index]["output"]+"\n"

            base_loss_lists, base_gradients = [], []
            self.logger.info(f"============ len(dev_data): {len(dev_data)} ============")
            cnt = 0
            
            for dp in tqdm(dev_data):
                losses, grads, label_id, flops = self.forward_estim(gpt2, metaicl_model, best_input_str, dp, base_test_example["task"], return_loss=True)
                total_flops+=flops
                base_loss_lists.append(losses)
                base_gradients.append(grads)
                cnt += (dp["options"][label_id]==dp["output"])

            best_candidate = base_index
            best_candidate_accuracy = cnt / len(dev_data)
            self.logger.info(f"====== base_accuracy : {best_candidate_accuracy}")
            # exit()
            base_loss_tensor = torch.tensor(base_loss_lists, device=device)
            # embedding_gradient = torch.tensor(base_gradients, device=device)
            embedding_gradient = torch.stack([torch.stack(g, dim=0) for g in base_gradients], dim=0)
            self.logger.info(f"============ Done base estimate ============")
            self.logger.info(f"====len(test_data): {len(test_data)}, self.k: {self.k}====")
            for i, candidate_test in tqdm(enumerate(test_data), total=len(test_data), leave=True, position=0):
                if (i in selected_indices) or (i == base_index): 
                    continue
                candidate_str = "Input: " + candidate_test["input"] + " Label: " + candidate_test["output"]+"\n"
                
                correct_count = 0

                for dp_idx, dp in enumerate(dev_data):
                    dev_str = "Input: " + dp["input"] + " Label:"
                    taylor_loss_list = []
                    delta_P = self.compute_embedding_difference_(gpt2, metaicl_model, best_input_str+base_str+dev_str, best_input_str+candidate_str+dev_str)  # P(S', x_q) - P(S, x_q)

                    for j in range(len(base_loss_lists[0])):
                        taylor_correction = torch.sum(embedding_gradient[dp_idx][j] * delta_P).item()
                        taylor_approx_loss = base_loss_tensor[dp_idx][j].item() + taylor_correction
                        taylor_loss_list.append(taylor_approx_loss)

                    predicted_label_id = np.argmin(taylor_loss_list)
                    predicted_label = test_data[i]["options"][predicted_label_id]

                    if predicted_label == dp["output"]:
                        correct_count += 1

                candidate_accuracy = correct_count / len(dev_data)
                #if candidate_accuracy > best_candidate_accuracy:
                print(candidate_accuracy)
                if candidate_accuracy < best_candidate_accuracy:
                    best_candidate = i
                    best_candidate_accuracy = candidate_accuracy
            
            self.logger.info("-------------one loop done--------------")
            selected_indices.append(best_candidate)
            best_input_str += "Input: " + test_data[best_candidate]["input"] + " Label: " + test_data[best_candidate]["output"]+"\n"
            best_demonstrations.append(test_data[best_candidate])
            self.logger.info(f"Selected index {best_candidate}, current best accuracy: {best_candidate_accuracy:.4f}")

        return best_demonstrations, best_candidate_accuracy, total_flops

    def greedy_select_subset3(self, gpt2, metaicl_model, test_data, dev_data):
        def get_length(example, prompt_text, options):
            return max(len(prompt_text + example["input"] + op + "\n") for op in options)
        def get_max_tokenized_length(tokenizer, test_data, prompt_text, options):
            max_len = 0
            for example in test_data:
                for op in options:
                    full_text = prompt_text + example["input"] + op + "\n"
                    input_ids = tokenizer(full_text, return_tensors="pt", truncation=False)["input_ids"]
                    max_len = max(max_len, input_ids.size(1))
            return max_len
        self.options = test_data[0]["options"]
        
        selected_indices, best_demonstrations = [], []
        device = torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")


        total_flops = 0
        prompt_text = ""

        while len(selected_indices) < self.k:
            max_token_len = get_max_tokenized_length(self.tokenizer, test_data, prompt_text, self.options)
            # Contruct input for different options
            
            #base_index = next(i for i in range(len(test_data)) if i not in selected_indices)
            base_index = max(
                (i for i in range(len(test_data)) if i not in selected_indices),
                key=lambda i: get_length(test_data[i], prompt_text, self.options)
            )
            base_example = test_data[base_index]
            # print(f"test_data[base_index] : {test_data[base_index]}")
            base_text_option_dict = {}
            loss_option_dict, gradient_option_dict = {}, {}
            for op in self.options:
                #base_text = "Input: " + base_example["input"]+ op +"\n"
                base_text = prompt_text + base_example["input"]+ op +"\n"
                base_text_option_dict[op] = base_text
                #print(base_text)
                base_token = self.tokenizer(base_text, return_tensors="pt", truncation=False)["input_ids"]
                #print(base_token)
                #print(self.tokenizer.decode(base_token[0]))
                #print(a)

                # get loss and gradient
                loss_op, embedding_grad_op = _get_embedding_loss(model=metaicl_model, tokenizer=self.tokenizer, input_texts=[base_text], pad_to_length=max_token_len)

                #print(loss_op)
                #print(embedding_grad_op)
                loss_option_dict[op] = loss_op
                gradient_option_dict[op] = embedding_grad_op

            best_candidate = base_index
            best_candidate_loss = loss_option_dict[base_example["output"]].cpu().item()
            for i, candidate_sample in tqdm(enumerate(test_data), total=len(test_data), leave=True, position=0):
                if (i in selected_indices) or (i == base_index): 
                    continue
                candidate_approx_loss_dict = {}
                for op in self.options:
                    #candidate_str = "Input: " + candidate_sample["input"] + candidate_sample["output"]+"\n"
                    candidate_text =  prompt_text + candidate_sample["input"] + op +"\n"

                    delta_P = self.compute_embedding_difference(gpt2, metaicl_model, base_str=base_text_option_dict[op], candidate_str=candidate_text, pad_to_length=max_token_len)
                    #print(gradient_option_dict[op].shape)
                    #print(delta_P.shape)
                    taylor_correction = torch.sum(gradient_option_dict[op] * delta_P).item()
                    taylor_approx_loss = loss_option_dict[op] + taylor_correction
                    candidate_approx_loss_dict[op] = taylor_approx_loss
                
                candidate_approx_loss = candidate_approx_loss_dict[candidate_sample["output"]].cpu().item()
                if candidate_approx_loss < best_candidate_loss:
                    best_candidate = i
                    best_candidate_loss = candidate_approx_loss
                #print(f"candidate_approx_loss : {candidate_approx_loss}")

                
            self.logger.info("-------------one loop done--------------")
            selected_indices.append(best_candidate)
            #best_input_str += "Input: " + test_data[best_candidate]["input"] + " Label: " + test_data[best_candidate]["output"]+"\n"
            prompt_text += test_data[best_candidate]["input"] + test_data[best_candidate]["output"]+"\n"
            best_demonstrations.append(test_data[best_candidate])
            best_candidate_accuracy = 0
            self.logger.info(f"Selected index {best_candidate}")

        return best_demonstrations, best_candidate_accuracy, total_flops
    
    def greedy_select_subset4(self, gpt2, metaicl_model, test_data, dev_data):
        def get_length(example, prompt_text, options):
            return max(len(prompt_text + example["input"] + op + "\n") for op in options)
        def get_max_tokenized_length(tokenizer, test_data, prompt_text, options):
            max_len = 0
            for example in test_data:
                for op in options:
                    full_text = prompt_text + example["input"] + op + "\n"
                    input_ids = tokenizer(full_text, return_tensors="pt", truncation=False)["input_ids"]
                    max_len = max(max_len, input_ids.size(1))
            return max_len
        self.options = test_data[0]["options"]
        
        selected_indices, best_demonstrations = [], []
        device = torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")


        total_flops = 0
        prompt_text = ""
        # ramdom select
        while len(selected_indices) < 3:
            random_index = random.randint(0, len(test_data)-1)
            selected_indices.append(random_index)
            prompt_text += test_data[random_index]["input"] + test_data[random_index]["output"]+"\n"
            best_demonstrations.append(test_data[random_index])
    
        print(len(selected_indices))

        while len(selected_indices) < self.k:
            max_token_len = get_max_tokenized_length(self.tokenizer, test_data, prompt_text, self.options)
            # Contruct input for different options
            
            #base_index = next(i for i in range(len(test_data)) if i not in selected_indices)
            base_index = max(
                (i for i in range(len(test_data)) if i not in selected_indices),
                key=lambda i: get_length(test_data[i], prompt_text, self.options)
            )
            base_example = test_data[base_index]
            # print(f"test_data[base_index] : {test_data[base_index]}")
            base_text_option_dict = {}
            loss_option_dict, gradient_option_dict = {}, {}
            for op in self.options:
                #base_text = "Input: " + base_example["input"]+ op +"\n"
                base_text = prompt_text + base_example["input"]+ op +"\n"
                base_text_option_dict[op] = base_text

                # get loss and gradient
                loss_op, embedding_grad_op = _get_embedding_loss(model=metaicl_model, tokenizer=self.tokenizer, input_texts=[base_text], pad_to_length=max_token_len)

                #print(loss_op)
                #print(embedding_grad_op)
                loss_option_dict[op] = loss_op
                gradient_option_dict[op] = embedding_grad_op

            best_candidate = base_index
            best_candidate_loss = loss_option_dict[base_example["output"]].cpu().item()
            for i, candidate_sample in tqdm(enumerate(test_data), total=len(test_data), leave=True, position=0):
                if (i in selected_indices) or (i == base_index): 
                    continue
                candidate_approx_loss_dict = {}
                for op in self.options:
                    #candidate_str = "Input: " + candidate_sample["input"] + candidate_sample["output"]+"\n"
                    candidate_text =  prompt_text + candidate_sample["input"] + op +"\n"

                    delta_P = self.compute_embedding_difference(gpt2, metaicl_model, base_str=base_text_option_dict[op], candidate_str=candidate_text, pad_to_length=max_token_len)
                    #print(gradient_option_dict[op].shape)
                    #print(delta_P.shape)
                    taylor_correction = torch.sum(gradient_option_dict[op] * delta_P).item()
                    taylor_approx_loss = loss_option_dict[op] + taylor_correction
                    candidate_approx_loss_dict[op] = taylor_approx_loss
                
                candidate_approx_loss = candidate_approx_loss_dict[candidate_sample["output"]].cpu().item()
                if candidate_approx_loss < best_candidate_loss:
                    best_candidate = i
                    best_candidate_loss = candidate_approx_loss
                #print(f"candidate_approx_loss : {candidate_approx_loss}")

                
            self.logger.info("-------------one loop done--------------")
            selected_indices.append(best_candidate)
            #best_input_str += "Input: " + test_data[best_candidate]["input"] + " Label: " + test_data[best_candidate]["output"]+"\n"
            prompt_text += test_data[best_candidate]["input"] + test_data[best_candidate]["output"]+"\n"
            best_demonstrations.append(test_data[best_candidate])
            best_candidate_accuracy = 0
            self.logger.info(f"Selected index {best_candidate}")

        return best_demonstrations, best_candidate_accuracy, total_flops
    
    # def greedy_select_subset5(self, gpt2, metaicl_model, test_data, dev_data, true_step=0):
    #     def get_length(example, prompt_text, options):
    #         return max(len(prompt_text + example["input"] + op + "\n") for op in options)
    #     def get_max_tokenized_length(tokenizer, test_data, prompt_text, options):
    #         max_len = 0
    #         for example in test_data:
    #             for op in options:
    #                 full_text = prompt_text + example["input"] + op + "\n"
    #                 input_ids = tokenizer(full_text, return_tensors="pt", truncation=False)["input_ids"]
    #                 max_len = max(max_len, input_ids.size(1))
    #         return max_len
    #     def build_text(prefix_text, base_sample, query_sample, op):
    #         return prefix_text + base_sample["input"] + base_sample["output"] + query_sample["input"] + op + "\n"
    #     self.options = test_data[0]["options"]
        
    #     selected_indices, best_demonstrations = [], []
    #     device = torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")


    #     total_flops = 0
    #     prompt_text = ""

    #     while len(selected_indices) < self.k:
    #         max_token_len = get_max_tokenized_length(self.tokenizer, test_data, prompt_text, self.options)
    #         # Contruct input for different options
            
    #         #base_index = next(i for i in range(len(test_data)) if i not in selected_indices)
    #         base_index = max(
    #             (i for i in range(len(test_data)) if i not in selected_indices),
    #             key=lambda i: get_length(test_data[i], prompt_text, self.options)
    #         )
    #         base_example = test_data[base_index]
    #         # print(f"test_data[base_index] : {test_data[base_index]}")
    #         base_text_option_dict = {}
            
    #         query_loss_op_dict_list = []
    #         approx_acc_list = [0 for _ in range(len(test_data))]
    #         approx_loss_list = [0 for _ in range(len(test_data))]
    #         for query_idx, query in enumerate(dev_data):
    #             loss_option_dict, gradient_option_dict = {}, {}
    #             for op in self.options:
    #                 base_text = build_text(prompt_text, base_example, query, op)
    #                 base_text_option_dict[op] = base_text
    #                 base_token = self.tokenizer(base_text, return_tensors="pt", truncation=False)["input_ids"]

    #                 loss_op, embedding_grad_op, flops = _get_embedding_loss(model=metaicl_model, tokenizer=self.tokenizer, input_texts=[base_text], pad_to_length=max_token_len, is_flops=self.is_flops)
    #                 total_flops+=flops

    #                 loss_option_dict[op] = loss_op
    #                 gradient_option_dict[op] = embedding_grad_op
    #             loss_ = loss_option_dict[query["output"]].cpu().item()
    #             approx_loss_list[base_index] += loss_
    #             if query['options'][np.argmin(loss_option_dict.values())] == query["output"]:
    #                 approx_acc_list[base_index] += 1

    #             best_candidate = base_index
    #             best_candidate_loss = loss_option_dict[base_example["output"]].cpu().item()
    #             best_candidate_score = np.std([loss_option_dict[op].cpu().item() for op in self.options])
    #             for i, candidate_sample in tqdm(enumerate(test_data), total=len(test_data), leave=True, position=0):
    #                 if (i in selected_indices) or (i == base_index): 
    #                     continue
    #                 candidate_approx_loss_dict = {}
    #                 for op in self.options:
    #                     candidate_text = build_text(prompt_text, candidate_sample, query, op)

    #                     delta_P = self.compute_embedding_difference(gpt2, metaicl_model, base_str=base_text_option_dict[op], candidate_str=candidate_text, pad_to_length=max_token_len)
    #                     taylor_correction = torch.sum(gradient_option_dict[op] * delta_P).item()
    #                     taylor_approx_loss = loss_option_dict[op] + taylor_correction
    #                     candidate_approx_loss_dict[op] = taylor_approx_loss

    #                 approx_loss_list[i] += candidate_approx_loss_dict[query["output"]].cpu().item()
    #                 if query['options'][np.argmin(candidate_approx_loss_dict.values())] == query["output"]:
    #                     approx_acc_list[i] += 1

    #         for si in selected_indices:
    #             approx_loss_list[si] = 1e10
    #         best_candidate = np.argmin(approx_loss_list)
    #         print(approx_loss_list)

                
    #         self.logger.info("-------------one loop done--------------")
    #         selected_indices.append(best_candidate)
    #         #best_input_str += "Input: " + test_data[best_candidate]["input"] + " Label: " + test_data[best_candidate]["output"]+"\n"
    #         prompt_text += test_data[best_candidate]["input"] + test_data[best_candidate]["output"]+"\n"
    #         best_demonstrations.append(test_data[best_candidate])
    #         best_candidate_accuracy = 0
    #         self.logger.info(f"Selected index {best_candidate}")

    #     return best_demonstrations, best_candidate_accuracy, total_flops

    def greedy_select_subset5(self, gpt2, metaicl_model, test_data, dev_data, true_step=0):
        def get_length(example, prompt_text, options):
            return max(len(prompt_text + example["input"] + op + "\n") for op in options)

        def get_max_tokenized_length(tokenizer, test_data, prompt_text, options):
            max_len = 0
            for example in test_data:
                for op in options:
                    full_text = prompt_text + example["input"] + op + "\n"
                    input_ids = tokenizer(full_text, return_tensors="pt", truncation=False)["input_ids"]
                    max_len = max(max_len, input_ids.size(1))
            return max_len

        def build_text(prefix_text, base_sample, query_sample, op):
            return prefix_text + base_sample["input"] + base_sample["output"] + query_sample["input"] + op + "\n"

        self.options = test_data[0]["options"]
        selected_indices, best_demonstrations = [], []
        device = torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")
        total_flops = 0
        prompt_text = ""

        while len(selected_indices) < self.k:
            step = len(selected_indices)
            max_token_len = get_max_tokenized_length(self.tokenizer, test_data, prompt_text, self.options)

            base_index = max(
                (i for i in range(len(test_data)) if i not in selected_indices),
                key=lambda i: get_length(test_data[i], prompt_text, self.options)
            )
            base_example = test_data[base_index]
            base_text_option_dict = {}
            approx_acc_list = [0 for _ in range(len(test_data))]
            approx_loss_list = [0 for _ in range(len(test_data))]

            for query_idx, query in enumerate(dev_data):
                loss_option_dict, gradient_option_dict = {}, {}

                for op in self.options:
                    base_text = build_text(prompt_text, base_example, query, op)
                    base_text_option_dict[op] = base_text

                    loss_op, embedding_grad_op, flops = _get_embedding_loss(
                        model=metaicl_model, tokenizer=self.tokenizer,
                        input_texts=[base_text], pad_to_length=max_token_len,
                        is_flops=self.is_flops
                    )
                    total_flops += flops
                    loss_option_dict[op] = loss_op
                    gradient_option_dict[op] = embedding_grad_op

                loss_ = loss_option_dict[query["output"]].cpu().item()
                approx_loss_list[base_index] += loss_
                if query['options'][np.argmin(loss_option_dict.values())] == query["output"]:
                    approx_acc_list[base_index] += 1

                for i, candidate_sample in enumerate(test_data):
                    if i in selected_indices:
                        continue
                    if step < true_step:
                        # --------- True inference path ---------
                        candidate_loss_option_dict = {}
                        for op in self.options:
                            candidate_text = build_text(prompt_text, candidate_sample, query,op)
                            loss_op, _, flops = _get_embedding_loss(
                                model=metaicl_model, tokenizer=self.tokenizer,
                                input_texts=[candidate_text], pad_to_length=max_token_len,
                                is_flops=self.is_flops
                            )
                            candidate_loss_option_dict[op] = loss_op
                            total_flops += flops
                        approx_loss_list[i] += candidate_loss_option_dict[query["output"]].cpu().item()
                        if query['options'][np.argmin(candidate_loss_option_dict.values())] == query["output"]:
                            approx_acc_list[i] += 1
                    else:
                        # --------- Estimate via Taylor approximation ---------
                        candidate_approx_loss_dict = {}
                        for op in self.options:
                            candidate_text = build_text(prompt_text, candidate_sample, query, op)
                            delta_P = self.compute_embedding_difference(
                                gpt2, metaicl_model, base_str=base_text_option_dict[op],
                                candidate_str=candidate_text, pad_to_length=max_token_len
                            )
                            taylor_correction = torch.sum(gradient_option_dict[op] * delta_P).item()
                            taylor_approx_loss = loss_option_dict[op] + taylor_correction
                            candidate_approx_loss_dict[op] = taylor_approx_loss

                        approx_loss_list[i] += candidate_approx_loss_dict[query["output"]].cpu().item()
                        if query['options'][np.argmin(candidate_approx_loss_dict.values())] == query["output"]:
                            approx_acc_list[i] += 1

            for si in selected_indices:
                approx_loss_list[si] = 1e10
            best_candidate = np.argmin(approx_loss_list)
            print(approx_loss_list)

            self.logger.info("-------------one loop done--------------")
            selected_indices.append(best_candidate)
            prompt_text += test_data[best_candidate]["input"] + test_data[best_candidate]["output"] + "\n"
            best_demonstrations.append(test_data[best_candidate])
            best_candidate_accuracy = 0
            self.logger.info(f"Selected index {best_candidate}")

        return best_demonstrations, best_candidate_accuracy, total_flops

    def _select_top_k_neighbors(self, test_sample_embedding, test_embeddings, test_data, k, dp_idx):
        similarities = []
        for idx, dp in enumerate(test_embeddings):
            if idx == len(test_data): break
            if idx == dp_idx:
                similarities.append(-1.0)
                continue
            similarity = 1 - cosine(test_sample_embedding, dp)
            similarities.append(similarity)
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [test_data[i] for i in top_k_indices], top_k_indices , similarities
    
    def tensorize_estimate_(self, gpt2, _test_data, _val_data, is_quant, pseudo_k=3, options=None, add_newlines=True):
        print("options: ", options)
        if options is not None:
            print("len(_test_data) : ", len(_test_data))
            print(_test_data[0])
            for i, dp in enumerate(_test_data):
                assert "options" not in dp,print(i,dp)
                _test_data[i] = {"input": dp, "options": options}
            for i, dp in enumerate(_val_data):
                assert "options" not in dp
                _val_data[i] = {"input": dp, "options": options}
        print("len(_test_data) : ",len(_test_data)," ; len(_val_data) : ", len(_val_data))

        val_data, unlabeled_data, psudo_data, test_data = [], [], [], []
        for dp in _test_data:
            if "output" not in dp: dp["output"] = dp["options"][0]
            test_data.append(dp.copy())
        for dp in _val_data:
            if "output" not in dp: dp["output"] = dp["options"][0]
            val_data.append(dp.copy())
            unlabeled_data.append(dp.copy())
        task = _test_data[0]["task"]
        with open(f"./features/{task}_test_features.json", "r") as file: test_features = json.load(file)
        with open(f"./features/{task}_val_features.json", "r") as file: val_features = json.load(file)
        
        total_flops = 0

        add_newlines = False
        checkpoint = None
        metaicl_model = MetaICLModel(logger=self.logger, out_dir= "./cache", device_num=self.device)
        print(f"-------------- gpt2: {gpt2} ------------")
        metaicl_model.load(gpt2=gpt2,is_quant=is_quant)

        print("gpt2 : ",gpt2)
        print("origin type(metaicl_model) : ",type(metaicl_model.model))
        #if "Llama" in gpt2:
        #    metaicl_model.resize(self.tokenizer)

        correct = 0     
        if pseudo_k<=10:   
            for idx,dp in tqdm(enumerate(unlabeled_data), total=len(unlabeled_data), leave=True, position=0):
                samples, top_indices, _ = self._select_top_k_neighbors(val_features[idx], test_features, test_data, k=pseudo_k,dp_idx=-1)
                demonstration=[]

                zt_output = dp["output"]

                #for dk in samples:
                #    demonstration+=self.tokenizer("Input: " + dk["input"] + " " + "Label: "+dk["output"])["input_ids"]
                #_, dp["output"], flops= self.forward(gpt2, metaicl_model, demonstration, dp, dp["task"])
                #if self.is_flops: self.logger.info(f"----- flops : {flops / 1e9:.2f} GFLOPs")
                #total_flops+=flops

                correct +=(zt_output==dp["output"])

                psudo_data.append(dp)
            self.logger.info(f"ZT_Accuracy = {float(correct/(len(unlabeled_data)))}")
        # psudo_data = val_data.copy()

        input_ids, attention_mask, token_type_ids = [], [], []
        metadata = []
        
        ground, _, flops = self.greedy_select_subset2(gpt2=gpt2, metaicl_model=metaicl_model, test_data=test_data, dev_data=psudo_data)
        demonstrations = []

        total_flops+= flops

        for i, neighbor_dp in enumerate(ground):
            demonstrations+=self.tokenizer("Input: " + neighbor_dp["input"] + " " +neighbor_dp["output"] + "\n")["input_ids"]

        cnt=0
        
        for dp in tqdm(val_data):
            _, output, flops = self.forward(gpt2, metaicl_model, demonstrations, dp, dp["task"])
            if self.is_flops: self.logger.info(f"----- flops : {flops / 1e9:.2f} GFLOPs")
            total_flops+=flops
            
            cnt += (output==dp["output"])
        self.logger.info(f"Accuracy : {cnt/len(val_data)}")
        self.logger.info(f"Total_FLOPS: {total_flops / 1e9:.2f} GFLOPs")

        for i, neighbor_dp in enumerate(ground):
            input_, output_ = self._prepro_each_datapoint(
                neighbor_dp, is_first=i == 0, for_demonstrations=True, add_newlines=add_newlines)
            demonstrations += input_ + output_

        for dp_idx, dp in enumerate(val_data):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)
                

            indices = [[i] for i in range(len(input_ids), len(input_ids) + len(inputs))]

            metadata.append({"indices": indices, "answer": answer, "options": dp["options"]})

            for inputs_, outputs_ in zip(inputs, outputs):
                if self.use_demonstrations:
                    inputs_ = demonstrations + inputs_
                encoded = prepro_sentence_pair_single(
                    inputs_, outputs_, self.max_length,  self.tokenizer, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id,
                    allow_truncation=self.use_demonstrations
                )
                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])

        self.tensorized_inputs = dict(input_ids=torch.LongTensor(input_ids),
                                      attention_mask=torch.LongTensor(attention_mask),
                                      token_type_ids=torch.LongTensor(token_type_ids))
        self.metadata = metadata

    def tensorize_estimate(self, gpt2, _test_data, _val_data, is_quant, method="forsel", pseudo_k=3, num_anchors=1, true_step=0, options=None, add_newlines=True):
        print("options: ", options)
        if options is not None:
            print("len(_test_data) : ", len(_test_data))
            print(_test_data[0])
            for i, dp in enumerate(_test_data):
                _test_data[i] = {"input": dp, "options": options}
            for i, dp in enumerate(_val_data):
                _val_data[i] = {"input": dp, "options": options}
        print("len(_test_data) : ",len(_test_data)," ; len(_val_data) : ", len(_val_data))

        val_data, unlabeled_data, psudo_data, test_data = [], [], [], []
        for dp in _test_data:
            if "output" not in dp: dp["output"] = dp["options"][0]
            test_data.append(dp.copy())
        for dp in _val_data:
            if "output" not in dp: dp["output"] = dp["options"][0]
            val_data.append(dp.copy())
            unlabeled_data.append(dp.copy())
        task = _test_data[0]["task"]
        with open(f"./features/{task}_test_features.json", "r") as file: test_features = json.load(file)
        with open(f"./features/{task}_val_features.json", "r") as file: val_features = json.load(file)
        
        total_flops = 0

        add_newlines = True
        checkpoint = None
        metaicl_model = MetaICLModel(logger=self.logger, out_dir= "./cache", device_num=self.device)
        print(f"-------------- gpt2: {gpt2} ------------")
        metaicl_model.load(gpt2=gpt2,is_quant=is_quant)

        print("gpt2 : ",gpt2)
        print("origin type(metaicl_model) : ",type(metaicl_model.model))
        if "Llama" in gpt2:
           metaicl_model.resize(self.tokenizer)

        correct = 0     

        input_ids, attention_mask, token_type_ids = [], [], []
        metadata = []
        correct = 0     
        if pseudo_k<=10:   
            for idx,dp in tqdm(enumerate(unlabeled_data), total=len(unlabeled_data), leave=True, position=0):
                # samples, top_indices, _ = self._select_top_k_neighbors(val_features[idx], test_features, test_data, k=20,dp_idx=-1)
                # demonstration=[]

                # zt_output = dp["output"]

                # for dk in samples:
                #     demonstration+=self.tokenizer("Input: " + dk["input"] + " " + "Label: "+dk["output"])["input_ids"]
                # _, dp["output"], flops= self.forward(gpt2, metaicl_model, demonstration, dp, dp["task"])
                # #if self.is_flops: self.logger.info(f"----- flops : {flops / 1e9:.2f} GFLOPs")
                # #total_flops+=flops

                # correct +=(zt_output==dp["output"])

                psudo_data.append(dp)
            self.logger.info(f"ZT_Accuracy = {float(correct/(len(unlabeled_data)))}")
        if method=="forsel":
            ground, _, flops = self.greedy_select_subset5(gpt2=gpt2, metaicl_model=metaicl_model, test_data=test_data, dev_data=psudo_data, true_step=true_step)
        elif method=='ranens':
            ground, _, flops = self.random_ensemble(gpt2=gpt2, k=self.k, metaicl_model=metaicl_model, test_data=test_data, dev_data=psudo_data, num_anchors=num_anchors)
        else:
            ground, _, flops = self.greedy_select_subset_cone
        demonstrations = []

        total_flops+= flops

        for i, neighbor_dp in enumerate(ground):
            demonstrations+=self.tokenizer(neighbor_dp["input"] + " " +neighbor_dp["output"] + "\n")["input_ids"]

        cnt=0
        
        # for dp in tqdm(val_data):
        #     _, output, flops = self.forward(gpt2, metaicl_model, demonstrations, dp, dp["task"])
        #     if self.is_flops: self.logger.info(f"----- flops : {flops / 1e9:.2f} GFLOPs")
        #     total_flops+=flops
            
        #     cnt += (output==dp["output"])
        # self.logger.info(f"Accuracy : {cnt/len(val_data)}")
        # self.logger.info(f"Total_FLOPS: {total_flops / 1e9:.2f} GFLOPs")

        demonstrations = []
        for i, neighbor_dp in enumerate(ground):
            input_, output_ = self._prepro_each_datapoint(
                neighbor_dp, is_first=i == 0, for_demonstrations=True, add_newlines=add_newlines)
            demonstrations += input_[1:] + output_[1:]

        for dp_idx, dp in enumerate(val_data):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)
                

            indices = [[i] for i in range(len(input_ids), len(input_ids) + len(inputs))]

            metadata.append({"indices": indices, "answer": answer, "options": dp["options"]})

            for inputs_, outputs_ in zip(inputs, outputs):
                if self.use_demonstrations:
                    inputs_ = demonstrations + inputs_[1:]
                encoded = prepro_sentence_pair_single(
                    inputs_, outputs_, self.max_length,  self.tokenizer, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id,
                    allow_truncation=self.use_demonstrations
                )
                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])

        self.tensorized_inputs = dict(input_ids=torch.LongTensor(input_ids),
                                      attention_mask=torch.LongTensor(attention_mask),
                                      token_type_ids=torch.LongTensor(token_type_ids))
        self.metadata = metadata

    def get_dataloader(self, batch_size, is_training):
        inputs = self.tensorized_inputs
        for k, v in inputs.items():
            if type(v)==list:
                inputs[k] = torch.LongTensor(v)
        shape = inputs["input_ids"].shape
        self.logger.info(shape)
        for v in inputs.values():
            assert v.shape==shape
        if "labels" in inputs:
            dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"], inputs["labels"])
        else:
            dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"])
        if is_training:
            sampler=RandomSampler(dataset)
        else:
            sampler=SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        return dataloader

    def evaluate(self, predictions, groundtruths, is_classification):
        assert len(predictions)==len(self.metadata)
        accs = []
        precisions = defaultdict(list)
        recalls = defaultdict(list)
        for prediction, groundtruth in zip(predictions, groundtruths):
            prediction = prediction.strip()
            groundtruth = [gt.strip() for gt in groundtruth] if type(groundtruth)==list else groundtruth.strip()
            is_correct = prediction in groundtruth if type(groundtruth)==list else prediction==groundtruth
            accs.append(is_correct)
            if is_classification:
                recalls[groundtruth].append(is_correct)
                precisions[prediction].append(is_correct)

        if not is_classification:
            return np.mean(accs)

        f1s = []
        for key in recalls:
            precision = np.mean(precisions[key]) if key in precisions else 1.0
            recall = np.mean(recalls[key])
            if precision+recall==0:
                f1s.append(0)
            else:
                f1s.append(2*precision*recall / (precision+recall))

        return np.mean(f1s)

    def _prepro_each_datapoint(self, dp, is_first=True, is_training=False, for_demonstrations=False,
                               add_newlines=True):
        dp = dp.copy()
        if add_newlines:
            no_label = np.all([option=="" for option in dp["options"]])
            no_input = dp["input"]==""
            if self.method=="direct":
                if not is_first:
                    if no_input:
                        dp["input"] = "\n" + dp["input"]
                    else:
                        dp["input"] = "\n" + dp["input"]
                if not no_label:
                    dp["output"] = "\n" + dp["output"]
                    if "options" in dp:
                        dp["options"] = ["\n" + opt for opt in dp["options"]]
            else:
                raise NotImplementedError()

        input_tokens = self.tokenizer(dp["input"])["input_ids"]

        if is_training or for_demonstrations:
            output_tokens = self.tokenizer(dp["output"])["input_ids"]
            if "task" in dp:
                if (dp["task"].startswith("inst:piqa") or dp["task"].startswith("inst:yahoo_answers_topics")) and \
                        len(input_tokens)+len(output_tokens)+2>self.max_length_per_example:
                    input_tokens = input_tokens[:self.max_length_per_example // 2]
                    output_tokens = output_tokens[:self.max_length_per_example // 2 - 2]

                elif len(input_tokens)>=self.max_length_per_example - 2 - len(output_tokens):
                    if dp["task"].startswith("inst:") and len(input_tokens)<len(output_tokens):
                        output_tokens = output_tokens[:self.max_length_per_example - 2 - len(input_tokens)]
                    else:
                        input_tokens = input_tokens[:self.max_length_per_example - 2 - len(output_tokens)]

            assert len(input_tokens)+len(output_tokens)+2<=self.max_length_per_example, \
                (dp.get("task", None), len(input_tokens), len(output_tokens), self.max_length_per_example)

            if self.method=="direct":
                return input_tokens, output_tokens
            elif self.method=="channel":
                return output_tokens, input_tokens
            else:
                raise NotImplementedError()

        else:
            assert len(dp["options"])>=2, dp
            assert dp["output"] in dp["options"]
            option_tokens = [self.tokenizer(option)["input_ids"] for option in dp["options"]]
            option_length = np.max([len(option) for option in option_tokens])

            if len(input_tokens)>=self.max_length_per_example - 2 - option_length:
                input_tokens = input_tokens[:self.max_length_per_example - 2 - option_length]

            input_tokens = [input_tokens for _ in option_tokens]
            output_tokens = option_tokens
            option_tokens = [dp["options"].index(dp["output"])]

            if self.method=="direct":
                return input_tokens, output_tokens, option_tokens
            elif self.method=="channel":
                return output_tokens, input_tokens, option_tokens
            else:
                raise NotImplementedError()

    def tensorize_topk(self, _test_data, _val_data, options=None, add_newlines=True):
        if options is not None:
            for i, dp in enumerate(_test_data):
                _test_data[i] = {"input": dp, "options": options}
            for i, dp in enumerate(_val_data):
                _val_data[i] = {"input": dp, "options": options}

        val_data, test_data =  [], []

        for dp in _test_data:
            if "output" not in dp:
                dp["output"] = dp["options"][0]  # randomly choose one (we don't need it anyways)
            test_data.append(dp.copy())
        for dp in _val_data:
            if "output" not in dp:
                dp["output"] = dp["options"][0]  # randomly choose one (we don't need it anyways)
            val_data.append(dp.copy())
        
        task = _test_data[0]["task"]
        test_features_path = f"./features/{task}_test_features.json"
        with open(test_features_path, "r") as file:
            test_features = json.load(file)
        val_features_path = f"./features/{task}_val_features.json"
        with open(val_features_path, "r") as file:
            val_features = json.load(file)

        input_ids, attention_mask, token_type_ids = [], [], []
        metadata = []

        for dp_idx, dp in enumerate(val_data):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)
            if self.use_demonstrations:
                dp_feature = val_features[dp_idx]            

                top_k_neighbors, _, __ = self._select_top_k_neighbors(
                    dp_feature, test_features, test_data, self.k, dp_idx
                )

                demonstrations = []
                for i, neighbor_dp in enumerate(reversed(top_k_neighbors)):
                    input_, output_ = self._prepro_each_datapoint(
                        neighbor_dp, is_first=i == 0, for_demonstrations=True, add_newlines=add_newlines)
                    demonstrations += input_[1:] + output_[1:]
                #print(demonstrations)
            #print(a)
            indices = [[i] for i in range(len(input_ids), len(input_ids) + len(inputs))]
            # print("indices : ",indices)
            # print("inputs : ",inputs)
            # print("answer : ",answer)
            # print("demonstrations : ",demonstrations)

            metadata.append({"indices": indices, "answer": answer, "options": dp["options"]})

            for inputs_, outputs_ in zip(inputs, outputs):

                if self.use_demonstrations:
                    inputs_ = demonstrations + inputs_[1:]
                encoded = prepro_sentence_pair_single(
                    inputs_, outputs_, self.max_length, self.tokenizer,self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, 
                    allow_truncation=self.use_demonstrations
                )
                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])


        self.tensorized_inputs = dict(input_ids=torch.LongTensor(input_ids),
                                      attention_mask=torch.LongTensor(attention_mask),
                                      token_type_ids=torch.LongTensor(token_type_ids))
        self.metadata = metadata


    def _forward_selection(self, embeddings, top_k_indices, m, candidate_labels, test_data, similarities, seed, temperature=0.1):
        assert m <= len(top_k_indices), "Error: m must less than k"

        for idx in range(len(embeddings)):
            embeddings[idx] = torch.tensor(embeddings[idx], dtype=torch.float32)
            embeddings[idx] = embeddings[idx] / torch.norm(embeddings[idx])

        top_indice = np.argsort(similarities)[-1:][::-1]

        selected_indices = set(top_indice)
        top_k_indices = [item for item in top_k_indices if item != top_indice]
        remaining_indices = set(top_k_indices)

        while len(selected_indices) < m:
            best_candidate = None
            best_loss = float('inf')

            for candidate in remaining_indices:
                temp_selected = list(selected_indices.union({candidate}))

                simloss = sum(similarities[idx] for idx in temp_selected) / len(temp_selected)

                con_loss = 0.0
                for idx1 in temp_selected:
                    cnt_pos = 0
                    pos_loss = 0.0
                    # print("len(embeddings) : ", len(embeddings),"idx1 : ", idx1, "len(candidate_labels) : ",len(candidate_labels))
                    idx1_embedding, idx1_label = embeddings[idx1], candidate_labels[idx1]
                    logits = []
                    for idx2 in temp_selected:
                        if idx1 == idx2:
                            continue
                        idx2_embedding, idx2_label = embeddings[idx2], candidate_labels[idx2]
                        similarity = torch.matmul(idx1_embedding, idx2_embedding) / temperature
                        logits.append(similarity)
                        if idx1_label == idx2_label:
                            pos_loss += torch.exp(similarity)
                            cnt_pos += 1

                    logits = torch.tensor(logits)
                    logits_max = torch.max(logits)
                    logits = logits - logits_max.detach()
                    exp_logits = torch.exp(logits)
                    exp_logits_sum = exp_logits.sum()

                    if cnt_pos > 0:
                        pos_prob = pos_loss / exp_logits_sum
                        idx_loss = pos_prob / cnt_pos
                    else:
                        idx_loss = 1e-6
                    idx_loss = torch.tensor(idx_loss, dtype=torch.float32)
                    con_loss += -1.0 * torch.log(idx_loss)

                lam = 0.05
                total_loss = -simloss + lam * con_loss

                if total_loss < best_loss:
                    best_loss = total_loss
                    best_candidate = candidate

            selected_indices.add(best_candidate)
            remaining_indices.remove(best_candidate)

        real_id = [idx for idx in selected_indices]
        return [test_data[idx] for idx in real_id], real_id

    def _select_random_k_neighbors(self, test_sample_embedding, test_embeddings, test_data, k, dp_idx):
        length = len(test_data)
        candidates = [i for i in range(length) if i!= dp_idx]
        random_indices = random.sample(candidates, k)

        return [test_data[i] for i in random_indices]
    
    def tensorize_randomk(self, _test_data, _val_data, options=None, add_newlines=True):
        if options is not None:
            for i, dp in enumerate(_test_data):
                _test_data[i] = {"input": dp, "options": options}
            for i, dp in enumerate(_val_data):
                _val_data[i] = {"input": dp, "options": options}

        val_data, test_data =  [], []

        for dp in _test_data:
            if "output" not in dp:
                dp["output"] = dp["options"][0]  # randomly choose one (we don't need it anyways)
            test_data.append(dp.copy())
        for dp in _val_data:
            if "output" not in dp:
                dp["output"] = dp["options"][0]  # randomly choose one (we don't need it anyways)
            val_data.append(dp.copy())
        
        task = _test_data[0]["task"]
        test_features_path = f"./features/{task}_test_features.json"
        with open(test_features_path, "r") as file:
            test_features = json.load(file)
        val_features_path = f"./features/{task}_val_features.json"
        with open(val_features_path, "r") as file:
            val_features = json.load(file)

        input_ids, attention_mask, token_type_ids = [], [], []
        metadata = []

        for dp_idx, dp in enumerate(val_data):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)

            demonstrations = []
            if self.use_demonstrations and self.k>0:
                dp_feature = val_features[dp_idx]            

                random_k_neighbors = self._select_random_k_neighbors(
                    dp_feature, test_features, test_data, self.k-self.k//4, dp_idx
                )

                top_k_neighbors, _, __ = self._select_top_k_neighbors(
                    dp_feature, test_features, test_data, self.k//4, dp_idx
                )

                demonstrations = []
                for i, neighbor_dp in enumerate(random_k_neighbors):
                    input_, output_ = self._prepro_each_datapoint(
                        neighbor_dp, is_first=i == 0, for_demonstrations=True, add_newlines=add_newlines)
                    demonstrations += input_ + output_
                for i, neighbor_dp in enumerate(top_k_neighbors):
                    input_, output_ = self._prepro_each_datapoint(
                        neighbor_dp, is_first=i == 0, for_demonstrations=True, add_newlines=add_newlines)
                    demonstrations += input_ + output_

            indices = [[i] for i in range(len(input_ids), len(input_ids) + len(inputs))]

            metadata.append({"indices": indices, "answer": answer, "options": dp["options"]})

            for inputs_, outputs_ in zip(inputs, outputs):
                if self.use_demonstrations:
                    inputs_ = demonstrations + inputs_
                encoded = prepro_sentence_pair_single(
                    inputs_, outputs_, self.max_length, self.tokenizer, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id,
                    allow_truncation=self.use_demonstrations
                )
                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])

        self.tensorized_inputs = dict(input_ids=torch.LongTensor(input_ids),
                                      attention_mask=torch.LongTensor(attention_mask),
                                      token_type_ids=torch.LongTensor(token_type_ids))
        self.metadata = metadata

    def _random_datasource(self, task, datapath, m):
        with open(datapath, "r") as file:
            data = [json.loads(line) for line in file]
        
        candidates = [item for item in data if item['task']!=task]
        output = random.sample(candidates, m)
        return output

    def tensorize_bm25(self, _test_data, _val_data, options=None, add_newlines=True):
        if options is not None:
            for i, dp in enumerate(_test_data):
                _test_data[i] = {"input": dp, "options": options}
            for i, dp in enumerate(_val_data):
                _val_data[i] = {"input": dp, "options": options}

        val_data, test_data = [], []
        for dp in _test_data:
            if "output" not in dp:
                dp["output"] = dp["options"][0]
            test_data.append(dp.copy())
        for dp in _val_data:
            if "output" not in dp:
                dp["output"] = dp["options"][0]
            val_data.append(dp.copy())

        task = _test_data[0]["task"]
        input_ids, attention_mask, token_type_ids = [], [], []
        metadata = []

        test_inputs = [dp["input"].split() for dp in test_data]
        bm25 = BM25Okapi(test_inputs)

        instructions = f"Here are {len(test_data[0]['options'])} options: "
        for option in test_data[0]["options"]:
            instructions += option + ", "
        instructions += f"You should choose one of them to answer at the end. \nHere are {self.k} samples for your reference. \n"
        init_tokens = self.tokenizer(instructions)["input_ids"][1:]

        for dp_idx, dp in enumerate(val_data):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)

            if self.use_demonstrations:
                query_terms = dp["input"].split()
                scores = bm25.get_scores(query_terms)
                scores[dp_idx] = -1e9

                topk_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.k]
                top_k_neighbors = [test_data[i] for i in topk_indices]

                demonstrations = init_tokens
                for i, neighbor_dp in enumerate(top_k_neighbors):
                    input_, output_ = self._prepro_each_datapoint(
                        neighbor_dp, is_first=i == 0, for_demonstrations=True, add_newlines=add_newlines)
                    demonstrations += input_[1:] + output_[1:]

            indices = [[i] for i in range(len(input_ids), len(input_ids) + len(inputs))]

            metadata.append({"indices": indices, "answer": answer, "options": dp["options"]})
            for inputs_, outputs_ in zip(inputs, outputs):
                if self.use_demonstrations:
                    inputs_ = demonstrations + inputs_[1:]
                encoded = prepro_sentence_pair_single(
                    inputs_, [outputs_[2]], self.max_length, self.tokenizer,
                    self.tokenizer.bos_token_id, self.tokenizer.eos_token_id,
                    allow_truncation=self.use_demonstrations)
                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])

        self.tensorized_inputs = dict(
            input_ids=torch.LongTensor(input_ids),
            attention_mask=torch.LongTensor(attention_mask),
            token_type_ids=torch.LongTensor(token_type_ids))
        self.metadata = metadata
    
    def score_by_subset(self, gpt2, metaicl_model, test_data, dev_data, subset_indices):
        print(f"Subset indices: {subset_indices}")
        integer_indices = subset_indices.nonzero(as_tuple=True)[0]
        demonstrations = [test_data[i] for i in integer_indices]
        acc = self.evaluate_accuracy(gpt2, metaicl_model, demonstrations, dev_data, test_data[0]["task"])
        
        return acc, demonstrations

    def bayesian_optimization_for_subsets(self, gpt2, metaicl_model, test_data, val_data, options=None, add_newlines=True, n_examples=20, n_init=10, n_eval=20):
        """
        Uses Bayesian Optimization to find the best subset of examples.
        """
        # Define the search space: a discrete space of {0, 1}^m
        # where m is the total number of available examples.
        search_space_bounds = torch.stack([
            torch.zeros(n_examples, dtype=torch.float64), 
            torch.ones(n_examples, dtype=torch.float64)
        ])

        # train_X stores the evaluated subsets (binary vectors)
        train_X = torch.empty(n_init, n_examples, dtype=torch.float64)
        # perf_Y stores the raw performance g(e) from the black-box function
        perf_Y = torch.empty(n_init, 1, dtype=torch.float64)
        
        # Store all evaluated points to find the best one at the end
        evaluated_subsets = {}

        for i in range(n_init):
            # Generate a random non-empty subset
            random_subset = torch.randint(0, 2, (n_examples,), dtype=torch.float64)
            if random_subset.sum() == 0:
                random_subset[torch.randint(0, n_examples, (1,))] = 1.0

            train_X[i] = random_subset
            accuracy, _ = self.score_by_subset(gpt2, metaicl_model, test_data, val_data, random_subset)
            perf_Y[i, 0] = accuracy
            evaluated_subsets[tuple(random_subset.tolist())] = accuracy

        # ----- BO Iteration Loop (Algorithm 2, Steps 5-9) -----
        for t in range(n_init, n_eval):
            print(f"\n--- BO Iteration {t+1}/{n_eval} ---")
            
            # ----- Objective Scalarization (Algorithm 2, Step 6 & 7) -----
            # This is the core idea from the paper: use random scalarization to handle
            # the bi-objective problem (performance vs. sparsity).
            # The scalarized objective is h_t(e) using Tchebyshev scalarization.
            beta_t = torch.rand(1).item() * 0.75 + 0.25  # beta_t is in [0.25, 1] as per the paper
            g_max = perf_Y.max()
            
            cardinality_term = (1 - beta_t) * train_X.sum(dim=-1)
            perf_term = beta_t * (perf_Y.squeeze(-1) - g_max)
            
            # Combine into a single objective to be modeled by the GP
            train_Y_scalarized = torch.max(perf_term, -cardinality_term).unsqueeze(-1)
            
            # ----- Fit GPR Surrogate Model -----
            gp = SingleTaskGP(train_X, train_Y_scalarized)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            # CHANGED: Use the new MLL fitting function
            fit_gpytorch_mll(mll)
            
            # ----- Define and Optimize Acquisition Function -----
            # The paper uses Expected Improvement (EI).
            # 'best_f' is the best *scalarized* objective value found so far.
            ei_acqf = LogExpectedImprovement(model=gp, best_f=train_Y_scalarized.max())
            
            # Optimize the acquisition function over the discrete search space
            # to get the next candidate point to evaluate.
            candidate, _ = optimize_acqf(
                acq_function=ei_acqf,
                bounds=search_space_bounds,
                q=1,               # Recommend one candidate at a time
                num_restarts=10,   # Internal optimization setting for BoTorch
                raw_samples=512,   # Internal optimization setting for BoTorch
                options={"binary": True} # Critical for discrete/binary spaces
            )
            
            # ----- Evaluate the new candidate and update data -----
            new_x = candidate.squeeze(0).round() # Ensure it's a binary vector
            accuracy, _ = self.score_by_subset(gpt2, metaicl_model, test_data, val_data, new_x)
            
            # Append new data to the training set for the next iteration
            train_X = torch.cat([train_X, new_x.unsqueeze(0)])
            perf_Y = torch.cat([perf_Y, torch.tensor([[accuracy]], dtype=torch.float64)])
            evaluated_subsets[tuple(new_x.tolist())] = accuracy

        # ----- Return Best Found Solution (Algorithm 2, Step 10) -----
        # At the end, search through all *actually evaluated* points and return
        # the one with the highest raw performance g(e), not the scalarized one.
        print("\nOptimization finished. Finding the best subset...")
        best_subset_tuple = max(evaluated_subsets, key=evaluated_subsets.get)
        best_accuracy = evaluated_subsets[best_subset_tuple]
        best_subset_tensor = torch.tensor(best_subset_tuple, dtype=torch.float64)
        
        print(f"Found best subset (size: {int(best_subset_tensor.sum())}) with accuracy: {best_accuracy:.4f}")
        
        return best_subset_tensor

    def tensorize_bridge(self, gpt2, _test_data, _val_data, _train_data, is_quant, method="forsel", pseudo_k=3, num_anchors=1, true_step=0, options=None, add_newlines=True, sub_sample=False, use_proj=False, proj_dim=512):
        print(gpt2)
        print("options: ", options)
        if options is not None:
            print("len(_test_data) : ", len(_test_data))
            print(_test_data[0])
            for i, dp in enumerate(_test_data):
                _test_data[i] = {"input": dp, "options": options}
            for i, dp in enumerate(_val_data):
                _val_data[i] = {"input": dp, "options": options}
            for i, dp in enumerate(_train_data):
                _train_data[i] = {"input": dp, "options": options}
        print("len(_test_data) : ",len(_test_data)," ; len(_val_data) : ", len(_val_data), " ; len(_train_data) : ", len(_train_data))

        val_data, unlabeled_data, psudo_data, test_data = [], [], [], []
        train_data = []

        for dp in _test_data:
            if "output" not in dp: dp["output"] = dp["options"][0]
            test_data.append(dp.copy())
        for dp in _val_data:
            if "output" not in dp: dp["output"] = dp["options"][0]
            val_data.append(dp.copy())
            unlabeled_data.append(dp.copy())
        for dp in _train_data:
            if "output" not in dp: dp["output"] = dp["options"][0]
            train_data.append(dp.copy())
        task = _test_data[0]["task"]
        with open(f"./features/{task}_test_features.json", "r") as file: test_features = json.load(file)
        with open(f"./features/{task}_val_features.json", "r") as file: val_features = json.load(file)

        
        total_flops = 0

        add_newlines = True
        checkpoint = None
        metaicl_model = MetaICLModel(logger=self.logger, out_dir= "./cache", device_num=self.device)
        print(f"-------------- gpt2: {gpt2} ------------")
        metaicl_model.load(gpt2=gpt2,is_quant=is_quant)

        print("gpt2 : ",gpt2)
        print("origin type(metaicl_model) : ",type(metaicl_model.model))
        if "Llama" in gpt2:
           metaicl_model.resize(self.tokenizer)
        embedding_dim = metaicl_model.model.model.embed_tokens.weight.shape[1]
        if use_proj:
            self.proj_dim = proj_dim
            self.proj = torch.randn(embedding_dim, self.proj_dim).to(self.device)

        input_ids, attention_mask, token_type_ids = [], [], []
        metadata = []

        bridge_subset = self.bayesian_optimization_for_subsets(
            gpt2=gpt2, metaicl_model=metaicl_model,
            test_data=test_data, val_data=train_data, options=options, add_newlines=add_newlines,
            n_examples=20, n_init=16, n_eval=32
        )
        print(f"Selected bridge subset: {bridge_subset}")
        bridge_candidates = [test_data[i] for i in bridge_subset.nonzero(as_tuple=True)[0]]
        
        demonstrations = []
        for i, neighbor_dp in enumerate(bridge_candidates):
            input_, output_ = self._prepro_each_datapoint(
                neighbor_dp, is_first=i == 0, for_demonstrations=True, add_newlines=add_newlines)
            demonstrations += input_ + output_

        for dp_idx, dp in enumerate(val_data):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)
                

            indices = [[i] for i in range(len(input_ids), len(input_ids) + len(inputs))]

            metadata.append({"indices": indices, "answer": answer, "options": dp["options"]})

            for inputs_, outputs_ in zip(inputs, outputs):
                if self.use_demonstrations:
                    inputs_ = demonstrations + inputs_[1:]
                encoded = prepro_sentence_pair_single(
                    inputs_, outputs_, self.max_length,  self.tokenizer, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id,
                    allow_truncation=self.use_demonstrations
                )
                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])

        self.tensorized_inputs = dict(input_ids=torch.LongTensor(input_ids),
                                      attention_mask=torch.LongTensor(attention_mask),
                                      token_type_ids=torch.LongTensor(token_type_ids))
        self.metadata = metadata

    def tensorize(self, _train_data, _test_data, options=None,
                  add_newlines=True):

        if options is not None:
            for i, dp in enumerate(_test_data):
                assert "options" not in dp
                assert type(dp)==str
                _test_data[i] = {"input": dp, "options": options}

        train_data, test_data = [], []

        for dp in _test_data:
            assert type(dp)==dict, ("Each example should be a dictionary", dp)
            assert "input" in dp and "options" in dp and type(dp["options"])==list, \
                ("Test example should contain input and options in a list format", dp)
            if "output" not in dp:
                dp["output"] = dp["options"][0] # randomly choose one (we don't need it anyways)
            test_data.append(dp.copy())

        # each datapoint: passage, question, options, output
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id

        input_ids, attention_mask, token_type_ids = [], [], []
        metadata = []

        if self.use_demonstrations:
            assert len(train_data)==self.k
            demonstrations = []
            for i, dp in enumerate(train_data):
                input_, output_ = self._prepro_each_datapoint(
                    dp, is_first=i==0, for_demonstrations=True,
                    add_newlines=add_newlines)
                demonstrations += input_ + output_

        for dp_idx, dp in enumerate(test_data):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)

            indices = [[i] for i in range(len(input_ids), len(input_ids)+len(inputs))]

            metadata.append({"indices": indices, "answer": answer, "options": dp["options"]})

            print("inputs: ",inputs)
            print("outputs: ",outputs)

            for inputs_, outputs_ in zip(inputs, outputs):
                print("inputs_ : ",inputs_)
                print("outputs_ : ",outputs_)
                
                if self.use_demonstrations:
                    inputs_ = demonstrations + inputs_

                encoded = prepro_sentence_pair_single(
                    inputs_, outputs_, self.max_length, self.tokenizer, bos_token_id, eos_token_id,
                    allow_truncation=self.use_demonstrations)

                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])

        self.tensorized_inputs = dict(input_ids=torch.LongTensor(input_ids),
                                      attention_mask=torch.LongTensor(attention_mask),
                                      token_type_ids=torch.LongTensor(token_type_ids))
        self.metadata = metadata

    def print_tensorized_example(self, return_string=False):
        assert self.tensorized_inputs is not None

        idx = 0
        text = "Checking the first example..."
        input_ids = self.tensorized_inputs["input_ids"][idx]
        token_type_ids = self.tensorized_inputs["token_type_ids"][idx]
        if type(input_ids)!=list:
            input_ids = input_ids.numpy().tolist()
        if type(token_type_ids)!=list:
            token_type_ids = token_type_ids.numpy().tolist()

        text += "\nInput:\n"
        text += self.tokenizer.decode(input_ids[:token_type_ids.index(1)])
        text += "\nOutput:\n"
        text += self.tokenizer.decode([_id for _id, _type_id in zip(input_ids, token_type_ids) if _type_id==1])

        if return_string:
            return text

        if self.local_rank<=0:
            self.logger.info(text)



    # def tensorize_unlabeled(self, gpt2, _test_data, _val_data, m, options=None, add_newlines=True):
    #     val_data, test_data, unlab_data =  [], [], []
    #     cut_point = len(_test_data)-len(_test_data)//4
    #     for idx, dp in enumerate(_test_data):
    #         if "output" not in dp: dp["output"] = dp["options"][0]
    #         if idx < cut_point: test_data.append(dp.copy())
    #         else: unlab_data.append(dp.copy())
    #     for dp in _val_data:
    #         if "output" not in dp: dp["output"] = dp["options"][0]
    #         val_data.append(dp.copy())
    #     task = _test_data[0]["task"]
    #     test_features_path = f"./features/{task}_test_features.json"
    #     with open(test_features_path, "r") as file:
    #         test_features = json.load(file)
    #     val_features_path = f"./features/{task}_val_features.json"
    #     with open(val_features_path, "r") as file:
    #         val_features = json.load(file)

    #     input_ids, attention_mask, token_type_ids = [], [], []
    #     metadata = []
    #     if self.use_demonstrations:
    #         test_texts = [dp["input"] + " " + dp["output"] for dp in test_data]
    #         test_labels = [dp["output"] for dp in test_data]

    #     all_indices = [i for i in range(len(test_data))]
    #     # print(all_indices)
    #     similarities = [0 for i in range(len(test_data))]
    #     forsel, _ = self._forward_selection(
    #         embeddings=test_features,
    #         top_k_indices=all_indices,
    #         m=m, 
    #         candidate_labels=test_labels, 
    #         test_data=test_data,
    #         similarities = similarities,
    #         seed=42
    #     )
    #     demonstration,psu_data = [], []
    #     for dp in forsel:
    #         demonstration+=self.tokenizer("Input: " + dp["input"] + " " + "Label: "+dp["output"])["input_ids"]
    #     for idx, dp in enumerate(unlab_data):
    #         used = dp["output"]
    #         _, dp["output"]= self.forward(gpt2, demonstration,dp,dp["task"])
    #         self.logger.info(used+" ;;; "+dp["output"])
    #         psu_data.append(dp)

    #     for dp in psu_data:
    #         test_data.append(dp)
    #     test_labels = [dp["output"] for dp in test_data]
    #     # print("len(test_data) : ",len(test_data))

    #     for dp_idx, dp in enumerate(val_data):
    #         inputs, outputs, answer = self._prepro_each_datapoint(
    #             dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)

    #         if self.use_demonstrations:
    #             test_text = dp["input"]
    #             dp_feature = val_features[dp_idx]            

    #             top_k_neighbors, top_k_indices, similarities = self._select_top_k_neighbors(
    #                 dp_feature, test_features, test_data, self.k, len(test_data)+1
    #             )

    #             forsel, _ = self._forward_selection(
    #                 embeddings=test_features,
    #                 top_k_indices=top_k_indices,
    #                 m=m, 
    #                 candidate_labels=test_labels, 
    #                 test_data=test_data,
    #                 similarities = similarities,
    #                 seed=42
    #             )

    #             demonstrations = []
    #             for i, neighbor_dp in enumerate(forsel):
    #                 input_, output_ = self._prepro_each_datapoint(
    #                     neighbor_dp, is_first=i == 0, for_demonstrations=True, add_newlines=add_newlines)
    #                 demonstrations += input_ + output_

    #         indices = [[i] for i in range(len(input_ids), len(input_ids) + len(inputs))]

    #         metadata.append({"indices": indices, "answer": answer, "options": dp["options"]})

    #         for inputs_, outputs_ in zip(inputs, outputs):
    #             if self.use_demonstrations:
    #                 inputs_ = demonstrations + inputs_
    #             encoded = prepro_sentence_pair_single(
    #                 inputs_, outputs_, self.max_length, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id,
    #                 allow_truncation=self.use_demonstrations
    #             )
    #             input_ids.append(encoded[0])
    #             attention_mask.append(encoded[1])
    #             token_type_ids.append(encoded[2])

    #     self.tensorized_inputs = dict(input_ids=torch.LongTensor(input_ids),
    #                                   attention_mask=torch.LongTensor(attention_mask),
    #                                   token_type_ids=torch.LongTensor(token_type_ids))
    #     self.metadata = metadata



    # def tensorize_ground(self, gpt2, _test_data, _val_data, options=None, add_newlines=True):
    #     print("options: ", options)
    #     if options is not None:
    #         print("len(_test_data) : ", len(_test_data))
    #         print(_test_data[0])
    #         for i, dp in enumerate(_test_data):
    #             assert "options" not in dp,print(i,dp)
    #             _test_data[i] = {"input": dp, "options": options}
    #         for i, dp in enumerate(_val_data):
    #             assert "options" not in dp
    #             _val_data[i] = {"input": dp, "options": options}
    #     print("len(_test_data) : ",len(_test_data))
    #     print("len(_val_data) : ", len(_val_data))

    #     val_data, dev_data, test_data = [], [], []
    #     for dp in _test_data:
    #         if "output" not in dp: dp["output"] = dp["options"][0]
    #         test_data.append(dp.copy())
    #     for idx, dp in enumerate(_val_data):
    #         if "output" not in dp: dp["output"] = dp["options"][0]
    #         if idx<= len(_val_data)//2: val_data.append(dp.copy())
    #         else: dev_data.append(dp.copy())
    #     task = _test_data[0]["task"]
    #     with open(f"./features/{task}_test_features.json", "r") as file: test_features = json.load(file)
    #     with open(f"./features/{task}_val_features.json", "r") as file: val_features = json.load(file)

    #     input_ids, attention_mask, token_type_ids = [], [], []
    #     metadata = []

    #     ground, _ = self.greedy_select_subset(gpt2=gpt2,test_data=test_data, dev_data=dev_data, subset_size=self.k)

    #     for dp_idx, dp in enumerate(val_data):
    #         inputs, outputs, answer = self._prepro_each_datapoint(
    #             dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)

    #         if self.use_demonstrations:
    #             test_text = dp["input"]
    #             dp_feature = val_features[dp_idx]

                
    #             # def greedy_select_subset(self, test_data, dp_data, subset_size=10):
    #             demonstrations = []
    #             for i, neighbor_dp in enumerate(ground):
    #                 input_, output_ = self._prepro_each_datapoint(
    #                     neighbor_dp, is_first=i == 0, for_demonstrations=True, add_newlines=add_newlines)
    #                 demonstrations += input_ + output_

    #         indices = [[i] for i in range(len(input_ids), len(input_ids) + len(inputs))]

    #         metadata.append({"indices": indices, "answer": answer, "options": dp["options"]})

    #         for inputs_, outputs_ in zip(inputs, outputs):
    #             if self.use_demonstrations:
    #                 inputs_ = demonstrations + inputs_
    #             encoded = prepro_sentence_pair_single(
    #                 inputs_, outputs_, self.max_length, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id,
    #                 allow_truncation=self.use_demonstrations
    #             )
    #             input_ids.append(encoded[0])
    #             attention_mask.append(encoded[1])
    #             token_type_ids.append(encoded[2])

    #     self.tensorized_inputs = dict(input_ids=torch.LongTensor(input_ids),
    #                                   attention_mask=torch.LongTensor(attention_mask),
    #                                   token_type_ids=torch.LongTensor(token_type_ids))
    #     self.metadata = metadata

def prepro_sentence_pair_single_(ids1, ids2, max_length, tokenizer,
                                bos_token_id, eos_token_id,
                                allow_truncation=False):

    if allow_truncation and len(ids1)+len(ids2) > max_length:
        ids1 = ids1[len(ids1)+len(ids2)-max_length:] # len = max_length-len(ids2)
        assert len(ids1)+len(ids2)==max_length

    n_mask = max_length-len(ids1)-len(ids2)
    assert n_mask>=0, (max_length, len(ids1), len(ids2))
    input_ids = ids1+ids2 + [eos_token_id for _ in range(n_mask)]
    #print("input_ids : ",len(input_ids))
    attention_mask = [1 for _ in ids1+ids2] + [eos_token_id for _ in range(n_mask)]
    token_type_ids = [0 for _ in ids1] + [1 for _ in ids2] + [eos_token_id for _ in range(n_mask)]

    return input_ids, attention_mask, token_type_ids

def prepro_sentence_pair_single(ids1, ids2, max_length,
                                tokenizer, bos_token_id, eos_token_id,
                                allow_truncation=False):
    # Remove special tokens
    #print(tokenizer.all_special_ids)
    special_ids = set(tokenizer.all_special_ids)
    #special_ids.extend([128000, 128001])
    ids1 = [i for i in ids1 if i not in special_ids]
    ids2 = [i for i in ids2 if i not in special_ids]

    # Add bos and eos tokens later, so leave space for them
    total_len = len(ids1) + len(ids2) + 2  # +2 for bos and eos

    if allow_truncation and total_len > max_length:
        # Truncate from the beginning of ids1
        overflow = total_len - max_length
        ids1 = ids1[overflow:]
        total_len = len(ids1) + len(ids2) + 2
        assert total_len == max_length

    # Add bos at start, eos at end
    input_ids = [bos_token_id] + ids1 + ids2 + [eos_token_id]

    # Padding if needed
    n_pad = max_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * n_pad

    # Attention mask: 1 for tokens, 0 for padding (if padding is eos_token)
    attention_mask = [1] * (len(input_ids) - n_pad) + [0] * n_pad

    # Token type ids: 0 for ids1, 1 for ids2, 0 for bos and eos (you can adjust this)
    token_type_ids = [0] + [0] * len(ids1) + [1] * len(ids2) + [0] + [0] * n_pad

    return input_ids, attention_mask, token_type_ids


def prepro_sentence_pair(train_inputs, test_inputs, max_length, tokenizer, 
                         bos_token_id, eos_token_id,
                         allow_truncation=False):
    input_ids, attention_mask, token_type_ids = [], [], []
    for test_input in test_inputs:
        for train_input in train_inputs:
            _input_ids, _attention_mask, _token_type_ids = \
                prepro_sentence_pair_single(train_input, test_input, max_length, tokenizer, 
                                            bos_token_id, eos_token_id,
                                            allow_truncation=allow_truncation)
            input_ids.append(_input_ids)
            attention_mask.append(_attention_mask)
            token_type_ids.append(_token_type_ids)

    return {"input_ids": torch.LongTensor(input_ids),
            "attention_mask": torch.LongTensor(attention_mask),
            "token_type_ids": torch.LongTensor(token_type_ids)}





    # def random_ensemble(self, gpt2, metaicl_model, test_data, dev_data, num_combinations=100, k=8, seed=42):
    #     random.seed(seed)
    #     device = torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")

    #     all_indices = list(range(len(test_data)))

    #     def sample_unique_combinations(all_indices, k, num_combinations, seed=42):
    #         random.seed(seed)
    #         seen = set()
    #         combinations = []
    #         max_trials = num_combinations * 10 
    #         trials = 0
    #         while len(combinations) < num_combinations and trials < max_trials:
    #             comb = tuple(sorted(random.sample(all_indices, k)))
    #             if comb not in seen:
    #                 seen.add(comb)
    #                 combinations.append(comb)
    #             trials += 1

    #         if len(combinations) < num_combinations:
    #             print(f"Warning: only sampled {len(combinations)} unique combinations out of requested {num_combinations}")
            
    #         return combinations
        
    #     sampled_combinations = sample_unique_combinations(all_indices, self.k, num_combinations, seed=seed)

    #     anchor_comb = random.choice(sampled_combinations)
    #     anchor_prompt = "".join([
    #         f"Input: {test_data[idx]['input']} Label: {test_data[idx]['output']}\n" for idx in anchor_comb
    #     ])
    #     base_losses, base_gradients, _, flops = zip(*[
    #         self.forward_estim(gpt2, metaicl_model, anchor_prompt, dp, dp["task"], return_loss=True)
    #         for dp in dev_data
    #     ])
    #     total_flops = sum(flops)

    #     base_loss_tensor = torch.tensor(base_losses, device=device)
    #     grad_tensor = torch.stack([torch.stack(g, dim=0) for g in base_gradients], dim=0)  # [len(dev), num_labels, D]

    #     # Step 3: For the other 99 combinations, estimate with Taylor
    #     accuracy_results = []
    #     for comb in tqdm(sampled_combinations, total=len(sampled_combinations)):
    #         if comb == anchor_comb:
    #             continue

    #         target_prompt = "".join([
    #             f"Input: {test_data[idx]['input']} Label: {test_data[idx]['output']}\n" for idx in comb
    #         ])

    #         correct = 0
    #         for dp_idx, dp in enumerate(dev_data):
    #             dev_str = f"Input: {dp['input']} Label:"
    #             delta_P = self.compute_embedding_difference_(
    #                 gpt2, metaicl_model, anchor_prompt + dev_str, target_prompt + dev_str
    #             )

    #             taylor_losses = []
    #             for j in range(len(base_loss_tensor[dp_idx])):  # over labels
    #                 correction = torch.sum(grad_tensor[dp_idx][j] * delta_P).item()
    #                 approx_loss = base_loss_tensor[dp_idx][j].item() + correction
    #                 taylor_losses.append(approx_loss)

    #             pred_id = np.argmin(taylor_losses)
    #             pred = dp["options"][pred_id]
    #             if pred == dp["output"]:
    #                 correct += 1

    #         acc = correct / len(dev_data)
    #         accuracy_results.append((comb, acc))

    #     point_scores = {i: [] for i in range(len(test_data))}

    #     for comb, acc in accuracy_results:
    #         for idx in comb:
    #             point_scores[idx].append(acc)

    #     avg_scores = []
    #     for idx, scores in point_scores.items():
    #         if scores:
    #             avg_scores.append((idx, sum(scores) / len(scores)))

    #     avg_scores.sort(key=lambda x: -x[1])
    #     final_indices = [idx for idx, _ in avg_scores[:self.k]]
    #     selected_data = [test_data[i] for i in final_indices]

    #     return selected_data, accuracy_results[0][1], total_flops
    
