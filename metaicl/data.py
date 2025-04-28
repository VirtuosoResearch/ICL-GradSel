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

logging.getLogger("thop").setLevel(logging.WARNING)

# import sys
# import os
# sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
# from forward_path import Forward

import random
from collections import defaultdict
from functools import partial
from multiprocessing import Pool

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from sklearn.linear_model import LinearRegression


from utils.data import load_data
from metaicl.model import MetaICLModel

class MetaICLData(object):

    def __init__(self, device=0, logger=None, tokenizer=None, method="channel", use_demonstrations=True, k=16, max_length=1024,
                 do_tensorize=False, tensorize_dir=None, n_process=None, n_gpu=None, local_rank=-1, is_flops=False):

        self.logger = logger
        self.tokenizer = tokenizer
        self.method = method
        self.use_demonstrations = use_demonstrations
        self.k = k
        self.max_length = max_length

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

        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.padding_side == "left":
            self.tokenizer.padding_side = "right"

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
            results = metaicl_model.run_model(input_ids, attention_mask, token_type_ids)

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

    def greedy_select_condition(self, gpt2, metaicl_model, test_data, dev_data, subset_size=10):
        selected_indices, best_demonstrations = [], []
        best_demonstrations = []

        total_flops = 0
        loss_list = []
        for i in range(len(test_data)):
            candidate_demonstrations = best_demonstrations + [test_data[i]]
            candidate_loss, flops = self.evaluate_loss(gpt2, metaicl_model, candidate_demonstrations, dev_data, test_data[0]["task"])
            loss_list.append(candidate_loss)
            total_flops+=flops
            self.logger.info(f"----------------candidate_loss : {candidate_loss}----------------")

        loss_array = np.array(loss_list)
        selected_indices = np.argsort(loss_array)[:self.k]
        for i in selected_indices:
            best_demonstrations.append(test_data[i])
        self.logger.info("---------best_demonstrations---------")
        self.logger.info(best_demonstrations)
        return best_demonstrations, 0, total_flops

    def greedy_select_subset(self, gpt2, test_data, dev_data, subset_size=10):
        selected_indices, best_demonstrations = [], []
        best_demonstrations = []

        add_newlines = False
        checkpoint = None
        metaicl_model = MetaICLModel(logger=self.logger, out_dir= "./cache", device_num=self.device)
        metaicl_model.load(checkpoint, gpt2=gpt2)

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

    def tensorize_ground(self, gpt2, _test_data, _val_data, options=None, add_newlines=True):
        print("options: ", options)
        if options is not None:
            print(_test_data[0])
            for i, dp in enumerate(_test_data):
                _test_data[i] = {"input": dp, "options": options}
            for i, dp in enumerate(_val_data):
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
        for dp_idx, dp in tqdm(enumerate(val_data), total=len(val_data), leave=True, position=0):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)
            if self.use_demonstrations:
                test_text = dp["input"]
                dp_feature = val_features[dp_idx]

                samples, top_indices, _ = self._select_top_k_neighbors(dp_feature, test_features, test_data, k=12,dp_idx=-1)

                ground, _, flops = self.greedy_select_condition(gpt2=gpt2, metaicl_model=metaicl_model,test_data=samples, dev_data=dev_data, subset_size=self.k)
                total_flops+=flops
                # def greedy_select_subset(self, test_data, dp_data, subset_size=10):
                demonstrations = []
                for i, neighbor_dp in enumerate(ground):
                    # print("neighbor_dp : ",neighbor_dp)
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
        
        if self.is_flops: self.logger.info(f"Total_FLOPS: {total_flops / 1e9:.2f} GFLOPs")

        self.tensorized_inputs = dict(input_ids=torch.LongTensor(input_ids),
                                      attention_mask=torch.LongTensor(attention_mask),
                                      token_type_ids=torch.LongTensor(token_type_ids))
        self.metadata = metadata

    def compute_loss_and_gradient(self, gpt2, metaicl_model, tokenizer, input_tokens, output_tokens, device):

        tokenizer.pad_token = tokenizer.eos_token
        token_input = tokenizer(input_tokens, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
        input_ids = token_input["input_ids"].to(device)
        attention_mask = token_input["attention_mask"].to(device)
        output_ids = tokenizer(output_tokens, return_tensors="pt")["input_ids"][0][1].to(device)

        with torch.no_grad():
            if "gpt" in gpt2:
                embedding = metaicl_model.model.transformer.wte(input_ids)
            elif "opt" in gpt2: embedding = metaicl_model.model.model.decoder.embed_tokens(input_ids)
            else:
                embedding = metaicl_model.model.model.embed_tokens(input_ids)
        embedding.requires_grad = True 
        embedding = embedding.to(metaicl_model.model.dtype)

        output_logits = metaicl_model.model(inputs_embeds=embedding, attention_mask=attention_mask).logits
        last_token_idx = attention_mask.sum(dim=1).item()-1
        # last_token_idx = input_ids.shape[1] - 1 
        log_probs = F.log_softmax(output_logits[0, last_token_idx, :], dim=-1) 

        target_token = output_ids
        loss = -log_probs[target_token]
        
        # flops, params = profile(metaicl_model.model, inputs=(input_ids,))
        flops = 0
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
    
    def tensorize_estimate(self, gpt2, _test_data, _val_data, is_quant, pseudo_k=3, options=None, add_newlines=True):
        print("options: ", options)
        if options is not None:
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

        self.logger.info(f"accuracy: {np.mean(accs)}")
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

            # que_tokens = self.tokenizer("\nHere is the query input: ")["input_ids"]
            # input_tokens = [que_tokens + input_tokens[1:] for _ in option_tokens]
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

        instructions = f"Here are {len(test_data[0]['options'])} options: "
        for option in test_data[0]["options"]: instructions+=option+", "
        instructions+=f"You should choose one of them to answer at the end. \nHere are {self.k} samples for your reference. \n"
        init_tokens = self.tokenizer(instructions)["input_ids"][1:]
        # print("input_text : ", self.tokenizer.decode(init_tokens))

        for dp_idx, dp in enumerate(val_data):
            inputs, outputs, answer = self._prepro_each_datapoint( # for_demonstrations is Talse!!
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)
            # print("*********** seperate ***********")
            if self.use_demonstrations:
                dp_feature = val_features[dp_idx]            

                top_k_neighbors, _, __ = self._select_top_k_neighbors(
                    dp_feature, test_features, test_data, self.k, dp_idx
                )

                demonstrations = init_tokens
                for i, neighbor_dp in enumerate(top_k_neighbors):
                    # print("neighbor_dp: ",neighbor_dp)
                    # neighbor_dp["input"] = "input: " + neighbor_dp["input"]
                    # neighbor_dp["output"] = "output: "+ neighbor_dp["output"]
                    input_, output_ = self._prepro_each_datapoint( # for_demonstrations is True!!
                        neighbor_dp, is_first=i == 0, for_demonstrations=True, add_newlines=add_newlines)
                    demonstrations += input_[1:] + output_[1:]
                #print(demonstrations)
            indices = [[i] for i in range(len(input_ids), len(input_ids) + len(inputs))]


            metadata.append({"indices": indices, "answer": answer, "options": dp["options"]})

            for inputs_, outputs_ in zip(inputs, outputs):

                if self.use_demonstrations:
                    inputs_ = demonstrations + inputs_[1:]
                encoded = prepro_sentence_pair_single(
                    inputs_, [outputs_[2]], self.max_length, self.tokenizer,self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, 
                    allow_truncation=self.use_demonstrations
                )
                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])

        self.tensorized_inputs = dict(input_ids=torch.LongTensor(input_ids),
                                      attention_mask=torch.LongTensor(attention_mask),
                                      token_type_ids=torch.LongTensor(token_type_ids))
        self.metadata = metadata


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

        instructions = f"Here are {len(test_data[0]['options'])} options: "
        for option in test_data[0]["options"]: instructions+=option+", "
        instructions+=f"You should choose one of them to answer at the end. \nHere are {self.k} samples for your reference. \n"
        init_tokens = self.tokenizer(instructions)["input_ids"][1:]
        # print("input_text : ", self.tokenizer.decode(init_tokens))

        for dp_idx, dp in enumerate(val_data):
            inputs, outputs, answer = self._prepro_each_datapoint( # for_demonstrations is Talse!!
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)
            # print("*********** seperate ***********")
            if self.use_demonstrations:
                dp_feature = val_features[dp_idx]            

                random_k_neighbors = self._select_random_k_neighbors(
                    dp_feature, test_features, test_data, self.k, dp_idx
                )

                demonstrations = init_tokens
                for i, neighbor_dp in enumerate(random_k_neighbors):
                    # print("neighbor_dp: ",neighbor_dp)
                    # neighbor_dp["input"] = "input: " + neighbor_dp["input"]
                    # neighbor_dp["output"] = "output: "+ neighbor_dp["output"]
                    input_, output_ = self._prepro_each_datapoint( # for_demonstrations is True!!
                        neighbor_dp, is_first=i == 0, for_demonstrations=True, add_newlines=add_newlines)
                    demonstrations += input_[1:] + output_[1:]
                #print(demonstrations)
            indices = [[i] for i in range(len(input_ids), len(input_ids) + len(inputs))]


            metadata.append({"indices": indices, "answer": answer, "options": dp["options"]})

            for inputs_, outputs_ in zip(inputs, outputs):

                if self.use_demonstrations:
                    inputs_ = demonstrations + inputs_[1:]
                encoded = prepro_sentence_pair_single(
                    inputs_, [outputs_[2]], self.max_length, self.tokenizer,self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, 
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

def prepro_sentence_pair_single(ids1, ids2, max_length,
                                tokenizer, bos_token_id, eos_token_id,
                                allow_truncation=False):
    # Remove special tokens
    special_ids = set(tokenizer.all_special_ids)

    ids1 = [i for i in ids1 if i not in special_ids]
    ids2 = [i for i in ids2 if i not in special_ids]

    # Add bos and eos tokens later, so leave space for them
    total_len = len(ids1) + len(ids2) + 2  # +2 for bos and eos

    if allow_truncation and total_len > max_length:
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

