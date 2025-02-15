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


class MetaICLData(object):

    def __init__(self, device=0, logger=None, tokenizer=None, method="channel", use_demonstrations=True, k=16,
                 max_length=1024, max_length_per_example=256,
                 do_tensorize=False, tensorize_dir=None, n_process=None, n_gpu=None, local_rank=-1):

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

        # print(metaicl_model)
        if "Llama" in gpt2:
            metaicl_model.resize(tokenizer)

        max_length_per_example, max_length = 128, 256
        if self.use_demonstrations:
            max_length = min(max_length * self.k, 512)

        def run_a_forward_pass(input_tokens, output_tokens, tokenizer):
            encoded = prepro_sentence_pair_single(
                        input_tokens, output_tokens, max_length=512, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
                        allow_truncation=self.use_demonstrations
                )
            input_ids = torch.LongTensor([encoded[0]])
            attention_mask = torch.LongTensor([encoded[1]])
            token_type_ids = torch.LongTensor([encoded[2]])
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            results = metaicl_model.run_model(input_ids, attention_mask, token_type_ids)
            return input_ids, results.cpu().detach().item()

        option_tokens = [tokenizer(option)["input_ids"] for option in dp['options']]
        input_tokens = tokenizer("Input: " + dp["input"] + " " + "Label: ")["input_ids"]
        metaicl_model.model.eval()
        metaicl_model.model.to(device)
        demonstrations = []

        one_trial_losses = []
        for option_token in option_tokens:
            input_ids, results = run_a_forward_pass(demonstrations + input_tokens, option_token, tokenizer)
            one_trial_losses.append(results)

        label_id = np.argmin(one_trial_losses)
        label = dp["options"][label_id]
        if return_loss:
            return one_trial_losses
        return label_id, label

    def evaluate_accuracy(self, gpt2, metaicl_model, demonstrations, dev_data, task):
        correct = 0; total = len(dev_data)
        input_str = ""
        for item in demonstrations:
            input_str = input_str + "Input: "+ item["input"] + " "+ "Label: "+item["output"]+"\n"
        input_token = self.tokenizer(input_str)["input_ids"]
        for idx, dp in enumerate(dev_data):
            _, label = self.forward(gpt2, metaicl_model, input_token, dp, task)
            if label == dp["output"]: correct += 1
        return correct / total if total > 0 else 0

    def evaluate_conditional(self, gpt2, model, demonstrations, dev_data, task):
        pass

    def greedy_select_subset(self, gpt2, test_data, dev_data, subset_size=10):
        selected_indices, best_demonstrations = [], []
        best_demonstrations = []

        add_newlines = False
        checkpoint = None
        metaicl_model = MetaICLModel(logger=self.logger, out_dir= "./cache", device_num=self.device)
        metaicl_model.load(checkpoint, gpt2=gpt2)

        while len(selected_indices) < subset_size:
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
            if idx<= len(_val_data)//2: val_data.append(dp.copy())
            else: dev_data.append(dp.copy())
        task = _test_data[0]["task"]
        with open(f"./features/{task}_test_features.json", "r") as file: test_features = json.load(file)
        with open(f"./features/{task}_val_features.json", "r") as file: val_features = json.load(file)

        if self.use_demonstrations:
            test_texts = [dp["input"] + " " + dp["output"] for dp in test_data]
            test_labels = [dp["output"] for dp in test_data]

        input_ids, attention_mask, token_type_ids = [], [], []
        metadata = []

        ground, _ = self.greedy_select_subset(gpt2=gpt2,test_data=test_data, dev_data=dev_data, subset_size=self.k)

        for dp_idx, dp in enumerate(val_data):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)

            if self.use_demonstrations:
                test_text = dp["input"]
                dp_feature = val_features[dp_idx]

                
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
                    inputs_, outputs_, self.max_length, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id,
                    allow_truncation=self.use_demonstrations
                )
                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])

        self.tensorized_inputs = dict(input_ids=torch.LongTensor(input_ids),
                                      attention_mask=torch.LongTensor(attention_mask),
                                      token_type_ids=torch.LongTensor(token_type_ids))
        self.metadata = metadata

    def compute_loss_and_gradient(self, gpt2, model, tokenizer, input_tokens, output_tokens, device):

        tokenizer.pad_token = tokenizer.eos_token
        input_ids = tokenizer(input_tokens, return_tensors="pt", padding="max_length", truncation=True, max_length=800).input_ids.to(device)
        output_ids = tokenizer(output_tokens, return_tensors="pt")["input_ids"][0][-1].to(device)

        with torch.no_grad():
            if "gpt" in gpt2:
                embedding = model.transformer.wte(input_ids)
            else:
                embedding = model.model.embed_tokens(input_ids)
        embedding.requires_grad = True 

        output_logits = model(inputs_embeds=embedding).logits
        last_token_idx = input_ids.shape[1] - 1 
        log_probs = F.log_softmax(output_logits[0, last_token_idx, :], dim=-1) 

        target_token = output_ids
        loss = -log_probs[target_token]
        
        # self.logger.info(f"target_token : {target_token}")
        # self.logger.info(f"loss : {loss}")
        
        loss.backward()
        return loss.item(), embedding.grad

    def forward2(self, gpt2, metaicl_model, demonstrations, dp, task, return_loss=False):

        logger = logging.getLogger(__name__)
        device = torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")
        tokenizer = self.tokenizer

        # metaicl_model = MetaICLModel(logger=logger, out_dir="./cache", device_num=self.device)
        # metaicl_model.load(None, gpt2=gpt2)
        
        if "Llama" in gpt2:
            metaicl_model.resize_token_embeddings(len(tokenizer))

        option_tokens = dp['options']
        input_tokens = demonstrations + "Input: " + dp["input"] + " Label:"

        losses = []
        gradients = []
        for option in option_tokens:
            loss, grad = self.compute_loss_and_gradient(gpt2, metaicl_model, tokenizer, input_tokens, option, device)
            # compute_loss_and_gradient(self, gpt2, model, tokenizer, input_tokens, output_tokens, device):
            losses.append(loss)
            gradients.append(grad)

        label_id = np.argmin(losses)
        label = dp["options"][label_id]

        if return_loss:
            return losses, gradients, label_id
        return label_id, label

    def compute_embedding_difference(self, gpt2, metaicl_model, base_str, candidate_str):
        device = torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")
        input_tokens_1 = self.tokenizer(base_str, return_tensors="pt", padding="max_length", truncation=True, max_length=800)["input_ids"].to(device)
        input_tokens_2 = self.tokenizer(candidate_str, return_tensors="pt", padding="max_length", truncation=True, max_length=800)["input_ids"].to(device)

        tokenizer = self.tokenizer

        
        if "Llama" in gpt2:
            metaicl_model.resize_token_embeddings(len(tokenizer))


        with torch.no_grad():
            if "gpt" in gpt2:
                embedding_1 = metaicl_model.transformer.wte(input_tokens_1)
                embedding_2 = metaicl_model.transformer.wte(input_tokens_2)
            else:
                embedding_1 = metaicl_model.model.embed_tokens(input_tokens_1)
                embedding_2 = metaicl_model.model.embed_tokens(input_tokens_2)

        delta_P = embedding_2 - embedding_1.detach()


        return delta_P

    def greedy_select_subset2(self, gpt2, test_data, dev_data):
        selected_indices, best_demonstrations = [], []
        best_input_str = ""
        best_accuracy = 0.0
        device = torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")
        if "gpt2" in gpt2:
            metaicl_model = GPT2LMHeadModel.from_pretrained(gpt2)
        elif "opt" in gpt2:
            metaicl_model = OPTForCausalLM.from_pretrained(gpt2)
        else:
            metaicl_model = AutoModelForCausalLM.from_pretrained(gpt2)
        metaicl_model = metaicl_model.to(device)        


        while len(selected_indices) < self.k:
            
            base_index = next(i for i in range(len(test_data)) if i not in selected_indices)
            base_test_example = test_data[base_index]
            # print(f"test_data[base_index] : {test_data[base_index]}")
            base_str = "Input: " + test_data[base_index]["input"] + " Label: " + test_data[base_index]["output"]+"\n"

            base_loss_lists, base_gradients = [], []
            self.logger.info(f"============ len(dev_data): {len(dev_data)} ============")
            cnt = 0
            for dp in tqdm(dev_data):
                losses, grads, label_id = self.forward2(gpt2, metaicl_model, best_input_str, dp, base_test_example["task"], return_loss=True)
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
                    delta_P = self.compute_embedding_difference(gpt2, metaicl_model, best_input_str+base_str+dev_str, best_input_str+candidate_str+dev_str)  # P(S', x_q) - P(S, x_q)
                    # delta_P = torch.zeros((1,1,800,1280), device="cuda:1")

                    # self.logger.info(f"========delta_P:========\n{delta_P}")
                    # self.logger.info(f"========delta_P.size(): {delta_P.size()}")
                    # self.logger.info(f"========embedding_gradient:=======\n{embedding_gradient}")
                    # self.logger.info(f"========embedding_gradient.size(): {embedding_gradient.size()}")

                    for j in range(len(base_loss_lists[0])):
                        taylor_correction = torch.sum(embedding_gradient[dp_idx][j] * delta_P).item()
                        taylor_approx_loss = base_loss_tensor[dp_idx][j].item() + taylor_correction
                        taylor_loss_list.append(taylor_approx_loss)

                    # self.logger.info(f"delta_P : {delta_P}")
                    # self.logger.info(f"base_loss_lists : {base_loss_lists[dp_idx]}; taylor_loss_list : {taylor_loss_list}")

                    predicted_label_id = np.argmin(taylor_loss_list)
                    predicted_label = test_data[i]["options"][predicted_label_id]

                    if predicted_label == dp["output"]:
                        correct_count += 1

                candidate_accuracy = correct_count / len(dev_data)
                if candidate_accuracy > best_candidate_accuracy:
                    best_candidate = i
                    best_candidate_accuracy = candidate_accuracy
            
            self.logger.info("-------------one loop done--------------")
            selected_indices.append(best_candidate)
            best_input_str += "Input: " + test_data[best_candidate]["input"] + " Label: " + test_data[best_candidate]["output"]+"\n"
            best_demonstrations.append(test_data[best_candidate])
            self.logger.info(f"Selected index {best_candidate}, current best accuracy: {best_candidate_accuracy:.4f}")

        return best_demonstrations, best_candidate_accuracy

    def tensorize_estimate(self, gpt2, _test_data, _val_data, options=None, add_newlines=True):
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
            if idx<= len(_val_data)//2: val_data.append(dp.copy())
            else: dev_data.append(dp.copy())
        task = _test_data[0]["task"]
        with open(f"./features/{task}_test_features.json", "r") as file: test_features = json.load(file)
        with open(f"./features/{task}_val_features.json", "r") as file: val_features = json.load(file)

        if self.use_demonstrations:
            test_texts = [dp["input"] + " " + dp["output"] for dp in test_data]
            test_labels = [dp["output"] for dp in test_data]

        input_ids, attention_mask, token_type_ids = [], [], []
        metadata = []
        
        ground, _ = self.greedy_select_subset2(gpt2=gpt2, test_data=test_data, dev_data=dev_data)
        demonstrations = []
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
                    inputs_, outputs_, self.max_length, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id,
                    allow_truncation=self.use_demonstrations
                )
                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])

        self.tensorized_inputs = dict(input_ids=torch.LongTensor(input_ids),
                                      attention_mask=torch.LongTensor(attention_mask),
                                      token_type_ids=torch.LongTensor(token_type_ids))
        self.metadata = metadata



    def tensorize_unlabeled(self, gpt2, _test_data, _val_data, m, options=None, add_newlines=True):
        val_data, test_data, unlab_data =  [], [], []
        cut_point = len(_test_data)-len(_test_data)//4
        for idx, dp in enumerate(_test_data):
            if "output" not in dp: dp["output"] = dp["options"][0]
            if idx < cut_point: test_data.append(dp.copy())
            else: unlab_data.append(dp.copy())
        for dp in _val_data:
            if "output" not in dp: dp["output"] = dp["options"][0]
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
        if self.use_demonstrations:
            test_texts = [dp["input"] + " " + dp["output"] for dp in test_data]
            test_labels = [dp["output"] for dp in test_data]

        all_indices = [i for i in range(len(test_data))]
        # print(all_indices)
        similarities = [0 for i in range(len(test_data))]
        forsel, _ = self._forward_selection(
            embeddings=test_features,
            top_k_indices=all_indices,
            m=m, 
            candidate_labels=test_labels, 
            test_data=test_data,
            similarities = similarities,
            seed=42
        )
        demonstration,psu_data = [], []
        for dp in forsel:
            demonstration+=self.tokenizer("Input: " + dp["input"] + " " + "Label: "+dp["output"])["input_ids"]
        for idx, dp in enumerate(unlab_data):
            used = dp["output"]
            _, dp["output"]= self.forward(gpt2, demonstration,dp,dp["task"])
            self.logger.info(used+" ;;; "+dp["output"])
            psu_data.append(dp)

        for dp in psu_data:
            test_data.append(dp)
        test_labels = [dp["output"] for dp in test_data]
        # print("len(test_data) : ",len(test_data))

        for dp_idx, dp in enumerate(val_data):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)

            if self.use_demonstrations:
                test_text = dp["input"]
                dp_feature = val_features[dp_idx]            

                top_k_neighbors, top_k_indices, similarities = self._select_top_k_neighbors(
                    dp_feature, test_features, test_data, self.k, len(test_data)+1
                )
                
                # print("similarities : ",similarities)

                forsel, _ = self._forward_selection(
                    embeddings=test_features,
                    top_k_indices=top_k_indices,
                    m=m, 
                    candidate_labels=test_labels, 
                    test_data=test_data,
                    similarities = similarities,
                    seed=42
                )

                demonstrations = []
                for i, neighbor_dp in enumerate(forsel):
                    input_, output_ = self._prepro_each_datapoint(
                        neighbor_dp, is_first=i == 0, for_demonstrations=True, add_newlines=add_newlines)
                    demonstrations += input_ + output_

            indices = [[i] for i in range(len(input_ids), len(input_ids) + len(inputs))]

            metadata.append({"indices": indices, "answer": answer, "options": dp["options"]})

            for inputs_, outputs_ in zip(inputs, outputs):
                if self.use_demonstrations:
                    inputs_ = demonstrations + inputs_
                encoded = prepro_sentence_pair_single(
                    inputs_, outputs_, self.max_length, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id,
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
                        dp["input"] = "\n\n" + dp["input"]
                    else:
                        dp["input"] = "\n\n\n" + dp["input"]
                if not no_label:
                    dp["output"] = "\n" + dp["output"]
                    if "options" in dp:
                        dp["options"] = ["\n" + opt for opt in dp["options"]]
            elif self.method=="channel":
                if not is_first:
                    dp["output"] = "\n\n\n" + dp["output"]
                    if "options" in dp:
                        dp["options"] = ["\n\n\n" + opt for opt in dp["options"]]
                if not no_input:
                    if not no_label:
                        dp["input"] = "\n" + dp["input"]
            else:
                raise NotImplementedError()
        else:
            if not is_first:
                if self.method=="direct":
                    dp["input"] = " " + dp["input"]
                elif self.method=="channel":
                    dp["output"] = " " + dp["output"]
                    if "options" in dp:
                        dp["options"] = [" "+opt for opt in dp["options"]]
                else:
                    raise NotImplementedError()
            if self.method=="direct":
                dp["output"] = " " + dp["output"]
                if "options" in dp:
                    dp["options"] = [" " + opt for opt in dp["options"]]
            elif self.method=="channel":
                dp["input"] = " " + dp["input"]
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
        # print(top_k_indices)
        # print(len(similarities))
        # print("len(test_data)",len(test_data))
        return [test_data[i] for i in top_k_indices], top_k_indices , similarities
    
    def tensorize_topk(self, _test_data, _val_data, options=None, add_newlines=True):
        if options is not None:
            for i, dp in enumerate(_test_data):
                print(i,dp)
                assert "options" not in dp
                assert type(dp) == str
                _test_data[i] = {"input": dp, "options": options}
            for i, dp in enumerate(_val_data):
                assert "options" not in dp
                assert type(dp) == str
                _val_data[i] = {"input": dp, "options": options}
        print("len(_test_data) : ",len(_test_data))
        print("len(_val_data) : ", len(_val_data))

        val_data, test_data =  [], []

        for dp in _test_data:
            assert type(dp) == dict, ("Each example should be a dictionary", dp)
            assert "input" in dp and "options" in dp and type(dp["options"]) == list, \
                ("Test example should contain input and options in a list format", dp)
            if "output" not in dp:
                dp["output"] = dp["options"][0]  # randomly choose one (we don't need it anyways)
            test_data.append(dp.copy())
        for dp in _val_data:
            assert type(dp) == dict, ("Each example should be a dictionary", dp)
            assert "input" in dp and "options" in dp and type(dp["options"]) == list, \
                ("Test example should contain input and options in a list format", dp)
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
                    inputs_, outputs_, self.max_length, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id,
                    allow_truncation=self.use_demonstrations
                )
                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])

        self.tensorized_inputs = dict(input_ids=torch.LongTensor(input_ids),
                                      attention_mask=torch.LongTensor(attention_mask),
                                      token_type_ids=torch.LongTensor(token_type_ids))
        self.metadata = metadata


    def greedy_supcon(self, embeddings, top_k_indices, m, candidate_labels, test_data, similarities, temperature=0.1, softloss=True):

        assert m <= len(top_k_indices), "Error: m must less than k"

        all_combinations = list(combinations(top_k_indices, m))

        best_combination_data = []
        best_loss = float('inf') 
        best_labels = []

        for idx in range(len(embeddings)):
            embeddings[idx] = torch.tensor(embeddings[idx], dtype=torch.float32)
            embeddings[idx] = embeddings[idx] / torch.norm(embeddings[idx])

        for combination in all_combinations:
            selected_embeddings = [embeddings[i] for i in combination]
            selected_data = [test_data[i] for i in combination]

            # compute simloss
            simloss = 0.0
            for idx in combination:
                simloss+=similarities[idx]
            simloss /= m
            
            # compute supcon loss
            con_loss, pos_loss, all_loss = 0.0, 0.0, 0.0
            temperature=0.1
            if softloss==False:
                for idx1 in combination:
                    cnt_pos=0
                    pos_loss = all_loss =0.0
                    for idx2 in combination:
                        if idx1==idx2: continue
                        idx1_embedding, idx1_label = embeddings[idx1], candidate_labels[idx1]
                        idx2_embedding, idx2_label = embeddings[idx2], candidate_labels[idx2]
                        loss = torch.exp(torch.matmul(idx1_embedding, idx2_embedding) / temperature)
                        if idx1_label == idx2_label:
                            pos_loss += loss
                            cnt_pos+=1
                        all_loss += loss
                    idx_loss = pos_loss / all_loss / cnt_pos
                    con_loss += torch.tensor(-1.0, dtype=float)  * torch.log(idx_loss)
            else:
                for idx1 in combination:
                    cnt_pos = 0
                    pos_loss = 0.0
                    idx1_embedding, idx1_label = embeddings[idx1], candidate_labels[idx1]
                    logits = []

                    for idx2 in combination:
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

            current_labels = [candidate_labels[i] for i in combination]
            lam=0.05
            simcon_loss = -simloss + lam*con_loss
            # print("simloss : ",simloss, "con_loss : ",con_loss, "simcon_loss : ",simcon_loss)
            if simcon_loss < best_loss:
                best_loss = simcon_loss
                best_combination_data = selected_data
                best_labels = current_labels
            
        return best_combination_data, best_labels

    def tensorize_supcon(self, _test_data, _val_data, m, options=None, add_newlines=True):
        if options is not None:
            for i, dp in enumerate(_test_data):
                assert "options" not in dp
                assert type(dp) == str
                _test_data[i] = {"input": dp, "options": options}
            for i, dp in enumerate(_val_data):
                assert "options" not in dp
                assert type(dp) == str
                _val_data[i] = {"input": dp, "options": options}
        print("len(_test_data) : ",len(_test_data))
        print("len(_val_data) : ", len(_val_data))

        val_data, test_data =  [], []

        for dp in _test_data:
            assert type(dp) == dict, ("Each example should be a dictionary", dp)
            assert "input" in dp and "options" in dp and type(dp["options"]) == list, \
                ("Test example should contain input and options in a list format", dp)
            if "output" not in dp:
                dp["output"] = dp["options"][0]  # randomly choose one (we don't need it anyways)
            test_data.append(dp.copy())
        for dp in _val_data:
            assert type(dp) == dict, ("Each example should be a dictionary", dp)
            assert "input" in dp and "options" in dp and type(dp["options"]) == list, \
                ("Test example should contain input and options in a list format", dp)
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


        if self.use_demonstrations:
            test_texts = [dp["input"] + " " + dp["output"] for dp in test_data]
            test_labels = [dp["output"] for dp in test_data]

        input_ids, attention_mask, token_type_ids = [], [], []
        metadata = []

        for dp_idx, dp in enumerate(val_data):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)

            if self.use_demonstrations:
                test_text = dp["input"]
                dp_feature = val_features[dp_idx]            

                top_k_neighbors, top_k_indices, similarities = self._select_top_k_neighbors(
                    dp_feature, test_features, test_data, self.k, dp_idx
                )

                greedy, best_labels = self.greedy_supcon(
                    embeddings=test_features,
                    top_k_indices=top_k_indices,
                    m=m, 
                    candidate_labels=test_labels, 
                    test_data=test_data,
                    similarities = similarities
                )

                demonstrations = []
                for i, neighbor_dp in enumerate(greedy):
                    input_, output_ = self._prepro_each_datapoint(
                        neighbor_dp, is_first=i == 0, for_demonstrations=True, add_newlines=add_newlines)
                    demonstrations += input_ + output_

            indices = [[i] for i in range(len(input_ids), len(input_ids) + len(inputs))]

            metadata.append({"indices": indices, "answer": answer, "options": dp["options"]})

            for inputs_, outputs_ in zip(inputs, outputs):
                if self.use_demonstrations:
                    inputs_ = demonstrations + inputs_
                encoded = prepro_sentence_pair_single(
                    inputs_, outputs_, self.max_length, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id,
                    allow_truncation=self.use_demonstrations
                )
                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])

        self.tensorized_inputs = dict(input_ids=torch.LongTensor(input_ids),
                                      attention_mask=torch.LongTensor(attention_mask),
                                      token_type_ids=torch.LongTensor(token_type_ids))
        self.metadata = metadata

    def _random_ensemble(self, embeddings, top_k_indices, m, candidate_labels, test_data, similarities, seed, temperature=0.1):

        assert m <= len(top_k_indices), "Error: m must less than k"

        all_combinations = list(combinations(top_k_indices, m))
        random.seed(seed)
        random.shuffle(all_combinations)

        for idx in range(len(embeddings)):
            embeddings[idx] = torch.tensor(embeddings[idx], dtype=torch.float32)
            embeddings[idx] = embeddings[idx] / torch.norm(embeddings[idx])

        all_loss_list = []

        for idx, combination in enumerate(all_combinations):
            if idx >=len(all_combinations)/2: break
            selected_embeddings = [embeddings[i] for i in combination]
            selected_data = [test_data[i] for i in combination]

            # compute simloss
            simloss = 0.0
            for idx in combination:
                simloss+=similarities[idx]
            simloss /= m
            
            # compute supcon loss
            con_loss, pos_loss, all_loss = 0.0, 0.0, 0.0
            for idx1 in combination:
                cnt_pos = 0
                pos_loss = 0.0
                idx1_embedding, idx1_label = embeddings[idx1], candidate_labels[idx1]
                logits = []
                for idx2 in combination:
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

            current_labels = [candidate_labels[i] for i in combination]
            lam=0.05
            simcon_loss = -simloss + lam*con_loss
            # print("simloss : ",simloss, "con_loss : ",con_loss, "simcon_loss : ",simcon_loss)
            all_loss_list.append(simcon_loss)

        point_score = [1001.0 for i in range(len(top_k_indices))]
        cnt_point = [0 for i in range(len(top_k_indices))]
        for i, combination in enumerate(all_combinations):
            if i>=len(all_combinations)/2: break
            # print(f"combination: {combination}")
            for j, indice in enumerate(top_k_indices):
                if indice in combination:
                    if point_score[j]>1000.0:
                        point_score[j] = all_loss_list[i]
                    else:
                        point_score[j]+=all_loss_list[i]
                    cnt_point[j]+=1
        for i in range(len(point_score)):
            if point_score[i] <= 1000.0:
                point_score[i]/=cnt_point[i]
        
        indexed_score = list(enumerate(point_score))
        sorted_score = sorted(indexed_score, key=lambda x: x[1])
        min_indices = [x[0] for x in sorted_score[:m]]
        real_indices = [top_k_indices[x] for x in min_indices]

        return [test_data[x] for x in real_indices]

     
    def tensorize_ranens(self, _test_data, _val_data, m, seed, options=None, add_newlines=True):
        if options is not None:
            for i, dp in enumerate(_test_data):
                assert "options" not in dp
                assert type(dp) == str
                _test_data[i] = {"input": dp, "options": options}
            for i, dp in enumerate(_val_data):
                assert "options" not in dp
                assert type(dp) == str
                _val_data[i] = {"input": dp, "options": options}
        print("len(_test_data) : ",len(_test_data))
        print("len(_val_data) : ", len(_val_data))

        val_data, test_data =  [], []

        for dp in _test_data:
            assert type(dp) == dict, ("Each example should be a dictionary", dp)
            assert "input" in dp and "options" in dp and type(dp["options"]) == list, \
                ("Test example should contain input and options in a list format", dp)
            if "output" not in dp:
                dp["output"] = dp["options"][0]  # randomly choose one (we don't need it anyways)
            test_data.append(dp.copy())
        for dp in _val_data:
            assert type(dp) == dict, ("Each example should be a dictionary", dp)
            assert "input" in dp and "options" in dp and type(dp["options"]) == list, \
                ("Test example should contain input and options in a list format", dp)
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
        

        if self.use_demonstrations:
            test_texts = [dp["input"] + " " + dp["output"] for dp in test_data]
            test_labels = [dp["output"] for dp in test_data]

        input_ids, attention_mask, token_type_ids = [], [], []
        metadata = []

        for dp_idx, dp in enumerate(val_data):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)

            if self.use_demonstrations:
                test_text = dp["input"]
                dp_feature = val_features[dp_idx]            

                top_k_neighbors, top_k_indices, similarities = self._select_top_k_neighbors(
                    dp_feature, test_features, test_data, self.k, dp_idx
                )
                
                # print("similarities : ",similarities)

                ranens = self._random_ensemble(
                    embeddings=test_features,
                    top_k_indices=top_k_indices,
                    m=m, 
                    candidate_labels=test_labels, 
                    test_data=test_data,
                    similarities = similarities,
                    seed=seed
                )

                demonstrations = []
                for i, neighbor_dp in enumerate(ranens):
                    input_, output_ = self._prepro_each_datapoint(
                        neighbor_dp, is_first=i == 0, for_demonstrations=True, add_newlines=add_newlines)
                    demonstrations += input_ + output_

            indices = [[i] for i in range(len(input_ids), len(input_ids) + len(inputs))]

            metadata.append({"indices": indices, "answer": answer, "options": dp["options"]})

            for inputs_, outputs_ in zip(inputs, outputs):
                if self.use_demonstrations:
                    inputs_ = demonstrations + inputs_
                encoded = prepro_sentence_pair_single(
                    inputs_, outputs_, self.max_length, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id,
                    allow_truncation=self.use_demonstrations
                )
                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])

        self.tensorized_inputs = dict(input_ids=torch.LongTensor(input_ids),
                                      attention_mask=torch.LongTensor(attention_mask),
                                      token_type_ids=torch.LongTensor(token_type_ids))
        self.metadata = metadata


    def _theta_ensemble(self, embeddings, top_k_indices, m, candidate_labels, test_data, similarities, seed, temperature=0.1):

        assert m <= len(top_k_indices), "Error: m must less than k"

        all_combinations = list(combinations(top_k_indices, m))
        random.seed(seed)
        random.shuffle(all_combinations)

        for idx in range(len(embeddings)):
            embeddings[idx] = torch.tensor(embeddings[idx], dtype=torch.float32)
            embeddings[idx] = embeddings[idx] / torch.norm(embeddings[idx])

        all_loss_list = []

        cnt_samples = int(len(all_combinations)/2)

        for idx, combination in enumerate(all_combinations):
            if idx >=cnt_samples: break
            selected_embeddings = [embeddings[i] for i in combination]
            selected_data = [test_data[i] for i in combination]

            # compute simloss
            simloss = 0.0
            for idx in combination:
                simloss+=similarities[idx]
            simloss /= m
            
            # compute supcon loss
            con_loss, pos_loss, all_loss = 0.0, 0.0, 0.0
            for idx1 in combination:
                cnt_pos = 0
                pos_loss = 0.0
                idx1_embedding, idx1_label = embeddings[idx1], candidate_labels[idx1]
                logits = []
                for idx2 in combination:
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

            current_labels = [candidate_labels[i] for i in combination]
            lam=0.05
            simcon_loss = -simloss + lam*con_loss
            all_loss_list.append(simcon_loss)

        X = np.zeros((cnt_samples, len(top_k_indices)))
        y = np.zeros(cnt_samples)
        model = LinearRegression()

        for i, combination in enumerate(all_combinations):
            if i >= cnt_samples: break
            y[i] = all_loss_list[i]
            for j, item in enumerate(top_k_indices):
                if item in combination:
                    X[i, j] = 1
        
        model.fit(X,y)
        theta = model.coef_

        selected_indices = set()
        remaining_indices = set(range(len(top_k_indices)))

        while len(selected_indices) < m:
            best_candidate = None
            best_loss = float('inf')

            for idx in remaining_indices:
                if idx in selected_indices:
                    continue

                temp_selected = selected_indices.union({idx})
                temp_loss = 0.0

                for i in temp_selected:
                    temp_loss += theta[i]

                if temp_loss < best_loss:
                    best_loss = temp_loss
                    best_candidate = idx

            selected_indices.add(best_candidate)
            remaining_indices.remove(best_candidate)

        real_id = [top_k_indices[idx] for idx in selected_indices]

        return [test_data[idx] for idx in real_id]
        
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


    def tensorize_forsel(self, _test_data, _val_data, m, seed, options=None, add_newlines=True):
        if options is not None:
            for i, dp in enumerate(_test_data):
                assert "options" not in dp
                assert type(dp) == str
                _test_data[i] = {"input": dp, "options": options}
            for i, dp in enumerate(_val_data):
                assert "options" not in dp
                assert type(dp) == str
                _val_data[i] = {"input": dp, "options": options}
        print("len(_test_data) : ",len(_test_data))
        print("len(_val_data) : ", len(_val_data))

        val_data, test_data =  [], []

        for dp in _test_data:
            assert type(dp) == dict, ("Each example should be a dictionary", dp)
            assert "input" in dp and "options" in dp and type(dp["options"]) == list, \
                ("Test example should contain input and options in a list format", dp)
            if "output" not in dp:
                dp["output"] = dp["options"][0]  # randomly choose one (we don't need it anyways)
            test_data.append(dp.copy())
        for dp in _val_data:
            assert type(dp) == dict, ("Each example should be a dictionary", dp)
            assert "input" in dp and "options" in dp and type(dp["options"]) == list, \
                ("Test example should contain input and options in a list format", dp)
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

        if self.use_demonstrations:
            test_texts = [dp["input"] + " " + dp["output"] for dp in test_data]
            test_labels = [dp["output"] for dp in test_data]

        input_ids, attention_mask, token_type_ids = [], [], []
        metadata = []

        for dp_idx, dp in enumerate(val_data):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)

            if self.use_demonstrations:
                test_text = dp["input"]
                dp_feature = val_features[dp_idx]            

                top_k_neighbors, top_k_indices, similarities = self._select_top_k_neighbors(
                    dp_feature, test_features, test_data, self.k, dp_idx
                )
                
                # print("similarities : ",similarities)

                forsel, _ = self._forward_selection(
                    embeddings=test_features,
                    top_k_indices=top_k_indices,
                    m=m, 
                    candidate_labels=test_labels, 
                    test_data=test_data,
                    similarities = similarities,
                    seed=seed
                )

                demonstrations = []
                for i, neighbor_dp in enumerate(forsel):
                    input_, output_ = self._prepro_each_datapoint(
                        neighbor_dp, is_first=i == 0, for_demonstrations=True, add_newlines=add_newlines)
                    demonstrations += input_ + output_

            indices = [[i] for i in range(len(input_ids), len(input_ids) + len(inputs))]

            metadata.append({"indices": indices, "answer": answer, "options": dp["options"]})

            for inputs_, outputs_ in zip(inputs, outputs):
                if self.use_demonstrations:
                    inputs_ = demonstrations + inputs_
                encoded = prepro_sentence_pair_single(
                    inputs_, outputs_, self.max_length, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id,
                    allow_truncation=self.use_demonstrations
                )
                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])

        self.tensorized_inputs = dict(input_ids=torch.LongTensor(input_ids),
                                      attention_mask=torch.LongTensor(attention_mask),
                                      token_type_ids=torch.LongTensor(token_type_ids))
        self.metadata = metadata

    def synonym_augmentation(sentence, num_replacements=1):
        words = sentence.split()
        augmented_sentence = words.copy()
        for _ in range(num_replacements):
            word_to_replace = random.choice(words)
            synonyms = wordnet.synsets(word_to_replace)
            if synonyms:
                synonym_words = [lemma.name() for syn in synonyms for lemma in syn.lemmas()]
                if synonym_words:
                    synonym = random.choice(synonym_words)
                    augmented_sentence[words.index(word_to_replace)] = synonym.replace("_", " ")
        return " ".join(augmented_sentence)

    def _select_random_k_neighbors(self, test_sample_embedding, test_embeddings, test_data, k, dp_idx):
        
        length = len(test_data)
        candidates = [i for i in range(length) if i!= dp_idx]
        random_indices = random.sample(candidates, k)

        return [test_data[i] for i in random_indices]
    
    def tensorize_randomk(self, _test_data, options=None, add_newlines=True):
        if options is not None:
            for i, dp in enumerate(_test_data):
                assert "options" not in dp
                assert type(dp) == str
                _test_data[i] = {"input": dp, "options": options}

        print(("-"*20))
        print(f"len(_test_data): {len(_test_data)}")
        train_data, test_data = [], []


        for dp in _test_data:
            assert type(dp) == dict, ("Each example should be a dictionary", dp)
            assert "input" in dp and "options" in dp and type(dp["options"]) == list, \
                ("Test example should contain input and options in a list format", dp)
            if "output" not in dp:
                dp["output"] = dp["options"][0]  
            test_data.append(dp.copy())

        print("-"*20)
        print(f"len(test_data) : {len(test_data)}")

        if self.use_demonstrations:
            test_texts = [dp["input"] + " " + dp["output"] for dp in test_data]
            test_embeddings = [
                self.tokenizer.encode(text, add_special_tokens=False) for text in test_texts
            ]
            print(len(test_embeddings[0]), len(test_embeddings[1]), len(test_embeddings[2]))
            test_embeddings_pad=[]
            max_length=self.max_length_per_example
            for i,embedding in enumerate(test_embeddings):
                if len(embedding) > max_length:
                    test_embeddings_pad.append(embedding[:max_length])
                else:
                    test_embeddings_pad.append(embedding + [0] * (max_length - len(embedding)))
            # train_embeddings = np.array(train_embeddings)
            print(len(test_embeddings_pad[0]), len(test_embeddings_pad[1]), len(test_embeddings_pad[2]))

        input_ids, attention_mask, token_type_ids = [], [], []
        metadata = []

        # print("test_data : ",test_data)

        for dp_idx, dp in enumerate(test_data):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)

            if self.use_demonstrations:
                test_text = dp["input"]
                test_embedding = test_embeddings_pad[dp_idx]            

                randomk_neighbors = self._select_random_k_neighbors(
                    test_embedding, test_embeddings_pad, test_data, self.k, dp_idx
                )
                demonstrations = []
                for i, neighbor_dp in enumerate(randomk_neighbors):
                    input_, output_ = self._prepro_each_datapoint(
                        neighbor_dp, is_first=i == 0, for_demonstrations=True, add_newlines=add_newlines)
                    demonstrations += input_ + output_

            indices = [[i] for i in range(len(input_ids), len(input_ids) + len(inputs))]

            metadata.append({"indices": indices, "answer": answer, "options": dp["options"]})

            for inputs_, outputs_ in zip(inputs, outputs):
                if self.use_demonstrations:
                    inputs_ = demonstrations + inputs_
                encoded = prepro_sentence_pair_single(
                    inputs_, outputs_, self.max_length, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id,
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

    def tensorize_multidata(self, _test_data, datapath, m, options=None, add_newlines=True):
        if options is not None:
            for i, dp in enumerate(_test_data):
                assert "options" not in dp
                assert type(dp) == str
                _test_data[i] = {"input": dp, "options": options}

        train_data, test_data =  [], []

        for dp in _test_data:
            assert type(dp) == dict, ("Each example should be a dictionary", dp)
            assert "input" in dp and "options" in dp and type(dp["options"]) == list, \
                ("Test example should contain input and options in a list format", dp)
            if "output" not in dp:
                dp["output"] = dp["options"][0]  # randomly choose one (we don't need it anyways)
            test_data.append(dp.copy())
            task = dp["task"]

        if self.use_demonstrations:
            test_texts = [dp["input"] + " " + dp["output"] for dp in test_data]
            test_embeddings = [
                self.tokenizer.encode(text, add_special_tokens=False) for text in test_texts
            ]
            print(len(test_embeddings[0]), len(test_embeddings[1]), len(test_embeddings[2]))
            test_embeddings_pad=[]
            max_length=self.max_length_per_example
            for i,embedding in enumerate(test_embeddings):
                if len(embedding) > max_length:
                    test_embeddings_pad.append(embedding[:max_length])
                else:
                    test_embeddings_pad.append(embedding + [0] * (max_length - len(embedding)))
            # train_embeddings = np.array(train_embeddings)
            print(len(test_embeddings_pad[0]), len(test_embeddings_pad[1]), len(test_embeddings_pad[2]))

        input_ids, attention_mask, token_type_ids = [], [], []
        metadata = []

        # print("test_data : ",test_data)

        for dp_idx, dp in enumerate(test_data):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)

            if self.use_demonstrations:
                test_text = dp["input"]
                test_embedding = test_embeddings_pad[dp_idx]            

                random_source = self._random_datasource(
                    task, datapath, m
                )

                demonstrations = []
                for i, neighbor_dp in enumerate(random_source):
                    input_, output_ = self._prepro_each_datapoint(
                        neighbor_dp, is_first=i == 0, for_demonstrations=True, add_newlines=add_newlines)
                    demonstrations += input_ + output_

            indices = [[i] for i in range(len(input_ids), len(input_ids) + len(inputs))]

            metadata.append({"indices": indices, "answer": answer, "options": dp["options"]})

            for inputs_, outputs_ in zip(inputs, outputs):
                if self.use_demonstrations:
                    inputs_ = demonstrations + inputs_
                encoded = prepro_sentence_pair_single(
                    inputs_, outputs_, self.max_length, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id,
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
                    inputs_, outputs_, self.max_length, bos_token_id, eos_token_id,
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
                                bos_token_id, eos_token_id,
                                allow_truncation=False):

    if allow_truncation and len(ids1)+len(ids2) > max_length:
        ids1 = ids1[len(ids1)+len(ids2)-max_length:] # len = max_length-len(ids2)
        assert len(ids1)+len(ids2)==max_length

    n_mask = max_length-len(ids1)-len(ids2)
    assert n_mask>=0, (max_length, len(ids1), len(ids2))
    input_ids = ids1+ids2+[0 for _ in range(n_mask)]
    attention_mask = [1 for _ in ids1+ids2] + [0 for _ in range(n_mask)]
    token_type_ids = [0 for _ in ids1] + [1 for _ in ids2] + [0 for _ in range(n_mask)]
    return input_ids, attention_mask, token_type_ids

def prepro_sentence_pair(train_inputs, test_inputs, max_length,
                         bos_token_id, eos_token_id,
                         allow_truncation=False):
    input_ids, attention_mask, token_type_ids = [], [], []
    for test_input in test_inputs:
        for train_input in train_inputs:
            _input_ids, _attention_mask, _token_type_ids = \
                prepro_sentence_pair_single(train_input, test_input, max_length,
                                            bos_token_id, eos_token_id,
                                            allow_truncation=allow_truncation)
            input_ids.append(_input_ids)
            attention_mask.append(_attention_mask)
            token_type_ids.append(_token_type_ids)

    return {"input_ids": torch.LongTensor(input_ids),
            "attention_mask": torch.LongTensor(attention_mask),
            "token_type_ids": torch.LongTensor(token_type_ids)}

