# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import csv
import json
import string
import numpy as np
import pickle as pkl
import math
import torch
import random
from itertools import combinations

from collections import defaultdict
from functools import partial
from multiprocessing import Pool

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine

class MetaICLData(object):

    def __init__(self, logger=None, tokenizer=None, method="channel", use_demonstrations=True, k=16,
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
            # print("dp[\"output\"]",dp["output"])
            # print("dp[\"options\"]",dp["options"])
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

            if idx == dp_idx:
                similarities.append(-1.0)
                continue
            similarity = 1 - cosine(test_sample_embedding, dp)
            similarities.append(similarity)

        # print("k : ",k)
        # print("similarities : ",similarities)
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [test_data[i] for i in top_k_indices], top_k_indices , similarities
    
    def tensorize_topk(self, _test_data, options=None, add_newlines=True):
        if options is not None:
            for i, dp in enumerate(_test_data):
                assert "options" not in dp
                assert type(dp) == str
                _test_data[i] = {"input": dp, "options": options}
        print("len(_test_data) : ",len(_test_data))
        train_data, test_data =  [], []

        for dp in _test_data:
            assert type(dp) == dict, ("Each example should be a dictionary", dp)
            assert "input" in dp and "options" in dp and type(dp["options"]) == list, \
                ("Test example should contain input and options in a list format", dp)
            if "output" not in dp:
                dp["output"] = dp["options"][0]  # randomly choose one (we don't need it anyways)
            test_data.append(dp.copy())
        
        task = _test_data[0]["task"]
        features_path = f"./features/{task}_features.json"
        with open(features_path, "r") as file:
            test_features = json.load(file)
        
        # print("--"*20)
        # print(len(test_features))
        # print("--"*20)

        if self.use_demonstrations:
            test_texts = [dp["input"] + " " + dp["output"] for dp in test_data]

        input_ids, attention_mask, token_type_ids = [], [], []
        metadata = []

        for dp_idx, dp in enumerate(test_data):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)

            if self.use_demonstrations:
                # if dp_idx>=len(test_features):
                #     print(f"len(test_data): {len(test_data)} dp_idx: {dp_idx} len(test_features): {len(test_features)}")
                dp_feature = test_features[dp_idx]            

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
            
            # con_loss = torch.tensor(0.0, dtype=float)
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

    def tensorize_supcon(self, _test_data, m, options=None, add_newlines=True):
        if options is not None:
            for i, dp in enumerate(_test_data):
                assert "options" not in dp
                assert type(dp) == str
                _test_data[i] = {"input": dp, "options": options}

        train_data, test_data, test_labels = [], [], []


        for dp in _test_data:
            assert type(dp) == dict, ("Each example should be a dictionary", dp)
            assert "input" in dp and "options" in dp and type(dp["options"]) == list, \
                ("Test example should contain input and options in a list format", dp)
            if "output" not in dp:
                dp["output"] = dp["options"][0]  # randomly choose one (we don't need it anyways)
            test_data.append(dp.copy())


        task = _test_data[0]["task"]
        features_path = f"./features/{task}_features.json"
        with open(features_path, "r") as file:
            test_features = json.load(file)
        

        if self.use_demonstrations:
            test_texts = [dp["input"] + " " + dp["output"] for dp in test_data]
            test_labels = [dp["output"] for dp in test_data]

        input_ids, attention_mask, token_type_ids = [], [], []
        metadata = []

        for dp_idx, dp in enumerate(test_data):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)

            if self.use_demonstrations:
                test_text = dp["input"]
                dp_feature = test_features[dp_idx]            

                top_k_neighbors, top_k_indices, similarities = self._select_top_k_neighbors(
                    dp_feature, test_features, test_data, self.k, dp_idx
                )
                
                # print("similarities : ",similarities)

                greedy, best_labels = self.greedy_supcon(
                    embeddings=test_features,
                    top_k_indices=top_k_indices,
                    m=m, 
                    candidate_labels=test_labels, 
                    test_data=test_data,
                    similarities = similarities
                )

                # file_path = f"./labels/{task}_{self.k}_{m}.json"

                # if os.path.exists(file_path):
                #     with open(file_path, "r") as file:
                #         try:
                #             existing_data = json.load(file) 
                #         except json.JSONDecodeError:
                #             existing_data = [] 
                # else:
                #     existing_data = []
                # item = {idx : val for idx,val in enumerate(best_labels)}
                # existing_data.append(item)

                # with open(file_path, "w") as file:
                #     json.dump(existing_data, file, indent=4)

                # file_path = 
                # with open (f"./labels/{task}_{self.k}_{m}.json","a") as file:
                #     file.write("\n".join(map(str, best_labels)) + "\n")
                # print("-----------Greedy------------")
                # print(greedy)
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

    def _random_ensemble(self, embeddings, top_k_indices, m, candidate_labels, test_data, similarities, temperature=0.1):

        assert m <= len(top_k_indices), "Error: m must less than k"

        all_combinations = list(combinations(top_k_indices, m))
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

        point_score = [1001.0 for i in top_k_indices]
        # print("point_score : ",point_score)
        # print("len(point_score) : ",len(point_score))
        cnt_point = [0 for i in top_k_indices]
        for i, combination in enumerate(all_combinations):
            if i>=len(all_combinations)/2: break
            # print(f"combination: {combination}")
            for j, indice in enumerate(top_k_indices):
                # print(f"indice: {indice}, point_score[j] : {point_score[j]}")
                if indice in combination:
                    if point_score[j]>1000.0:
                        point_score[j] = all_loss_list[i]
                    else:
                        point_score[j]+=all_loss_list[i]
                    cnt_point[j]+=1
        for i in range(len(point_score)):
            if point_score[i] is not float("inf"):
                point_score[i]/=cnt_point[i]
        
        indexed_score = list(enumerate(point_score))
        sorted_score = sorted(indexed_score, key=lambda x: x[1])
        min_indices = [x[0] for x in sorted_score[:m]]
        real_indices = [top_k_indices[x] for x in min_indices]

        # print("-*-"*10)
        # print(f"top_k_indices : {top_k_indices}; point_score : {point_score}; cnt_point : {cnt_point}")
        # print(f"real_indices : {real_indices}")
        # print("-*-"*10)

        return [test_data[x] for x in real_indices]

     
    def tensorize_ranens(self, _test_data, m, options=None, add_newlines=True):
        if options is not None:
            for i, dp in enumerate(_test_data):
                assert "options" not in dp
                assert type(dp) == str
                _test_data[i] = {"input": dp, "options": options}

        train_data, test_data, test_labels = [], [], []


        for dp in _test_data:
            assert type(dp) == dict, ("Each example should be a dictionary", dp)
            assert "input" in dp and "options" in dp and type(dp["options"]) == list, \
                ("Test example should contain input and options in a list format", dp)
            if "output" not in dp:
                dp["output"] = dp["options"][0]  # randomly choose one (we don't need it anyways)
            test_data.append(dp.copy())


        task = _test_data[0]["task"]
        features_path = f"./features/{task}_features.json"
        with open(features_path, "r") as file:
            test_features = json.load(file)
        

        if self.use_demonstrations:
            test_texts = [dp["input"] + " " + dp["output"] for dp in test_data]
            test_labels = [dp["output"] for dp in test_data]

        input_ids, attention_mask, token_type_ids = [], [], []
        metadata = []

        for dp_idx, dp in enumerate(test_data):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)

            if self.use_demonstrations:
                test_text = dp["input"]
                dp_feature = test_features[dp_idx]            

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
                    similarities = similarities
                )

    # def _random_ensemble(self, embeddings, top_k_indices, m, candidate_labels, test_data, similarities, temperature=0.1):


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
                # print("inputs_ : ",inputs_)
                # print("outputs_ : ",outputs_)
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

    # def _select_unlabeled(self, test_data, k, dp_idx):
        
    #     length = len(test_data)
    #     candidates = [i for i in range(length) if i!= dp_idx]
    #     random_indices = random.sample(candidates, k)

    #     return [test_data[i] for i in random_indices]
    

    def tensorize_unlabeled(self, _test_data, options=None, add_newlines=True):


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
            print(len(test_embeddings_pad[0]), len(test_embeddings_pad[1]), len(test_embeddings_pad[2]))

        input_ids, attention_mask, token_type_ids = [], [], []
        metadata = []

        for dp_idx, dp in enumerate(test_data):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)

            if self.use_demonstrations:
                test_text = dp["input"]
                test_embedding = test_embeddings_pad[dp_idx]            

                top_k_neighbors, _, __ = self._select_top_k_neighbors(
                    test_embedding, test_embeddings_pad, test_data, self.k, dp_idx
                )
                demonstrations = []
                for i, neighbor_dp in enumerate(top_k_neighbors):
                    input_, output_ = self._prepro_each_datapoint(
                        neighbor_dp, is_first=i == 0, for_demonstrations=True, add_newlines=add_newlines)
                    if i<= self.k//2:
                        demonstrations += input_ + output_
                    else:
                        demonstrations += input_

            indices = [[i] for i in range(len(input_ids), len(input_ids) + len(inputs))]

            metadata.append({"indices": indices, "answer": answer, "options": dp["options"]})

            for inputs_, outputs_ in zip(inputs, outputs):
                # print("inputs_ : ",inputs_)
                # print("outputs_ : ",outputs_)
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

    #if bos_token_id is not None:
    #    ids1 = [bos_token_id] + ids1
    #if eos_token_id is not None:
    #    ids2 = ids2 + [eos_token_id]
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

