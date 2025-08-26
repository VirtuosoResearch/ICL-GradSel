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
from transformers import modelLMHeadModel, modelTokenizer, OPTForCausalLM
from tqdm import tqdm
from thop import profile

from rank_bm25 import BM25Okapi
logging.getLogger("thop").setLevel(logging.WARNING)

import random

from collections import defaultdict
from functools import partial
from multiprocessing import Pool

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from sklearn.linear_model import LinearRegression


from gradsel.model import gradselModel
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

class gradselData(object):

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


        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("model")

    def __len__(self):
        if self.tensorized_inputs is None:
            return 0
        return len(self.tensorized_inputs["input_ids"])

    def __str__(self):
        text = "[gradsel Data]: method=%d, "
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

    def forward(self, model, gradsel_model, demonstrations, dp, task, return_loss = False):
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
            results, _ = gradsel_model.run_model(input_ids, attention_mask, token_type_ids)

            if self.is_flops:
                self.logger.info(f"len(input_ids): {input_ids.size()}")
                flops, params = profile(gradsel_model.model, inputs=(input_ids,))
            else: flops =0

            return input_ids, results.cpu().detach().item(), flops

        option_tokens = [tokenizer(option)["input_ids"] for option in dp['options']]
        input_tokens = tokenizer(dp["input"] + " ")["input_ids"]
        gradsel_model.model.eval()
        # gradsel_model.model.to(device)

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
    
    def random_ensemble(self, model, gradsel_model, test_data, dev_data,
                        num_combinations=100, k=8, seed=42, num_anchors=None):
        from collections import defaultdict
        import random

        random.seed(seed)
        device = torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")

        all_indices = list(range(len(test_data)))

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

        anchors = sampled_combinations if (num_anchors is None or num_anchors >= len(sampled_combinations)) \
            else sampled_combinations[:num_anchors]
        self.logger.info(f"number of anchors (using all sampled when None/large): {len(anchors)}")

        anchor_info = {}
        total_flops = 0

        for anchor in tqdm(anchors, desc="Computing anchor info"):
            anchor_prompt = "".join([
                f"Input: {test_data[idx]['input']} Label: {test_data[idx]['output']}\n" for idx in anchor
            ])
            base_losses, base_gradients, _, flops = zip(*[
                self.forward_estim(model, gradsel_model, anchor_prompt, dp, dp["task"], return_loss=True)
                for dp in dev_data
            ])
            total_flops += sum(flops)

            loss_tensor = torch.tensor(base_losses, device=device)  # [len(dev), num_labels]
            grad_tensor = torch.stack([torch.stack(g, dim=0) for g in base_gradients], dim=0)  # [len(dev), num_labels, D]
            anchor_info[anchor] = (anchor_prompt, loss_tensor, grad_tensor)


        point_scores = defaultdict(list)
        all_accs = []

        for anchor in tqdm(anchors, desc="Evaluating all combos per anchor"):
            anchor_prompt, base_loss_tensor, grad_tensor = anchor_info[anchor]

            for comb in sampled_combinations:
                target_prompt = "".join([
                    f"Input: {test_data[idx]['input']} Label: {test_data[idx]['output']}\n" for idx in comb
                ])

                correct = 0
                for dp_idx, dp in enumerate(dev_data):
                    dev_str = f"Input: {dp['input']} Label:"
                    delta_P = self.compute_embedding_difference_(
                        model, gradsel_model, anchor_prompt + dev_str, target_prompt + dev_str
                    )

                    taylor_losses = []
                    for j in range(len(base_loss_tensor[dp_idx])):
                        correction = torch.sum(grad_tensor[dp_idx][j] * delta_P).item()
                        approx_loss = base_loss_tensor[dp_idx][j].item() + correction
                        taylor_losses.append(approx_loss)

                    pred_id = int(np.argmin(taylor_losses))
                    pred = dp["options"][pred_id]
                    if pred == dp["output"]:
                        correct += 1

                acc = correct / len(dev_data)
                all_accs.append(acc)

                for idx in comb:
                    point_scores[idx].append(acc)

        avg_scores = []
        for idx, scores in point_scores.items():
            avg_scores.append((idx, float(sum(scores) / len(scores))))

        avg_scores.sort(key=lambda x: -x[1])
        final_indices = [idx for idx, _ in avg_scores[:k]]
        selected_data = [test_data[i] for i in final_indices]

        mean_acc_overall = float(np.mean(all_accs)) if all_accs else 0.0

        return selected_data, mean_acc_overall, total_flops


    def random_ensemble_nearest_anchor(self, model, gradsel_model, candidate_data, dev_data, num_combinations=100, k=8, seed=42, num_anchors=1):
        random.seed(seed)
        device = torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")

        self.logger.info(f"number of anchors: {num_anchors}")
        all_indices = list(range(len(candidate_data)))

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

        anchor_combs = random.sample(sampled_combinations, num_anchors)
        anchor_info = {}

        total_flops = 0
        for anchor in tqdm(anchor_combs):
            anchor_prompt = "".join([
                f"Input: {candidate_data[idx]['input']} Label: {candidate_data[idx]['output']}\n" for idx in anchor
            ])
            base_losses, base_gradients, _, flops = zip(*[
                self.forward_estim(model, gradsel_model, anchor_prompt, dp, dp["task"], return_loss=True)
                for dp in dev_data
            ])
            total_flops += sum(flops)

            loss_tensor = torch.tensor(base_losses, device=device)
            grad_tensor = torch.stack([torch.stack(g, dim=0) for g in base_gradients], dim=0)  # [len(dev), num_labels, D]
            anchor_info[anchor] = (anchor_prompt, loss_tensor, grad_tensor)

        accuracy_results = []
        for comb in tqdm(sampled_combinations, total=len(sampled_combinations)):
            if comb in anchor_combs:
                continue

            best_anchor = max(anchor_combs, key=lambda a: len(set(a) & set(comb)))
            anchor_prompt, base_loss_tensor, grad_tensor = anchor_info[best_anchor]

            target_prompt = "".join([
                f"Input: {candidate_data[idx]['input']} Label: {candidate_data[idx]['output']}\n" for idx in comb
            ])

            correct = 0
            for dp_idx, dp in enumerate(dev_data):
                dev_str = f"Input: {dp['input']} Label:"
                delta_P = self.compute_embedding_difference_(
                    model, gradsel_model, anchor_prompt + dev_str, target_prompt + dev_str
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

        point_scores = defaultdict(list)
        for comb, acc in accuracy_results:
            for idx in comb:
                point_scores[idx].append(acc)

        avg_scores = []
        for idx, scores in point_scores.items():
            avg_scores.append((idx, sum(scores) / len(scores)))

        avg_scores.sort(key=lambda x: -x[1])
        final_indices = [idx for idx, _ in avg_scores[:k]]
        selected_data = [candidate_data[i] for i in final_indices]

        return selected_data, accuracy_results[0][1], total_flops


    def tensorize_bm25(self, _test_data, _val_data, options=None, add_newlines=True):
        if options is not None:
            for i, dp in enumerate(_test_data):
                _test_data[i] = {"input": dp, "options": options}
            for i, dp in enumerate(_val_data):
                _val_data[i] = {"input": dp, "options": options}

        val_data, candidate_data = [], []
        for dp in _test_data:
            if "output" not in dp:
                dp["output"] = dp["options"][0]
            candidate_data.append(dp.copy())
        for dp in _val_data:
            if "output" not in dp:
                dp["output"] = dp["options"][0]
            val_data.append(dp.copy())

        task = _test_data[0]["task"]
        input_ids, attention_mask, token_type_ids = [], [], []
        metadata = []

        test_inputs = [dp["input"].split() for dp in candidate_data]
        bm25 = BM25Okapi(test_inputs)

        instructions = f"Here are {len(candidate_data[0]['options'])} options: "
        for option in candidate_data[0]["options"]:
            instructions += option + ", "
        instructions += f"You should choose one of them to answer at the end. \nHere are {self.k} samples for your reference. \n"
        init_tokens = self.tokenizer(instructions)["input_ids"][1:]

        for dp_idx, dp in enumerate(val_data):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)

            if self.use_demonstrations:
                top_k_neighbors, _, __ = self._select_top_k_neighbors_bm25(dp["input"], candidate_data, self.k)
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


    def evaluate_accuracy(self, model, gradsel_model, demonstrations, dev_data, task):
        correct = 0; total = len(dev_data)
        input_str = ""
        for item in demonstrations:
            input_str = input_str + item["input"] + " "+ "Label: "+item["output"]+"\n"
        input_token = self.tokenizer(input_str)["input_ids"]
        total_flops=0
        for idx, dp in enumerate(dev_data):
            _, label, flops = self.forward(model, gradsel_model, input_token, dp, task)
            if self.is_flops: self.logger.info(f"----- flops : {flops / 1e9:.2f} GFLOPs")
            total_flops+=flops
            if label == dp["output"]: correct += 1
        return correct / total if total > 0 else 0 , total_flops

    def evaluate_loss(self, model, gradsel_model, demonstrations, dev_data, task):
        total = len(dev_data)
        input_str = ""
        all_loss = 0.0
        for item in demonstrations:
            input_str = input_str + item["input"] + " "+ "Label: "+item["output"]+"\n"
        input_token = self.tokenizer(input_str)["input_ids"]
        total_flops =0
        for idx, dp in enumerate(dev_data):
            loss, flops = self.forward(model, gradsel_model, input_token, dp, task, return_loss=True)
            total_flops+=flops
            if self.is_flops: self.logger.info(f"----- flops : {flops / 1e9:.2f} GFLOPs")
            all_loss+=loss
        return all_loss, total_flops

    def greedy_select_condition(self, model, gradsel_model, candidate_data, dev_data, subset_size=10):
            loss_index_pairs = []
            total_flops = 0

            for i in range(len(candidate_data)):
                demonstrations = [candidate_data[i]]
                loss, flops = self.evaluate_loss(model, gradsel_model, demonstrations, dev_data, candidate_data[0]["task"])
                self.logger.info(f"Index {i} - Loss: {loss}")
                total_flops += flops
                loss_index_pairs.append((loss, i))

            topk = sorted(loss_index_pairs, key=lambda x: x[0])[:self.k]
            selected_indices = [i for _, i in topk]
            best_demonstrations = [candidate_data[i] for i in selected_indices]
            best_loss = sum([l for l, _ in topk]) / self.k

            self.logger.info("== Selected Indices ==")
            self.logger.info(selected_indices)

            return best_demonstrations, best_loss, total_flops

    def greedy_select_condition_estim(self, model, gradsel_model, candidate_data, dev_data):
        def build_text(prefix_text, base_sample, query_sample, op):
            return prefix_text + base_sample["input"] + base_sample["output"] + query_sample["input"] + op + "\n"
        
        self.options = candidate_data[0]["options"]
        prompt_text = ""
        device = torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")
        max_token_len = max(
            len(self.tokenizer(build_text("", ex, dev_data[0], self.options[0]), return_tensors="pt")["input_ids"][0])
            for ex in candidate_data
        )

        approx_loss_list = [0.0 for _ in range(len(candidate_data))]
        total_flops=0
        for query in dev_data:
            loss_option_dict, gradient_option_dict = {}, {}

            for op in self.options:
                base_text = build_text(prompt_text, candidate_data[0], query, op)
                loss_op, embedding_grad_op, flops = _get_embedding_loss(
                    model=gradsel_model, tokenizer=self.tokenizer, input_texts=[base_text], pad_to_length=max_token_len, is_flops=self.is_flops
                )
                loss_option_dict[op] = loss_op
                gradient_option_dict[op] = embedding_grad_op
                total_flops += flops

            for i, candidate_sample in enumerate(candidate_data):
                candidate_approx_loss_dict = {}
                for op in self.options:
                    candidate_text = build_text(prompt_text, candidate_sample, query, op)
                    delta_P = self.compute_embedding_difference(
                        model, gradsel_model, base_str=base_text, candidate_str=candidate_text, pad_to_length=max_token_len
                    )
                    taylor_correction = torch.sum(gradient_option_dict[op] * delta_P).item()
                    taylor_approx_loss = loss_option_dict[op] + taylor_correction
                    candidate_approx_loss_dict[op] = taylor_approx_loss

                approx_loss_list[i] += candidate_approx_loss_dict[query["output"]].cpu().item()
        
        for idx, loss in enumerate(approx_loss_list):
            self.logger.info(f"Index {idx} - Loss: {loss}")
        topk_indices = sorted(range(len(candidate_data)), key=lambda i: approx_loss_list[i])[:self.k]

        self.logger.info(f"selected indices: {topk_indices}")
        best_demonstrations = [candidate_data[i] for i in topk_indices]

        return best_demonstrations, None, total_flops


    def tensorize_ground(self, model, _test_data, _val_data, estimate=False, options=None, add_newlines=True):
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

        val_data, dev_data, candidate_data = [], [], []
        for idx, dp in enumerate(_test_data):
            if "output" not in dp: dp["output"] = dp["options"][0]
            if idx > len(_test_data)-len(_test_data)//4: dev_data.append(dp.copy())
            else: candidate_data.append(dp.copy())
        for dp in _val_data:
            if "output" not in dp: dp["output"] = dp["options"][0]
            val_data.append(dp.copy())
        task = _test_data[0]["task"]
        with open(f"./features/{task}_test_features.json", "r") as file: test_features = json.load(file)
        with open(f"./features/{task}_val_features.json", "r") as file: val_features = json.load(file)

        add_newlines = False
        checkpoint = None
        gradsel_model = gradselModel(logger=self.logger, out_dir= "./cache", device_num=self.device)
        gradsel_model.load(checkpoint, model=model)
        if "Llama" in model:
            gradsel_model.resize(self.tokenizer)

        input_ids, attention_mask, token_type_ids = [], [], []
        metadata = []
        total_flops = 0
        for dp_idx, dp in tqdm(enumerate(val_data), total=len(val_data)):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)
            if self.use_demonstrations:
                test_text = dp["input"]
                dp_feature = val_features[dp_idx]

                samples, top_indices, _ = self._select_top_k_neighbors(dp_feature, test_features, candidate_data, k=20,dp_idx=-1)

                if estimate==False:
                    ground, _, flops = self.greedy_select_condition(model=model, gradsel_model=gradsel_model,candidate_data=samples, dev_data=dev_data, subset_size=self.k)
                else:
                    ground, _, flops = self.greedy_select_condition_estim(model=model, gradsel_model=gradsel_model,candidate_data=samples, dev_data=dev_data)

                total_flops+=flops
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

    def compute_loss_and_gradient(self, model, gradsel_model, tokenizer, input_tokens, output_tokens, device):

        tokenizer.pad_token = tokenizer.eos_token
        input_ids = tokenizer(input_tokens, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length).input_ids.to(device)
        output_ids = tokenizer(output_tokens, return_tensors="pt")["input_ids"][0][-1].to(device)

        with torch.no_grad():
            if "gpt" in model:
                embedding = gradsel_model.model.transformer.wte(input_ids)
            elif "opt" in model: embedding = gradsel_model.model.model.decoder.embed_tokens(input_ids)
            else:
                embedding = gradsel_model.model.model.embed_tokens(input_ids)
        embedding.requires_grad = True 
        embedding = embedding.to(gradsel_model.model.dtype)

        output_logits = gradsel_model.model(inputs_embeds=embedding).logits
        last_token_idx = input_ids.shape[1] - 1 
        log_probs = F.log_softmax(output_logits[0, last_token_idx, :], dim=-1) 

        target_token = output_ids
        loss = -log_probs[target_token]
        
        flops = 0
        if self.is_flops:
            flops, params = profile(gradsel_model.model, inputs=(input_ids,))
            self.logger.info(f"----- flops : {flops / 1e9:.2f} GFLOPs")
        
        loss.backward()
        return loss.item(), embedding.grad, flops

    def compute_loss_and_gradient_op(self, model, gradsel_model, tokenizer, input_tokens, output_tokens, device):

        tokenizer.pad_token = tokenizer.eos_token

        input_ids = tokenizer(input_tokens, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length).input_ids.to(device)
        output_ids = tokenizer(output_tokens, return_tensors="pt")["input_ids"][0][-1].to(device)

        with torch.no_grad():
            if "gpt" in model:
                embedding = gradsel_model.model.transformer.wte(input_ids)
            elif "opt" in model: embedding = gradsel_model.model.model.decoder.embed_tokens(input_ids)
            else:
                embedding = gradsel_model.model.model.embed_tokens(input_ids)
        embedding.requires_grad = True 
        embedding = embedding.to(gradsel_model.model.dtype)

        output_logits = gradsel_model.model(inputs_embeds=embedding).logits
        last_token_idx = input_ids.shape[1] - 1 
        log_probs = F.log_softmax(output_logits[0, last_token_idx, :], dim=-1) 

        target_token = output_ids
        loss = -log_probs[target_token]
        
        if self.is_flops: 
            flops, params = profile(gradsel_model.model, inputs=(input_ids,))
            self.logger.info(f"----- flops : {flops / 1e9:.2f} GFLOPs")
        
        loss.backward()
        return loss.item(), embedding.grad, flops

    def forward_estim(self, model, gradsel_model, demonstrations, dp, task, return_loss=False):

        logger = logging.getLogger(__name__)
        device = torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")
        tokenizer = self.tokenizer

        option_tokens = dp['options']
        input_tokens = demonstrations + dp["input"] + " Label:"

        losses = []
        gradients = []
        total_flops =0
        for option in option_tokens:
            loss, grad, flops = self.compute_loss_and_gradient(model, gradsel_model, tokenizer, input_tokens, option, device)
            # compute_loss_and_gradient(self, model, model, tokenizer, input_tokens, output_tokens, device):
            losses.append(loss)
            gradients.append(grad)
            total_flops+=flops

        label_id = np.argmin(losses)
        label = dp["options"][label_id]

        if return_loss:
            return losses, gradients, label_id, total_flops
        return label_id, label, total_flops
    
    def compute_embedding_difference(self, model, gradsel_model, base_str, candidate_str, pad_to_length):
        device = torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")
        tokenizer = self.tokenizer

        input_tokens_1 = tokenizer(base_str, return_tensors="pt", padding="max_length", truncation=True, max_length=pad_to_length)["input_ids"].to(device)
        input_tokens_2 = tokenizer(candidate_str, return_tensors="pt", padding="max_length", truncation=True, max_length=pad_to_length)["input_ids"].to(device)

        with torch.no_grad():
            if "gpt" in model:
                embedding_1 = gradsel_model.model.transformer.wte(input_tokens_1)
                embedding_2 = gradsel_model.model.transformer.wte(input_tokens_2)
            elif "opt" in model:
                embedding_1 = gradsel_model.model.model.decoder.embed_tokens(input_tokens_1)
                embedding_2 = gradsel_model.model.model.decoder.embed_tokens(input_tokens_2)
            else:
                embedding_1 = gradsel_model.model.model.embed_tokens(input_tokens_1)
                embedding_2 = gradsel_model.model.model.embed_tokens(input_tokens_2)
            
        embedding_1 = embedding_1.to(gradsel_model.model.dtype)
        embedding_2 = embedding_2.to(gradsel_model.model.dtype)

        delta_P = embedding_2 - embedding_1.detach()

        delta_P_effective = delta_P[:, :-1, :]
        return delta_P_effective


    def compute_embedding_difference_(self, model, gradsel_model, base_str, candidate_str):
        device = torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")
        input_tokens_1 = self.tokenizer(base_str, return_tensors="pt", truncation=True, max_length=self.max_length)["input_ids"].to(device)
        input_tokens_2 = self.tokenizer(candidate_str, return_tensors="pt", truncation=True, max_length=self.max_length)["input_ids"].to(device)

        tokenizer = self.tokenizer
        max_len = self.max_length
        input_tokens_1 = torch.nn.functional.pad(input_tokens_1, (0, max_len - input_tokens_1.size(1)), value=tokenizer.pad_token_id)
        input_tokens_2 = torch.nn.functional.pad(input_tokens_2, (0, max_len - input_tokens_2.size(1)), value=tokenizer.pad_token_id)


        with torch.no_grad():
            if "gpt" in model:
                embedding_1 = gradsel_model.model.transformer.wte(input_tokens_1)
                embedding_2 = gradsel_model.model.transformer.wte(input_tokens_2)
            elif "opt" in model:
                embedding_1 = gradsel_model.model.model.decoder.embed_tokens(input_tokens_1)
                embedding_2 = gradsel_model.model.model.decoder.embed_tokens(input_tokens_2)
            else:
                embedding_1 = gradsel_model.model.model.embed_tokens(input_tokens_1)
                embedding_2 = gradsel_model.model.model.embed_tokens(input_tokens_2)
        
        embedding_1 = embedding_1.to(gradsel_model.model.dtype)
        embedding_2 = embedding_2.to(gradsel_model.model.dtype)

        delta_P = embedding_2 - embedding_1.detach()


        return delta_P

    def greedy_select_subset(self, model, gradsel_model, candidate_data, dev_data, true_step=0):
        def get_length(example, prompt_text, options):
            return max(len(prompt_text + example["input"] + op + "\n") for op in options)

        def get_max_tokenized_length(tokenizer, candidate_data, prompt_text, options):
            max_len = 0
            for example in candidate_data:
                for op in options:
                    full_text = prompt_text + example["input"] + op + "\n"
                    input_ids = tokenizer(full_text, return_tensors="pt", truncation=False)["input_ids"]
                    max_len = max(max_len, input_ids.size(1))
            return max_len

        def build_text(prefix_text, base_sample, query_sample, op):
            return prefix_text + base_sample["input"] + base_sample["output"] + query_sample["input"] + op + "\n"

        self.options = candidate_data[0]["options"]
        selected_indices, best_demonstrations = [], []
        device = torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")
        total_flops = 0
        prompt_text = ""

        while len(selected_indices) < self.k:
            step = len(selected_indices)
            max_token_len = get_max_tokenized_length(self.tokenizer, candidate_data, prompt_text, self.options)

            base_index = max(
                (i for i in range(len(candidate_data)) if i not in selected_indices),
                key=lambda i: get_length(candidate_data[i], prompt_text, self.options)
            )
            base_example = candidate_data[base_index]
            base_text_option_dict = {}
            approx_acc_list = [0 for _ in range(len(candidate_data))]
            approx_loss_list = [0 for _ in range(len(candidate_data))]

            for query_idx, query in enumerate(dev_data):
                loss_option_dict, gradient_option_dict = {}, {}

                for op in self.options:
                    base_text = build_text(prompt_text, base_example, query, op)
                    base_text_option_dict[op] = base_text

                    loss_op, embedding_grad_op, flops = _get_embedding_loss(
                        model=gradsel_model, tokenizer=self.tokenizer,
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

                for i, candidate_sample in enumerate(candidate_data):
                    if i in selected_indices:
                        continue
                    if step < true_step:
                        # --------- True inference path ---------
                        candidate_loss_option_dict = {}
                        for op in self.options:
                            candidate_text = build_text(prompt_text, candidate_sample, query,op)
                            loss_op, _, flops = _get_embedding_loss(
                                model=gradsel_model, tokenizer=self.tokenizer,
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
                                model, gradsel_model, base_str=base_text_option_dict[op],
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
            prompt_text += candidate_data[best_candidate]["input"] + candidate_data[best_candidate]["output"] + "\n"
            best_demonstrations.append(candidate_data[best_candidate])
            best_candidate_accuracy = 0
            self.logger.info(f"Selected index {best_candidate}")

        return best_demonstrations, best_candidate_accuracy, total_flops

    def _select_top_k_neighbors(self, test_sample_embedding, test_embeddings, candidate_data, k, dp_idx):
        similarities = []
        for idx, dp in enumerate(test_embeddings):
            if idx == len(candidate_data): break
            if idx == dp_idx:
                similarities.append(-1.0)
                continue
            similarity = 1 - cosine(test_sample_embedding, dp)
            similarities.append(similarity)
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [candidate_data[i] for i in top_k_indices], top_k_indices , similarities

    def tensorize_estimate(self, model, _test_data, _val_data, is_quant, method="forsel", num_anchors=1, true_step=0, options=None, add_newlines=True):
        print("options: ", options)
        if options is not None:
            print("len(_test_data) : ", len(_test_data))
            print(_test_data[0])
            for i, dp in enumerate(_test_data):
                _test_data[i] = {"input": dp, "options": options}
            for i, dp in enumerate(_val_data):
                _val_data[i] = {"input": dp, "options": options}
        print("len(_test_data) : ",len(_test_data)," ; len(_val_data) : ", len(_val_data))

        val_data, dev_data, candidate_data = [], [], []
        for i, dp in enumerate(_test_data):
            if "output" not in dp: dp["output"] = dp["options"][0]
            if i> len(_test_data)-len(_test_data)//4:
                dev_data.append(dp.copy())
            else:
                candidate_data.append(dp.copy())
        for dp in _val_data:
            if "output" not in dp: dp["output"] = dp["options"][0]
            val_data.append(dp.copy())
        task = _test_data[0]["task"]
        
        total_flops = 0

        add_newlines = True
        checkpoint = None
        gradsel_model = gradselModel(logger=self.logger, out_dir= "./cache", device_num=self.device)
        print(f"-------------- model: {model} ------------")
        gradsel_model.load(model=model,is_quant=is_quant)

        print("model : ",model)
        print("origin type(gradsel_model) : ",type(gradsel_model.model))
        if "Llama" in model:
           gradsel_model.resize(self.tokenizer)

        input_ids, attention_mask, token_type_ids = [], [], []
        metadata = []
        if method=="forsel":
            ground, _, flops = self.greedy_select_subset(model=model, gradsel_model=gradsel_model, candidate_data=candidate_data, dev_data=dev_data, true_step=true_step)
        elif method=='ranens':
            ground, _, flops = self.random_ensemble(model=model, k=self.k, gradsel_model=gradsel_model, candidate_data=candidate_data, dev_data=dev_data, num_anchors=num_anchors)
        else:
            ground, _, flops = self.greedy_select_subset_cone
        demonstrations = []
        total_flops+= flops

        for i, neighbor_dp in enumerate(ground):
            demonstrations+=self.tokenizer(neighbor_dp["input"] + " " +neighbor_dp["output"] + "\n")["input_ids"]

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

        val_data, candidate_data =  [], []

        for dp in _test_data:
            if "output" not in dp:
                dp["output"] = dp["options"][0]  # randomly choose one (we don't need it anyways)
            candidate_data.append(dp.copy())
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
                    dp_feature, test_features, candidate_data, self.k, dp_idx
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


    def _forward_selection(self, embeddings, top_k_indices, m, candidate_labels, candidate_data, similarities, seed, temperature=0.1):
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
        return [candidate_data[idx] for idx in real_id], real_id

    def _select_random_k_neighbors(self, test_sample_embedding, test_embeddings, candidate_data, k, dp_idx):
        length = len(candidate_data)
        candidates = [i for i in range(length) if i!= dp_idx]
        random_indices = random.sample(candidates, k)

        return [candidate_data[i] for i in random_indices]
    
    def tensorize_randomk(self, _test_data, _val_data, options=None, add_newlines=True):
        if options is not None:
            for i, dp in enumerate(_test_data):
                _test_data[i] = {"input": dp, "options": options}
            for i, dp in enumerate(_val_data):
                _val_data[i] = {"input": dp, "options": options}

        val_data, candidate_data =  [], []

        for dp in _test_data:
            if "output" not in dp:
                dp["output"] = dp["options"][0]  # randomly choose one (we don't need it anyways)
            candidate_data.append(dp.copy())
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
                    dp_feature, test_features, candidate_data, self.k-self.k//4, dp_idx
                )

                top_k_neighbors, _, __ = self._select_top_k_neighbors(
                    dp_feature, test_features, candidate_data, self.k//4, dp_idx
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

        val_data, candidate_data = [], []
        for dp in _test_data:
            if "output" not in dp:
                dp["output"] = dp["options"][0]
            candidate_data.append(dp.copy())
        for dp in _val_data:
            if "output" not in dp:
                dp["output"] = dp["options"][0]
            val_data.append(dp.copy())

        task = _test_data[0]["task"]
        input_ids, attention_mask, token_type_ids = [], [], []
        metadata = []

        test_inputs = [dp["input"].split() for dp in candidate_data]
        bm25 = BM25Okapi(test_inputs)

        instructions = f"Here are {len(candidate_data[0]['options'])} options: "
        for option in candidate_data[0]["options"]:
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
                top_k_neighbors = [candidate_data[i] for i in topk_indices]

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
    
    def score_by_subset(self, model, gradsel_model, candidate_data, dev_data, subset_indices):
        print(f"Subset indices: {subset_indices}")
        integer_indices = subset_indices.nonzero(as_tuple=True)[0]
        demonstrations = [candidate_data[i] for i in integer_indices]
        acc = self.evaluate_accuracy(model, gradsel_model, demonstrations, dev_data, candidate_data[0]["task"])
        
        return acc, demonstrations

    def tensorize(self, _train_data, _test_data, options=None,
                  add_newlines=True):

        if options is not None:
            for i, dp in enumerate(_test_data):
                assert "options" not in dp
                assert type(dp)==str
                _test_data[i] = {"input": dp, "options": options}

        train_data, candidate_data = [], []

        for dp in _test_data:
            assert type(dp)==dict, ("Each example should be a dictionary", dp)
            assert "input" in dp and "options" in dp and type(dp["options"])==list, \
                ("Test example should contain input and options in a list format", dp)
            if "output" not in dp:
                dp["output"] = dp["options"][0] # randomly choose one (we don't need it anyways)
            candidate_data.append(dp.copy())

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

        for dp_idx, dp in enumerate(candidate_data):
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
    #print(tokenizer.all_special_ids)
    special_ids = set(tokenizer.all_special_ids)
    #special_ids.extend([128000, 128001])
    ids1 = [i for i in ids1 if i not in special_ids]
    ids2 = [i for i in ids2 if i not in special_ids]

    total_len = len(ids1) + len(ids2) + 2  # +2 for bos and eos

    if allow_truncation and total_len > max_length:
        overflow = total_len - max_length
        ids1 = ids1[overflow:]
        total_len = len(ids1) + len(ids2) + 2
        assert total_len == max_length

    input_ids = [bos_token_id] + ids1 + ids2 + [eos_token_id]

    n_pad = max_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * n_pad
    attention_mask = [1] * (len(input_ids) - n_pad) + [0] * n_pad
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
