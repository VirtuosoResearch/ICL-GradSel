import torch
import json
import logging
import numpy as np
from transformers import GPT2Tokenizer, AutoTokenizer
from metaicl.data import MetaICLData
from metaicl.model import MetaICLModel
from metaicl.data import prepro_sentence_pair_single
from utils.data import load_data

class Forward():
    def __init__(self, k=3, dataset="glue-rte"):
        
        self.do_zeroshot = True
        self.use_demonstrations = True
        self.use_calibration = False
        self.unseen_domain_only = False
        self.log_file = None
        self.dataset = dataset
        self.k = k
        self.seed = 42
        self.device = 0
        self.test_batch_size = 4
        self.global_step = None
        self.checkpoint = None
        self.out_dir = "out/gpt2-large"
        self.split = "test"
        self.is_null = False
        self.method = "direct"
        self.gpt2 = "gpt2-large"

    def forward(self, idx):
        logger = logging.getLogger(__name__)
        device = torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")
        if self.gpt2.startswith("gpt2"):
            tokenizer = GPT2Tokenizer.from_pretrained(self.gpt2)
        elif "Llama" in self.gpt2:
            tokenizer = AutoTokenizer.from_pretrained(self.gpt2)
        else:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")

        add_newlines = not self.gpt2.startswith("gpt2")
        checkpoint = None
        metaicl_model = MetaICLModel(logger=logger, out_dir= self.out_dir, device_num=self.device)
        metaicl_model.load(checkpoint, gpt2=self.gpt2)

        max_length_per_example, max_length = 128, 256
        if self.use_demonstrations:
            max_length = min(max_length * self.k, 1024)

        metaicl_data = MetaICLData(logger, tokenizer, self.method, self.use_demonstrations, self.k,
                                    max_length, max_length_per_example)

        config_split = "unseen_domain_test" if self.unseen_domain_only else "test"
        test_data = load_data(None, self.split, self.k, seed=self.seed, config_split=config_split,
                    datasets=None if self.dataset is None else self.dataset.split(","), is_null=self.is_null)
        task = self.dataset

        with open(f"config/tasks/{task}.json", "r") as f:
            config = json.load(f)
        with open(f"./features/{task}_features.json", "r") as f:
            test_features = json.load(f)


        def run_a_forward_pass(input_tokens, output_tokens, tokenizer):
            encoded = prepro_sentence_pair_single(
                        input_tokens, output_tokens, max_length=1024, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
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

        dp_idx = idx
        dp = test_data[dp_idx]
        option_tokens = [
            tokenizer(option)["input_ids"] for option in dp['options']
        ]
        options = [dp["options"].index(dp["output"])]

        dp = test_data[dp_idx]
        input_tokens = tokenizer("Input: " + dp["input"] + " " + "Label: ")["input_ids"]
        output_tokens = tokenizer(dp["output"])["input_ids"]
        metaicl_model.model.eval()
        metaicl_model.model.to(device)
        demonstrations = []

        one_trial_losses = []
        for option_token in option_tokens:
            input_ids, results = run_a_forward_pass(demonstrations + input_tokens, option_token, tokenizer)
            one_trial_losses.append(results)

        label_id = np.argmin(one_trial_losses)
        label = dp["options"][label_id]
        return label_id, label
