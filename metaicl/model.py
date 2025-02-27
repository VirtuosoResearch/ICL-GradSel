# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
import torch
import torch.nn.functional as F

from tqdm import tqdm
from transformers import Adafactor, AdamW, get_linear_schedule_with_warmup
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig

from utils.utils import get_checkpoint_id, download_file

class MetaICLModel(object):

    def __init__(self, device_num, logger=None, out_dir=None, fp16=True, local_rank=-1):
        if logger is None:
            class Logger():
                def info(self, text):
                    print ("Logging from MetaICLModel:\t", text)
            logger = Logger()

        self.logger = logger
        self.out_dir = out_dir
        self.fp16 = fp16
        self.local_rank = local_rank
        self.device_num = device_num

        if self.local_rank == -1:
            device = torch.device(f"cuda:{self.device_num}" if torch.cuda.is_available() else "cpu")
            n_gpu = 1
            ws = 1
        else:  # distributed mode
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{self.device_num}", local_rank)
            ws = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
            torch.distributed.init_process_group(backend="nccl")
            n_gpu = 1

        self.n_gpu = n_gpu
        self.device = device
        if self.local_rank <= 0:
            logger.info("Setting up for local_rank=%d, world_size=%d" % (self.local_rank, ws))
        self.model_name = None
        self.model = None
        self.mode = None

    def __str__(self):
        text = "[MetaICL Model]: "
        if self.model_name is None:
            text += "No model loaded yet"
        else:
            text += self.model_name
            if self.mode is None:
                text += " (no mode setted - try .train() or .eval()"
            else:
                text += " (%s mode)" % self.mode
        text += "\nusing device %s, %d gpus, local_rank=%d" % (self.device, self.n_gpu, self.local_rank)
        return ("="*50) + "\n" + text + "\n" + ("="*50)

    def is_none(self):
        return self.model is None

    def train(self):
        self.model.train()
        self.mode = "train"

    def eval(self):
        self.model.eval()
        self.mode = "eval"

    def cuda(self):
        # self.model.cuda()
        self.model.to(self.device)
    
    def resize(self, tokenizer):
        self.model.resize_token_embeddings(len(tokenizer))


    def load(self, checkpoint=None, gpt2="gpt2-large", is_quant=False):
        '''
        checkpoint can be either keyword of the model or path to the checkpoint file
        '''
        if checkpoint is not None and checkpoint.startswith("gpt"):
            gpt2 = checkpoint
            checkpoint = None
        
        if is_quant:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                llm_int8_threshold=6.0
            )

            model = AutoModelForCausalLM.from_pretrained(
                gpt2, 
                quantization_config=bnb_config,
                device_map= {"": self.device}
            )
        elif gpt2.startswith("gpt2"):
            model = AutoModelForCausalLM.from_pretrained(gpt2)
        elif "gpt-j" in gpt2:
            model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b") #/gpt2)
        elif "Llama" in gpt2:
            model = AutoModelForCausalLM.from_pretrained(gpt2)
        else:
            raise NotImplementedError(checkpoint)
        self.model_name = gpt2

        self.model = model
        if is_quant != True:
            self.model.to(self.device)


    def save(self, step):
        if self.local_rank <= 0:
            model_state_dict = {key[7:] if key.startswith("module.") else key: value.cpu()
                                for key, value in self.model.state_dict().items()}
            torch.save(model_state_dict, os.path.join(self.out_dir, "model-{}.pt".format(step)))
            self.logger.info("Saving model parameters at step=%d" % step)

    def do_inference(self, data, batch_size=1, verbose=False):
        dataloader = data.get_dataloader(batch_size, is_training=False)
        # self.logger.info(f"len(dataloader) : {len(dataloader)}")
        losses = []
        for batch in tqdm(dataloader):
            input_ids=batch[0].cuda()
            attention_mask=batch[1].cuda()
            token_type_ids=batch[2].cuda()
            if len(batch)==3:
                labels=None
            else:
                labels=batch[3].cuda()
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            
            with torch.no_grad():
                loss = self.run_model(input_ids, attention_mask, token_type_ids, labels=labels)
            losses += loss.cpu().detach().numpy().tolist()
        return losses

    def do_predict(self, data, batch_size=4, losses=None, verbose=False):
        if losses is None:
            losses = self.do_inference(data, batch_size, verbose=verbose)
        losses = np.array(losses)
        assert len(losses)==len(data)
        predictions = []
        for idx, dp in enumerate(data.metadata):
            # print("---------------------------------------")
            # print(dp)
            curr_label_losses = [np.sum(losses[indices]) for indices in dp["indices"]]
            prediction_idx = sorted(enumerate(curr_label_losses), key=lambda x: x[1])[0][0]
            prediction = dp["options"][prediction_idx]
            predictions.append(prediction.strip())
        # print("len(data.metadata) : ",len(data.metadata))
        return predictions

    def run_model(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[..., :-1, :].contiguous()

        if labels is None:
            labels = input_ids
        labels = labels[..., 1:].contiguous()
        label_mask = token_type_ids[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        losses = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1)) # [batch_size, length]

        losses = losses.view(logits.size(0), logits.size(1)) * label_mask
        return torch.sum(losses, axis=1) / torch.sum(label_mask, axis=1)




