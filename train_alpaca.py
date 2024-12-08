import argparse
import logging
import os
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytorch_lightning as pl
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer

from src.custom.alpaca_model import AlpacaModel
from src.custom.alpaca_data_module import AlpacaDataModule

import numpy as np
import time

from adapters import AutoAdapterModel, DoubleSeqBnConfig

logging.basicConfig(level=logging.INFO)
torch.set_float32_matmul_precision("high")

def train_step(args, model, loss_fn, x, y, epsilon=0.01):
    model.eval()

    lr = args.lr

    model.train()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    optimizer.zero_grad()

    outputs = model(x)
    logits = outputs.logits
    loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
    loss.backward()
    optimizer.step() 

    model.eval()
    with torch.no_grad():
        outputs = model(x)
        logits = outputs.logits
        train_loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        
    return train_loss


def main(args):
    print("arguments".upper().center(80, "-"))
    print(args)
    print("-" * 80)
    device = torch.device(f"cuda:{args.devices[0]}")
    print("device: ",device)

    model_key = args.model_key.replace("/", "-")
    if "gpt" in args.model_key or "Llama" in model_key or "bloomz" in model_key or "gemma" in model_key or "Mistral" in model_key:
        hf_key = args.model_key.replace("_", "-")
        tokenizer = AutoTokenizer.from_pretrained(hf_key)
        tokenizer.padding_side = 'right'
        model = AutoModelForCausalLM.from_pretrained(hf_key)
        model_type = "decoder"
        append_eos = True
    elif "flan" in model_key:
        hf_key = "google/{}".format(model_key.replace("_", "-"))
        model = AutoModelForSeq2SeqLM.from_pretrained(hf_key)
        tokenizer = AutoTokenizer.from_pretrained(hf_key, model_max_length=512)
        model_type = "encoder_decoder"
        append_eos = False  # T5 tokenizers already append eos
    else:
        raise NotImplementedError(args.model_key)

    if args.train_adapter:
        model = AutoAdapterModel.from_pretrained(hf_key)

        reduction = args.reduction_factor
        bottleneck_config = DoubleSeqBnConfig(
            mh_adapter=True,
            output_adapter=True,
            reduction_factor=reduction,
            non_linearity="relu"
        )

        model.add_adapter(adapter_name="seq_bn", config=bottleneck_config)

        model.to(device)

        for name, param in model.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False

        model.set_active_adapters("seq_bn")

        print("-" * 20, "Bottleneck_Adapter", "-" * 20)
        trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params_count = sum(p.numel() for p in model.parameters())

        print(f"Trainable parameters: {trainable_params_count} || All parameters: {all_params_count} || ratio: {trainable_params_count / all_params_count}")
        
        print("model.device: ",model.device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    task_idxes = list(range(38))
    data_module = AlpacaDataModule(tokenizer=tokenizer,
                                   data_path="./data/alpaca_data/alpaca_final.pkl",
                                   dev_split_path="./data/alpaca_data/alpaca_dev_split_map.pkl",
                                   task_idxes=task_idxes,
                                   batch_size=args.batch_size,
                                   inference_batch_size=args.batch_size,
                                   context_length=args.max_length,
                                   downsample=args.downsample,
                                   model_type=model_type)

    data_module.setup(stage="fit")

    # Replace Trainer with zero-order optimization
    log_dir = "./loss_result/"+args.loss_file
    with open(log_dir, "w") as log_file:
        data_loader = data_module.train_dataloader()
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(args.epochs):
            print(f"Epoch {epoch + 1}/{args.epochs}")
            cnt_batch = 0
            data_loader = tqdm(data_module.train_dataloader(), desc=f"Epoch {epoch+1}")
            for batch in data_loader:
                x, y = batch["input_ids"], batch["labels"]
                x, y = x.to(model.device), y.to(model.device)
                cnt_batch += 1
                loss = train_step(args, model, loss_fn, x, y, epsilon=0.01)
                data_loader.set_postfix(loss=loss.item())
                log_file.write(f"Epoch {epoch+1}, Batch {cnt_batch}, loss: {loss.item()}\n")
                log_file.flush()

    print("Zero-order training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", type=str, default="EleutherAI/gpt-neo-1.3B")
    parser.add_argument("--train_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1])

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--load_model_dir", type=str, default=None)

    parser.add_argument("--compute_pretrained_outputs", action="store_true")
    parser.add_argument("--downsample", type=int, default=None)
    parser.add_argument("--num_batches_gradients", type=int, default=100)
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--abs_scale", type=float, default=-1.0)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--save_name", type=str, default="none")
    parser.add_argument("--use_test", action="store_true")
    
    parser.add_argument("--train_adapter",action="store_true")
    parser.add_argument("--reduction_factor", type=int, default=128)
    parser.add_argument("--use_qadapter", action= "store_true")
    
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--loss_file",type=str,default="loss.log")
    parser.add_argument("--num_samples", type=int, default=5)

    args = parser.parse_args()
    main(args)

