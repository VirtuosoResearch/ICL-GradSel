import argparse
import logging
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytorch_lightning as pl
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.custom.alpaca_model import AlpacaModel
from src.custom.alpaca_data_module import AlpacaDataModule
from src.custom.instruction_data_module import InstructionDataModule
from src.custom.truthfulqa_data_module import TruthfulQADataModule
from src.custom.toxigen_data_module import ToxiGenDataModule
from peft import get_peft_model, LoraConfig

from torch.utils.data import Subset
import numpy as np
from sklearn.linear_model import LogisticRegression
import time

from adapters import AutoAdapterModel,list_adapters, BnConfig, DoubleSeqBnConfig


logging.basicConfig(level=logging.INFO)
torch.set_float32_matmul_precision("high")

def main(args):
    print("arguments".upper().center(80, "-"))
    print(args)
    print("-" * 80)

    model_key = args.model_key.replace("/", "-")
    if "gpt" in args.model_key or "Llama" in model_key \
        or "bloomz" in model_key or "gemma" in model_key or "Mistral" in model_key:
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
        append_eos = False  # t5 tokenizers already append eos
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

        model.add_adapter(adapter_name="seq_bn",config=bottleneck_config)

        for name, param in model.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False
        
        model.set_active_adapters("seq_bn")

        print("-"*20,"Bottleneck_Adapter","-"*20)
        trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params_count = sum(p.numel() for p in model.parameters())

        print(f"Trainable parameters: {trainable_params_count} || All parameters: {all_params_count} || ratio: {trainable_params_count/all_params_count}")
        

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    task_idxes = list(range(38))
    data_module = AlpacaDataModule(tokenizer=tokenizer,
                            data_path="./data/alpaca_data/alpaca_final.pkl",
                            dev_split_path="./data/alpaca_data/alpaca_dev_split_map.pkl",
                            task_idxes=task_idxes,
                            batch_size = args.batch_size,
                            inference_batch_size = args.batch_size,
                            context_length=args.max_length,
                            downsample=args.downsample,
                            model_type=model_type)
    
    data_module.setup(stage="fit")
    save_name = ("Instruction_{}".format(model_key) if (args.train_instruction or args.load_truthfulqa or args.load_toxigen) else "Alpaca_{}".format(model_key)) + \
                (f"_lora_r_{args.lora_rank}" if args.train_lora else "") + \
                ("_{}".format(args.save_name) if args.save_name != "none" else "")
    gradient_dir = save_name + f"_dim_{args.project_dimension}_run_{args.run}" + ("_pretrained" if args.load_model_dir is None else "_pretrained")
    print("Gradient directory", gradient_dir)

    if args.train_lora or args.train_adapter:
        if not os.path.exists(os.path.join("gradients", gradient_dir)):
            os.makedirs(os.path.join("gradients", gradient_dir))
        model_path = os.path.join("gradients", gradient_dir) + "/initial_weights.pt"
        state_dict = model.state_dict()
        state_dict = {k: v.clone() for k, v in state_dict.items() if "lora" or "adapter" in k}
        torch.save(state_dict, model_path)


    lm = AlpacaModel(model=model, tokenizer=tokenizer, model_type=model_type,
                    lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, use_wandb=args.use_wandb,
                    intialize_project_matrix=args.project_gradients, run_seed=args.run, 
                    project_dim=args.project_dimension, gradient_dir=gradient_dir, use_sgd=True,
                    predict_steps=args.num_batches_gradients)
    if args.load_model_dir is not None:
        load_model_dir = f"./exported_model/{args.load_model_dir}.pt"
        if os.path.exists(load_model_dir):
            state_dict = torch.load(load_model_dir, map_location=lm.model.device)
            model.load_state_dict(state_dict, strict=False)
            print("Loaded model from checkpoint from ", load_model_dir)
    

    args.accumulate = 1; args.epochs = 20; args.enable_checkpointing = True
    default_root_dir = "external_lightning_logs/" + save_name + "/eval_output_approx/" # This is for creating a new directory

    trainer = pl.Trainer(accelerator="gpu", devices=args.devices, strategy=args.strategy,
                        default_root_dir=default_root_dir, min_epochs=args.epochs, max_epochs=args.epochs,
                        accumulate_grad_batches=args.accumulate, precision=args.precision,
                        enable_checkpointing=args.enable_checkpointing, inference_mode=False
            )
    
    trainer.fit(lm, datamodule=data_module)
    
    '''First compute pretrain outputs'''
    start_time = time.time()
    if args.use_test:
        pretrain_outputs = trainer.predict(lm, dataloaders=data_module.test_dataloader())
    else:
        pretrain_outputs = trainer.predict(lm, dataloaders=data_module.train_dataloader())
    end_time = time.time()
    print("Time for computing gradients & outputs", end_time - start_time)
    pretrain_outputs = np.concatenate(pretrain_outputs, axis=0)
    print("Pretrained outputs shape", pretrain_outputs.shape)
    np.save(f"./gradients/{gradient_dir}/pretrain_outputs.npy", pretrain_outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", type=str, default="EleutherAI/gpt-neo-1.3B")
    parser.add_argument("--train_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--use_qlora", action="store_true")

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--load_model_dir", type=str, default=None)

    parser.add_argument("--train_instruction", action="store_true")
    parser.add_argument("--load_truthfulqa", action="store_true")
    parser.add_argument("--load_toxigen", action="store_true")

    parser.add_argument("--compute_pretrained_outputs", action="store_true")
    parser.add_argument("--downsample", type=int, default=None)
    parser.add_argument("--num_batches_gradients", type=int, default=100)
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--project_gradients", action="store_true")
    parser.add_argument("--project_dimension", type=int, default=200)
    parser.add_argument("--abs_scale", type=float, default=-1.0)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--save_name", type=str, default="none")
    parser.add_argument("--use_test", action="store_true")
    
    parser.add_argument("--train_adapter",action="store_true")
    parser.add_argument("--reduction_factor", type=int, default=128)
    parser.add_argument("--use_qadapter", action= "store_true")
    
    args = parser.parse_args()
    main(args)
