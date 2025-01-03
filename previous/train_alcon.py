import json
import torch
from data.Alcon_QA import train_dataloader, test_dataloader

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

from adapters import AutoAdapterModel, DoubleSeqBnConfig



def main(args):
    print("arguments".upper().center(80, "-"))
    print(args)
    print("-" * 80)
    device = torch.device(f"cuda:{args.devices[0]}")
    print("device: ", device)

    hf_key = args.model_key.replace("_", "-")
    tokenizer = AutoTokenizer.from_pretrained(hf_key)
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(hf_key)
    model_type = "decoder"
    append_eos = True
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.train_adapter:
        model = AutoAdapterModel.from_pretrained(args.model_key)

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
        
        print("model.device: ", model.device)
