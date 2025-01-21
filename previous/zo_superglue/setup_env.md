using following commands to construct enviroment.

```bash
conda create -n mezo python=3.9.7
conda activate mezo
pip install transformers
pip install accelerate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install datasets
pip3 uninstall torchvision
pip install -U scikit-learn
pip uninstall transformers
pip install transformers==4.28.1
pip install peft
```

**Attention: `transformers` version should be 4.28.1**

---

Using following commands to run:

```bash
# Zero-shot
MODEL=facebook/opt-13b TASK=SST2 bash icl.sh --num_train 0

# In-context learning
MODEL=facebook/opt-13b TASK=SST2 bash icl.sh 

# Full-parameter fine-tuning, prefix-tuning, and LoRA
MODEL=facebook/opt-1.3b TASK=SST2 MODE=ft LR=1e-5 bash finetune.sh
MODEL=facebook/opt-1.3b TASK=SST2 MODE=prefix LR=1e-2 bash finetune.sh
MODEL=facebook/opt-1.3b TASK=SST2 MODE=lora LR=1e-4 bash finetune.sh

# Full-parameter fine-tuning using fully-sharded data parallel or FSDP (multi-GPU)
MODEL=facebook/opt-13b TASK=SST2 MODE=ft LR=1e-5 NUM_GPU=4 bash finetune_fsdp.sh

# MeZO (full-parameter, prefix-tuning, and LoRA)
MODEL=facebook/opt-13b TASK=SST2 MODE=ft LR=1e-7 EPS=1e-3 bash mezo.sh
MODEL=facebook/opt-13b TASK=SST2 MODE=prefix LR=1e-3 EPS=1e-1 bash mezo.sh
MODEL=facebook/opt-13b TASK=SST2 MODE=lora LR=5e-5 EPS=1e-2 bash mezo.sh

# MeZO with non-differentiable objective (SQuAD (F1) + MeZO prefix as an example)
MODEL=facebook/opt-13b TASK=SQuAD MODE=prefix LR=1e-2 EPS=1e-1 bash mezo.sh --non_diff --evaluation_strategy no --save_strategy no --save_model
```

---
Details about ZeRO is in class `trainer.OurTrainer`