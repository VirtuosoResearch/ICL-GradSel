## Exploring a contrastive approach to instruction tuning

Project document: https://docs.google.com/document/d/16zgvFrDF_G8YhasyIG2ejB6D1s6lU5NLMQ8ZuCzswWo/edit?usp=sharing

Here are the procedure to implement our work:

1. Enviroment reconstruction
```bash
conda create -n metaicl python=3.8
conda activate metaicl
# pip install the correct torch version here
pip install git+https://github.com/huggingface/transformers.git@c37573806ab3526dd805c49cbe2489ad4d68a9d7
pip install -U scikit-learn
# pip install --upgrade transformers huggingface_hub
bash build_feature.sh
```

2. We need to generate data features:
```bash
python get_feature.py --task {task}
```


3. Commands to run inferences (Use gpt2-large as an example).
```bash
# TopK
python test.py --dataset {dataset} --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k {number} --topk

# RandomK
python test.py --dataset {dataset} --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k {number} --randomk

# unlabeled
python test.py --dataset {dataset} --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k {number} --unlabeled

# Forward Selection (k is the filter scale so it must larger than m)
python test.py --dataset {dataset} --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k {number} --m {number} --forsel

# ground truth based selection
python test.py --dataset {dataset} --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k {number} --ground

# GrapsICL
python test.py --dataset {dataset} --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k {number} --estim
```
This repository contains the code and models for our paper:

**What Can Transformers Learn In-Context? A Case Study of Simple Function Classes** <br>
*Shivam Garg\*, Dimitris Tsipras\*, Percy Liang, Gregory Valiant* <br>
Paper: http://arxiv.org/abs/2208.01066 <br><br>

![](setting.jpg)

```bibtex
    @InProceedings{garg2022what,
        title={What Can Transformers Learn In-Context? A Case Study of Simple Function Classes},
        author={Shivam Garg and Dimitris Tsipras and Percy Liang and Gregory Valiant},
        year={2022},
        booktitle={arXiv preprint}
    }
```

## Getting started
You can start by cloning our repository and following the steps below.

1. Install the dependencies for our code using Conda. You may need to adjust the environment YAML file depending on your setup.

    ```
    conda env create -f environment.yml
    conda activate in-context-learning
    ```

2. Download [model checkpoints](https://github.com/dtsip/in-context-learning/releases/download/initial/models.zip) and extract them in the current directory.

    ```
    wget https://github.com/dtsip/in-context-learning/releases/download/initial/models.zip
    unzip models.zip
    ```

3. [Optional] If you plan to train, populate `conf/wandb.yaml` with you wandb info.

That's it! You can now explore our pre-trained models or train your own. The key entry points
are as follows (starting from `src`):
- The `eval.ipynb` notebook contains code to load our own pre-trained models, plot the pre-computed metrics, and evaluate them on new data.
- `train.py` takes as argument a configuration yaml from `conf` and trains the corresponding model. You can try `python train.py --config conf/toy.yaml` for a quick training run.

# Maintainers
* [Shivam Garg](https://cs.stanford.edu/~shivamg/)
* [Dimitris Tsipras](https://dtsipras.com/)
