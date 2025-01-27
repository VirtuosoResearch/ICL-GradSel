## Exploring a contrastive approach to instruction tuning

Project document: https://docs.google.com/document/d/16zgvFrDF_G8YhasyIG2ejB6D1s6lU5NLMQ8ZuCzswWo/edit?usp=sharing

<!-- 
A small change in `./preprocess/fewshot_gym_dataset.py`:
Change line 22 to:
```python
parser.add_argument('--do_test', default=True, action='store_true',
                    help="Run 2 tasks per process to test the code")
``` -->

---

There are two steps to implement this project:

### 1. generate the datas

```bash
conda create --name metaicl-data python=3.8
conda activate metaicl-data
pip install datasets==1.4.0 wget
cd preprocess
python {dataset}.py
```
{dataset} here can be `poem_sentiment`, `climate_fever`, `superglue-cb`, `glue-rte`, `glue-sst2`, `glue-wnli`. 

Pay attention that for some dataset, we only generate the training dataset, so we need to rename `{dataset}_train.jsonl` to `{dataset}_test.jsonl`

### 2. implement the experiments

**Note that a new environment need to be built.**

```bash
conda create -n metaicl python=3.8
conda activate metaicl
# pip install the correct torch version here
pip install git+https://github.com/huggingface/transformers.git@c37573806ab3526dd805c49cbe2489ad4d68a9d7
pip install -U scikit-learn
# pip install --upgrade transformers huggingface_hub
bash build_feature.sh
```

First need to generate data features:
```bash
python get_feature.py --task {task}
```


Commands to run inferences (Use gpt2-large as an example).
```bash
# TopK
python test.py --dataset {dataset} --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k {number} --topk

# RandomK
python test.py --dataset {dataset} --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k {number} --randomk

# unlabeled
python test.py --dataset {dataset} --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k {number} --unlabeled

# Supervised Contrastive Loss (k must larger than m)
python test.py --dataset {dataset} --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k {number} --m {number} --supcon

# Random Ensemble (k must larger than m)
python test.py --dataset {dataset} --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k {number} --m {number} --ranens

# Forward Selection (k must larger than m)
python test.py --dataset {dataset} --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k {number} --m {number} --forsel
```
