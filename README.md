## Linear-Time Demonstration Selection for In-Context Learning via Gradient Estimation

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
# GrapsICL
python test.py --dataset {dataset} --gpt2 gpt2-large --method direct --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k {number} --estim
```
