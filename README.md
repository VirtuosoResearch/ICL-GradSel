# Linear-Time Demonstration Selection for In-Context Learning via Gradient Estimation
- Authors: [Ziniu Zhang](https://ziniuzhang.github.io/), [Zhenshuo Zhang](https://zhenshuozhang.github.io/), [Dongyue Li](https://lidongyue12138.github.io/), [Lu Wang](https://web.eecs.umich.edu/~wangluxy/), [Jennifer Dy](https://mllabneu.github.io/) and [Hongyang R. Zhang](https://www.hongyangzhang.com/).



## Usage
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
python test.py --dataset {dataset} --gpt2 meta-llama/Llama-3.2-3B-Instruct --method direct --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k {number} --estim
```

## Reference
If you find this repository useful or happen to use it in a research paper, please cite our work with the following Bib information.

```
@article{zhang2025linear,
  title={Linear-Time Demonstration Selection for In-Context Learning via Gradient Estimation},
  author={Zhang, Ziniu and Zhang, Zhenshuo and Li, Dongyue and Wang, Lu and Dy, Jennifer and Zhang, Hongyang R.},
  booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
  year={2025},
}
```