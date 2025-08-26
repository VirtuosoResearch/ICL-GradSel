# Linear-Time Demonstration Selection for In-Context Learning via Gradient Estimation
- Authors: [Ziniu Zhang](https://ziniuzhang.github.io/), [Zhenshuo Zhang](https://zhenshuozhang.github.io/), [Dongyue Li](https://lidongyue12138.github.io/), [Lu Wang](https://web.eecs.umich.edu/~wangluxy/), [Jennifer Dy](https://mllabneu.github.io/) and [Hongyang R. Zhang](https://www.hongyangzhang.com/).



## Usage
Here are the procedure to implement our work:

1. Enviroment reconstruction
```bash
conda create -n gradsel python=3.8
conda activate gradsel
# pip install the correct torch version here
pip install git+https://github.com/huggingface/transformers.git@c37573806ab3526dd805c49cbe2489ad4d68a9d7
pip install -U scikit-learn
```
Also, we provide an [environment file](./environment.yml) including the python package versions we used in our experiments.

2. We need to first generate data features:
```bash
python ./utils/get_feature.py --task {task} --model {model key}
```

3. Commands to run inferences.
```bash
# GradRE
python test.py --dataset {dataset} --gpt2 {model key} --method direct --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 0 --k {number} --num_anchors {number} --estim
# GradFS
python test.py --dataset {dataset} --gpt2 {model key} --method direct --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 0 --k {number} --forsel
# GradCE
python test.py --dataset {dataset} --gpt2 {model key} --method direct --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 0 --k {number} --gradestim
```

4. Examples.

For the linear-regression task, we provide the [usage file](./linear_regression/README.md) as an example.

Then, we provide an example of running GradRE on SST-2 dataset.
```bash
python test.py --dataset sst2 --gpt2 deepseek-ai/deepseek-llm-7b-chat --method direct --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 0 --k 8 --ranens
```
The output is the F1-score of our method.

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