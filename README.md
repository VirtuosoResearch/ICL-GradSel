## Enhancing instruction tuning with gradient-free contrastive learning

Project document: https://docs.google.com/document/d/16zgvFrDF_G8YhasyIG2ejB6D1s6lU5NLMQ8ZuCzswWo/edit?usp=sharing

```bash
pip install scikit-learn
```

A small change in `./preprocess/fewshot_gym_dataset.py`:
Change line 22 to:
```python
parser.add_argument('--do_test', default=True, action='store_true',
                    help="Run 2 tasks per process to test the code")
```