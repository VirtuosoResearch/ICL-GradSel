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

```

**Attention: `transformers` version should be 4.28.1**

---

Using following commands to run:

```bash
# for Zero-order
MODEL=facebook/opt-13b TASK=SST2 bash icl.sh --num_train 0
# for In-context learning
MODEL=facebook/opt-13b TASK=SST2 bash icl.sh
```