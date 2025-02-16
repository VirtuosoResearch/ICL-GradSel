import json
import numpy as np

with open("randomk_hessian.json", "r", encoding="utf-8") as f:
    hessian_list = json.load(f)
with open("randomk_loss.json", "r", encoding="utf-8") as f:
    loss_list = json.load(f)

loss_list_k = np.array(loss_list)
hessian_list_k = np.array(hessian_list)

print(loss_list_k)

