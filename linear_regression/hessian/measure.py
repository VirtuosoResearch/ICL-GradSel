import torch

import math
from torch.autograd import Variable
import numpy as np

def tf_layer_distance(model):
    W = model.get_W()
    W0 = model.get_init_W().to(W.device)
    with torch.no_grad():
        l2_norm = torch.norm(W - W0, p=2)
    #print(l2_norm)
    return l2_norm

def measure_hessian(model, hessian_trace, train_num):
    #measurement = hessian_trace * tf_layer_distance(model)
    C = torch.tensor(5)
    measurement = torch.sqrt(C*hessian_trace/train_num) * tf_layer_distance(model)
    #measurement = C*hessian_trace * tf_layer_distance(model) / train_num

    return measurement
