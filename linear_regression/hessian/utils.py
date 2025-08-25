import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
import time

from pyhessian.utils import group_product, group_add, normalization, get_params_grad, hessian_vector_product, orthnormal

from scipy.stats import norm
from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from scipy.stats import entropy
from scipy.stats import pearsonr

def get_layers(model):
    layers = OrderedDict()
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and any(p.requires_grad for p in module.parameters(recurse=False)):
            #if (type(module) == torch.nn.Linear) and "LayerNorm" not in name and "ln" not in name and "embeddings" not in name and "pooler" not in name:
            if "LayerNorm" not in name and "ln" not in name and "pooler" not in name:
            #print(f"Layer: {name}, Module: {module}")
                layers[name] = module
    return layers