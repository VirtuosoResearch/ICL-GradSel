from collections import OrderedDict
import re
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm.notebook import tqdm
import numpy as np
import math

from eval import get_run_metrics, read_run_dir, get_model_from_run
from tasks import get_task_sampler
from samplers import get_data_sampler

class Sample():
    def __init__(self, x, y, c, beta):
        self.x = x
        self.y = y
        self.c = c
        self.beta = beta
    
    def __repr__(self):
        return f"x: {self.x}, y: {self.y}, c: {self.c}\n"

class Input_sequence():
    def __init__(self, xs, ys, length, beta, c):
        self.xs = xs
        self.ys = ys
        self.prompt_x = xs[:length]
        self.prompt_y = ys[:length]
        self.query_x = xs[-1]
        self.query_y = ys[-1]
        self.beta = beta
        self.c = c
    
    def add_sample(self, sample):
        self.prompt_x = torch.cat([self.prompt_x, sample.x.unsqueeze(0)], dim=0)
        self.prompt_y = torch.cat([self.prompt_y, sample.y.unsqueeze(0)], dim=0)
    
    def del_sample(self):
        self.prompt_x = self.prompt_x[:-1]
        self.prompt_y = self.prompt_y[:-1]

    def add_xy(self, x, y):
        self.prompt_x = torch.cat([self.prompt_x, x.unsqueeze(0)], dim=0)
        self.prompt_y = torch.cat([self.prompt_y, y.unsqueeze(0)], dim=0)
    
    def add_query(self, sample):
        self.query_x = sample.x
        self.query_y = sample.y
    
    def get_input(self, query_index=-1, query_range=0, query_sample=None, last_prompt=False):
        if last_prompt:
            self.input_x = self.prompt_x
            self.input_y = self.prompt_y
        elif query_sample is not None:
            self.input_x = torch.cat([self.prompt_x, query_sample.x.unsqueeze(0)], dim=0)
            self.input_y = torch.cat([self.prompt_y, query_sample.y.unsqueeze(0)], dim=0)
        elif query_range > 0:
            self.input_x = torch.cat([self.prompt_x, self.xs[-query_range-1:-1]], dim=0)
            self.input_y = torch.cat([self.prompt_y, self.ys[-query_range-1:-1]], dim=0)
        elif query_index == -1:
            self.input_x = torch.cat([self.prompt_x, self.query_x.unsqueeze(0)], dim=0)
            self.input_y = torch.cat([self.prompt_y, self.query_y.unsqueeze(0)], dim=0)
        else:
            self.input_x = torch.cat([self.prompt_x, self.xs[query_index].unsqueeze(0)], dim=0)
            self.input_y = torch.cat([self.prompt_y, self.ys[query_index].unsqueeze(0)], dim=0)
        return self.input_x, self.input_y
    
    def pad(self):
        self.prompt_x = torch.cat([self.prompt_x, self.prompt_x[-1].unsqueeze(0)], dim=0)
        self.prompt_y = torch.cat([self.prompt_y, self.prompt_y[-1].unsqueeze(0)], dim=0)
    
    def get_prompt_length(self):
        return self.prompt_x.shape[0]
    
    def __repr__(self):
        return f"prompt_x: {self.prompt_x}, prompt_y: {self.prompt_y}, query_x: {self.query_x}, query_y: {self.query_y}\n"

def sequence_to_tensor(seq_list):
    xs = torch.stack([s.input_x for s in seq_list], dim=0)
    ys = torch.stack([s.input_y for s in seq_list], dim=0)
    return xs, ys

run_dir = "../models"



task = "linear_regression"
#task = "sparse_linear_regression"
#task = "decision_tree"
#task = "relu_2nn_regression"

run_id = "pretrained"  # if you train more models, replace with the run_id from the table above

run_path = os.path.join(run_dir, task, run_id)
recompute_metrics = False
print(run_path)
if recompute_metrics:
    get_run_metrics(run_path)  # these are normally precomputed at the end of training

model, conf = get_model_from_run(run_path)

n_dims = conf.model.n_dims
batch_size = conf.training.batch_size

data_sampler = get_data_sampler(conf.training.data, n_dims)
task_sampler = get_task_sampler(
    conf.training.task,
    n_dims,
    batch_size,
    **conf.training.task_kwargs
)

def predict_full_label(model, xs, ys, labels):
    with torch.no_grad():
        pred = model(xs, ys)
    metric = task.get_metric()
    loss = metric(pred, labels).cpu().numpy()

    return pred, loss.mean(axis=0)

import torch.nn.functional as F
import copy
from datetime import datetime

device = 'cuda:1'
model = model.to(device)

model.eval()
n_total = 41
n_labeled = 25
n_dims = conf.model.n_dims
#b_size = conf.training.batch_size
b_size = 50
ratio = n_total // n_labeled

n_unlabeled = n_total - n_labeled
runs = 1
loss_full_label_list = []
loss_random_list = []
loss_beta_list = []
loss_loss_list = []
loss_contrastive_list = []
loss_fs_inference_list = []
loss_fs_estimate_list = []

def random_select(model, xs, ys, beta, sample_list, n_labeled, set_size_list):
    loss_list = []
    for set_size in set_size_list:
        seq_list = []
        for i in range(xs.shape[0]):
            seq_list.append(Input_sequence(xs[i], ys[i], n_labeled, beta[i], i))

        for i in range(xs.shape[0]):
            for j in range(set_size):
                select_index = torch.randint(0, len(sample_list), (1,))
                seq_list[i].add_sample(sample_list[select_index])

        for seq in seq_list:
            seq.get_input()

        xs, ys = sequence_to_tensor(seq_list)
        
        with torch.no_grad():
            pred = model(xs, ys)

        metric = task.get_metric()
        loss = metric(pred, ys).cpu().numpy()
        loss_query = loss.mean(axis=0)[-1]

        loss_list.append(loss_query)
    
    return loss_list

def fs_select(model, xs, ys, beta, sample_list, b_size, n_labeled, set_size_list, use_approx=True):
    n_points = xs.shape[1] - 1
    n = len(sample_list)
    print(n)
    pos_index = torch.zeros(n, n)
    seq_list = []
    for i in range(xs.shape[0]):
        seq_list.append(Input_sequence(xs[i], ys[i], n_labeled, beta[i], i))

    # init select index for query in each sequence
    loss_list = []
    loss_list_infer = []
    max_points = np.max(set_size_list)
    sub_sample_list = [[] for _ in range(b_size)]
    sub_sample_index = [[] for _ in range(b_size)]
    sub_sample_list_infer = [[] for _ in range(b_size)]
    sub_sample_index_infer = [[] for _ in range(b_size)]
    select_index = 0
    b_size = 5
    seq_list = seq_list[:b_size]
    for k in range(1, max_points+1):
        for i in range(b_size):
            sub_seq_list = []
            n_val = 5
            i_seq = copy.deepcopy(seq_list[i])
            i_seq.get_input(last_prompt=True)
            #sub_seq.get_input(query_range=n_val)
            xs, ys = sequence_to_tensor([i_seq])
            pred_0, embeds_0 = model.forward_with_embeds(xs, ys)
            grad = torch.autograd.grad(pred_0[0, -1], embeds_0, retain_graph=True, create_graph=True)[0]
            grad = grad[:, 0::2, :]
            grad_0 = grad[:, -1, :]
            embeds_0 = embeds_0[:, 0::2, :]

            metric = task.get_metric()

            delta_x = []
            for j in range(n):
                if j in sub_sample_index[i]:
                    continue
                sub_seq = copy.deepcopy(seq_list[i])

                sub_seq.add_sample(sample_list[j])
                sub_seq.get_input(last_prompt=True)

                sub_seq_list.append(sub_seq)

            sub_xs, sub_ys = sequence_to_tensor(sub_seq_list)
            embeds = model.embed_x(sub_xs)
            delta_embeds = embeds[:, -1, :] - embeds[:, -2, :]

            delta_term = torch.sum(grad_0 * delta_embeds, dim=1, keepdim=True)

            approx_pred = delta_term + pred_0[0, -1]

            xs, ys = sequence_to_tensor(sub_seq_list)
            with torch.no_grad():
                pred = model(xs, ys)

            metric = task.get_metric()
            loss_infer = metric(pred, ys)
            loss_approx = metric(approx_pred, ys)

            if use_approx:
                score = loss_approx[:, -1]
            else:
                score = loss_infer[:, -1]

            #score = score_infer
            select_index = torch.argmin(score)
            sub_sample_index[i].append(select_index)
            sub_sample_list[i].append(sample_list[select_index])
            seq_list[i].add_sample(sample_list[select_index])

            print(sub_sample_index[i])
            if k in set_size_list:
                seq_list[i].get_input()
        if k in set_size_list:
            # get the input tensor
            xs, ys = sequence_to_tensor(seq_list)
            with torch.no_grad():
                pred = model(xs, ys)
            metric = task.get_metric()
            loss = metric(pred, ys).cpu().numpy()
            loss_query = loss.mean(axis=0)[-1]
            #print(loss_query)
            loss_list.append(loss_query)

    return loss_list




def generate_synthetic_data(num_sequences=100, n_points=10, x_dim=8, diff_diftribution=False, alpha=0.9):
    xs_b = torch.randn(num_sequences, n_points, x_dim)
    # same mean for all points
    if diff_diftribution:
        for i in range(xs_b.shape[0]):
            mean_vectors = torch.randn(1, x_dim)
            for j in range(xs_b.shape[1]):
                xs_b[i, j, :] = ((1-alpha) * xs_b[i, j, :] + alpha * mean_vectors)
    #print(xs_b[0])
    return xs_b

def generate_orthogonal_matrix(n, m):
    if n > m:
        raise ValueError("The number of rows (n) should not be greater than the number of columns (m) for an orthogonal set.")
    
    # Generate a random matrix
    A = torch.randn(n, m)
    
    # Apply Gram-Schmidt process
    Q, _ = torch.linalg.qr(A.T)  # QR decomposition on the transpose
    return Q.T  # Transpose back to get row-wise orthogonality

runs = 1
for run in range(runs):
    task = task_sampler()
    xs = generate_synthetic_data(num_sequences=b_size, n_points=n_total, x_dim=n_dims, diff_diftribution=False, alpha=0.0)
    xs = xs.to(device)
    n_classes = 2

    anchor_points = generate_orthogonal_matrix(n_classes, n_dims).to(device)
    beta = torch.randn(b_size, n_dims, 1).to(device)
    alpha = 1
    for i in range(beta.shape[0]):
        i_class = torch.randint(0, n_classes, (1,))
        beta[i] = ((1-alpha)*beta[i] + anchor_points[i_class].T)

    ys = (xs @ beta)[:, :, 0]
    labels = ys.clone()
    with torch.no_grad():
        embedding = model.encoder(xs, ys)
    print(embedding.shape)

    sample_list = []
    for i in range(xs.shape[0]):
        # -1 for remaining the last sample as query
        for j in range(xs.shape[1]-1):
            s = Sample(xs[i, j], ys[i, j], i, beta[i])
            sample_list.append(s)

    set_size_list = [1,2,3,4,5,10]
    pred_full_label, loss_full_label = predict_full_label(model, xs, ys=labels, labels=labels)
    loss_full_label_list.append(loss_full_label)
    
    loss_fs_estimate = fs_select(model, xs, ys, beta, sample_list, b_size, n_labeled, set_size_list, use_approx=True)
    #loss_fs_estimate = fs_select_debug3(model, xs, ys, beta, sample_list, b_size, n_labeled, set_size_list)
    print(loss_fs_estimate)
    loss_fs_estimate_list.append(loss_fs_estimate)

    loss_fs_inference = fs_select(model, xs, ys, beta, sample_list, b_size, n_labeled, set_size_list, use_approx=False)
    print(loss_fs_inference)
    loss_fs_inference_list.append(loss_fs_inference)
    
    loss_random = random_select(model, xs, ys, beta, sample_list, n_labeled, set_size_list)
    print(loss_random)
    loss_random_list.append(loss_random)

x = np.arange(n_total)
plt.plot(x, np.mean(loss_full_label_list, axis=0), lw=2, label="Full label")
plt.fill_between(x, np.mean(loss_full_label_list, axis=0)-np.std(loss_full_label_list, axis=0), np.mean(loss_full_label_list, axis=0)+np.std(loss_full_label_list, axis=0), alpha=0.2)

x = []
for i in set_size_list:
    x.append(n_labeled + i)
x = np.array(x)
print(x)

#plt.plot(x, np.mean(loss_loss_list, axis=0), lw=2, label="loss")
#plt.fill_between(x, np.mean(loss_loss_list, axis=0)-np.std(loss_loss_list, axis=0), np.mean(loss_loss_list, axis=0)+np.std(loss_loss_list, axis=0), alpha=0.2)

#plt.plot(x, np.mean(loss_random_list, axis=0), lw=2, label="random")
#plt.fill_between(x, np.mean(loss_random_list, axis=0)-np.std(loss_random_list, axis=0), np.mean(loss_random_list, axis=0)+np.std(loss_random_list, axis=0), alpha=0.2)
#plt.plot(x, np.mean(loss_beta_list, axis=0), lw=2, label="beta")
#plt.fill_between(x, np.mean(loss_beta_list, axis=0)-np.std(loss_beta_list, axis=0), np.mean(loss_beta_list, axis=0)+np.std(loss_beta_list, axis=0), alpha=0.2)
#plt.plot(x, np.mean(loss_contrastive_list, axis=0), lw=2, label="contrastive")
#plt.fill_between(x, np.mean(loss_contrastive_list, axis=0)-np.std(loss_contrastive_list, axis=0), np.mean(loss_contrastive_list, axis=0)+np.std(loss_contrastive_list, axis=0), alpha=0.2)

plt.plot(x, np.mean(loss_fs_inference_list, axis=0), lw=2, label="inference")
plt.fill_between(x, np.mean(loss_fs_inference_list, axis=0)-np.std(loss_fs_inference_list, axis=0), np.mean(loss_fs_inference_list, axis=0)+np.std(loss_fs_inference_list, axis=0), alpha=0.2)
plt.plot(x, np.mean(loss_fs_estimate_list, axis=0), lw=2, label="estimate")
plt.fill_between(x, np.mean(loss_fs_estimate_list, axis=0)-np.std(loss_fs_estimate_list, axis=0), np.mean(loss_fs_estimate_list, axis=0)+np.std(loss_fs_estimate_list, axis=0), alpha=0.2)


#plt.plot(loss_full_label, lw=2, label="Full label")
#plt.plot(loss_unlabeled_once, lw=2, label="Unlabeled once")
#plt.plot(loss_unlabeled_iter, lw=2, label="Unlabeled iter")
#plt.plot(loss_unlabeled_stepbystep, lw=2, label="Unlabeled step by step")
plt.xlabel("# in-context examples")
plt.ylabel("squared error")
plt.legend()
plt.savefig("LR_test.png")