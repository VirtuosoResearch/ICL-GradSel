from collections import OrderedDict
import re
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm
import numpy as np
import math
import time
import json

from eval import get_run_metrics, read_run_dir, get_model_from_run
from plot_utils import basic_plot, collect_results, relevant_model_names

from torch.nn.attention import SDPBackend, sdpa_kernel

sns.set_theme('notebook', 'darkgrid')
palette = sns.color_palette('colorblind')

run_dir = "../models"

from samplers import get_data_sampler
from tasks import get_task_sampler
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
        #pred = model.seq_inference(xs, ys)
        pred = model(xs, ys)
    #print(pred[0])
    metric = task.get_metric()
    loss = metric(pred, labels).cpu().numpy()

    return pred, loss.mean(axis=0)

def predict_unlabled_once(model, xs, ys, labels, n_labeled):
    with torch.no_grad():
        #pred = model.seq_inference(xs, ys)
        pred = model(xs, ys)
    #print(pred[0])
    metric = task.get_metric()
    loss = metric(pred, labels).cpu().numpy()
    loss_labeled = loss[:, :n_labeled]
    loss_unlabeled = loss[:, n_labeled:]

    return pred, loss.mean(axis=0), loss_labeled.mean(axis=0), loss_unlabeled.mean(axis=0)

def predict_unlabled_iteration(model, xs, ys, labels, n_labeled, n_unlabeled, n_iter, ratio=1):
    #ratio = math.ceil((n_labeled + n_unlabeled) // n_unlabeled)
    #ratio = 2
    # find the most similar unlabeled data
    for b in range(xs.shape[0]):
        for i in range(n_unlabeled):
            l2_dist = torch.norm(xs[b, :] - xs[b, n_labeled+i].unsqueeze(0), dim=1)
            best_match = torch.argmin(l2_dist)
            #ys[b, n_labeled+i] = ys[b, best_match]


    print(f"ratio: {ratio}")
    loss_unlabeled_list = []
    for i in range(n_iter):
        with torch.no_grad():
            #pred = model.seq_inference(xs, ys)
            pred = model(xs, ys)
        #print(pred[0])
        metric = task.get_metric()
        loss = metric(pred, labels).cpu().numpy()
        loss_labeled = loss[:, :n_labeled]
        loss_unlabeled = loss[:, n_labeled:]
        #print(loss.shape)
        #print(loss_unlabeled.shape)
        loss_unlabeled_list.append(loss_unlabeled)
        
        #print(loss_unlabeled.mean())

        # update ys by random choice
        ys[:, n_labeled:] = pred[:, n_labeled:]
        
        index = np.random.choice(n_unlabeled, n_unlabeled//ratio, replace=False)
        print(f"iter {i}: labeled loss {loss_labeled.mean()}, unlabeled loss {loss_unlabeled.mean()}, mask {index}")
        if len(loss_unlabeled_list) > 1 and np.abs(loss_unlabeled_list[-1].mean() - loss_unlabeled_list[-2].mean()) < 1e-3:
            break
        #print(index)
        #print(ys[:, n_labeled:].shape)
        #print(ys[:, n_labeled:][index+ n_labeled].shape)
        for j in range(n_unlabeled):
            if j not in index:
                ys[:, n_labeled+j] = torch.zeros_like(ys[:, n_labeled+j])
        #print(ys[0, n_labeled:])
    
    return pred, loss.mean(axis=0), loss_labeled.mean(axis=0), loss_unlabeled.mean(axis=0)

def predict_unlabled_stepbysetp(model, xs, ys, labels, n_labeled, n_unlabeled):
    for i in range(n_unlabeled):
        with torch.no_grad():
            #pred = model.seq_inference(xs, ys)
            pred = model(xs, ys)
        #print(pred[0])
        metric = task.get_metric()
        loss = metric(pred, labels).cpu().numpy()
        loss_labeled = loss[:, :n_labeled]
        loss_unlabeled = loss[:, n_labeled:]
        #print(loss.shape)
        #print(loss_unlabeled.shape)
        #print(loss_unlabeled.mean())

        # update ys by random choice
        ys[:, n_labeled+i] = pred[:, n_labeled+i]
        print(f"iter {i}: labeled loss {loss_labeled.mean()}, unlabeled loss {loss_unlabeled.mean()}")

    return pred, loss.mean(axis=0), loss_labeled.mean(axis=0), loss_unlabeled.mean(axis=0)

import torch.nn.functional as F
import copy
from datetime import datetime

device = 'cuda:1'
model = model.to(device)

model.eval()
use_checkpoint = True
if not use_checkpoint:
    n_total = 5
    n_labeled = 2
    n_dims = 2
    b_size = 3
else:
    n_total = 41
    n_labeled = 20
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

same_distribution = True

noise = 1
add_set_size = 5

def loss_score(model, seq, n_labeled, all_sample_list, sub_sample_list, task):
    n_sub_prompt = n_labeled - 1
    n_val = 5
    n = len(all_sample_list)
    sub_seq_list = []
    start_time = datetime.now()
    for i in range(n):
        sub_seq = copy.deepcopy(seq)
        # becausre sample_list contains the last sample
        
        #sub_seq.get_input(query_index=-2)
        #sub_seq.add_sample(all_sample_list[i])
        #sub_seq.get_input(query_range=n_val)
        sub_seq.get_input(query_sample=all_sample_list[i])
        sub_seq_list.append(sub_seq)
    print(datetime.now() - start_time)
    start_time = datetime.now()
    xs, ys = sequence_to_tensor(sub_seq_list)
    with torch.no_grad():
        pred = model(xs, ys)
    metric = task.get_metric()
    loss = metric(pred, ys)
    loss_val = loss[:, -n_val:].mean(-1)
    #score = loss[:, -1]
    score = loss_val
    print("model time", datetime.now() - start_time)
    #print(loss_sample)
    return score



def contrastive_select(model, xs, ys, beta, sample_list, b_size, n_labeled, set_size_list):
    n_points = xs.shape[1] - 1
    n = len(sample_list)
    print(n)
    pos_index = torch.zeros(n, n)
    seq_list = []
    for i in range(xs.shape[0]):
        seq_list.append(Input_sequence(xs[i], ys[i], n_labeled, beta[i], i))

    # init select index for query in each sequence
    loss_list = []
    for set_size in set_size_list:
        for i in range(b_size):
            # use the first sample as the initilization
            first_index = i * n_points
            init_index = np.arange(i*n_points, i*n_points+n_labeled)
            #sub_sample_list  = [sample_list[j] for j in init_index]
            sub_sample_list = [sample_list[init_index[-1]]]
            #sub_sample_list.append(sample_list[first_index])
            # traverse all the samples except the first one
            #sub_sample_index = init_index.tolist()
            sub_sample_index = [init_index[-1]]
            #sub_sample_index.append(first_index)
            score = torch.zeros(n)
            #score[first_index] = 1e9
            for _ in init_index:
                score[_] = 1e9
            for k in range(set_size):
                for j in range(n):
                    if j in sub_sample_index:
                        continue
                    # compute the contrastive score for each sample
                    #score[j] = distance_score(model, seq_list[i], sub_sample_list, sample_list[j])
                    #score[j] = x_distance_score(model, sub_sample_list, sample_list[j])
                    #score[j] = contrastive_score(model, seq_list[i], n_labeled, sub_sample_list, sample_list[j])
                    score[j] = loss_score(model, seq_list[i], n_labeled, sub_sample_list, sample_list[j], task)
                select_index = torch.argmin(score)
                sub_sample_index.append(select_index)
                sub_sample_list.append(sample_list[select_index])
                seq_list[i].add_sample(sample_list[select_index])
                score[select_index] = 1e9
            print(sub_sample_index)
            # get prompt
            seq_list[i].get_input()
        # get the input tensor
        xs, ys = sequence_to_tensor(seq_list)
        with torch.no_grad():
            pred = model(xs, ys)
        metric = task.get_metric()
        loss = metric(pred, ys).numpy()
        loss_query = loss.mean(axis=0)[-1]
        #print(loss_query)
        loss_list.append(loss_query)

    return loss_list

def fs_inference_select_(model, xs, ys, beta, sample_list, b_size, n_labeled, set_size_list):
    n_points = xs.shape[1] - 1
    n = len(sample_list)
    print(n)
    pos_index = torch.zeros(n, n)
    seq_list = []
    for i in range(xs.shape[0]):
        seq_list.append(Input_sequence(xs[i], ys[i], n_labeled, beta[i], i))

    # init select index for query in each sequence
    loss_list = []
    max_points = np.max(set_size_list)
    sub_sample_list = [[] for _ in range(b_size)]
    sub_sample_index = [[] for _ in range(b_size)]
    b_size = 1
    for k in range(1, max_points+1):
        for i in range(b_size):
            
            same_label_index = [i * n_points + j for j in range(n_points)]

            score = loss_score(model, seq_list[i], n_labeled, sample_list, sub_sample_list[i], task)
            #score = estimate_score(model, seq_list[i], n_labeled, sample_list, sub_sample_list[i], task)

            select_index = torch.argmin(score)
            sub_sample_index[i].append(select_index)
            sub_sample_list[i].append(sample_list[select_index])
            seq_list[i].add_sample(sample_list[select_index])
            score[select_index] = 1e9
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

def fs_estimate_select(model, xs, ys, beta, sample_list, b_size, n_labeled, set_size_list):
    n_points = xs.shape[1] - 1
    n = len(sample_list)
    print(n)
    pos_index = torch.zeros(n, n)
    seq_list = []
    for i in range(xs.shape[0]):
        seq_list.append(Input_sequence(xs[i], ys[i], n_labeled, beta[i], i))

    # init select index for query in each sequence
    loss_list = []
    max_points = np.max(set_size_list)
    sub_sample_list = [[] for _ in range(b_size)]
    sub_sample_index = [[] for _ in range(b_size)]
    for k in range(1, max_points+1):
        for i in range(b_size):
            
            same_label_index = [i * n_points + j for j in range(n_points)]

            #score = loss_score(model, seq_list[i], n_labeled, sample_list, sub_sample_list[i], task)
            score = estimate_score(model, seq_list[i], n_labeled, sample_list, sub_sample_list[i], task)

            select_index = torch.argmin(score)
            sub_sample_index[i].append(select_index)
            sub_sample_list[i].append(sample_list[select_index])
            seq_list[i].add_sample(sample_list[select_index])
            score[select_index] = 1e9
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

def evaluate_loss(model, seq_list, compute_hessian):
    xs, ys = sequence_to_tensor(seq_list)
    pred, embeds = model.forward_with_embeds(xs, ys)
    metric = task.get_metric()
    loss = metric(pred, ys)
    loss_query = loss.mean(dim=0)[-1]
    trace_vhv = 0
    if compute_hessian:
        grad = torch.autograd.grad(loss_query, embeds, retain_graph=True, create_graph=True)[0]
        # estimate Hessian trace by vhv
        n_sample = 10
        for _ in range(n_sample):
            v = torch.randint_like(embeds, low=0, high=2, device=device) * 2 - 1
            v = v.to(embeds.dtype)
            Hv = torch.autograd.grad(grad, embeds, grad_outputs=v, retain_graph=True, create_graph=False)[0]
            vHv = (v * Hv).sum()
            trace_vhv += vHv
            del v
        trace_vhv /= n_sample
        trace_vhv = trace_vhv.item()
    loss_query = loss_query.item()

    return loss_query, trace_vhv

def fs_inference_select(model, xs, ys, beta, sample_list, b_size, n_labeled, set_size_list):
    n_points = xs.shape[1] - 1
    n = len(sample_list)
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
    b_size = 1
    seq_list = seq_list[:b_size]
    overfit_sample = []
    generalize_sample = []
    overfit = False
    k_max = 3
    for k in range(k_max):
        if overfit:
            print("early stop")
            break
        for i in range(b_size):
            sub_seq_list = []

            for j in range(n):
                if j in sub_sample_index[i]:
                    continue
                sub_seq = copy.deepcopy(seq_list[i])

                sub_seq.add_sample(sample_list[j])
                #sub_seq.get_input(last_prompt=True)
                sub_seq.get_input(query_index=-1)

                sub_seq_list.append(sub_seq)

            xs, ys = sequence_to_tensor(sub_seq_list)
            with torch.no_grad():
                pred = model(xs, ys)

            metric = task.get_metric()
            loss_infer = metric(pred, ys)
            score = loss_infer[:, -1]

            #score = score_infer
            # sort score
            sorted_score, sorted_index = torch.sort(score)
            candidate_indices = sorted_index[:20]
            # find the index with different beta
            overfit_idx = -1
            select_index = -1
            for idx in candidate_indices:
                if overfit_idx == -1 and  torch.norm(sample_list[idx].beta - seq_list[i].beta) > 1e-6:
                    overfit_idx = idx
                if select_index == -1 and torch.norm(sample_list[idx].beta - seq_list[i].beta) < 1e-6:
                    select_index = idx
                if overfit_idx != -1 and select_index != -1:
                    break
            sub_sample_index[i].append(select_index)
            # test beta
            delta_beta = sample_list[select_index].beta - seq_list[i].beta
            if torch.norm(delta_beta) > 1e-6:
                overfit = True 
            else:
                overfit = False
            
            sub_sample_list[i].append(sample_list[select_index])
            seq_list[i].add_sample(sample_list[select_index])

            if k == k_max - 1:
                seq_list[i].del_sample()    
                train_seq = copy.deepcopy(seq_list[i])
                train_seq.add_sample(sample_list[select_index])
                train_seq.get_input(query_index=-1)
                # compute loss and hessian
                train_loss, train_hessian = evaluate_loss(model, [train_seq], compute_hessian=True)

                # generate test samples
                test_seq_list = []
                for j in range(1):
                    sub_test_seq = copy.deepcopy(seq_list[i])
                    sub_test_seq.add_sample(sample_list[select_index])
                    new_x = torch.randn(n_dims).to(device)
                    new_y = (new_x @ seq_list[i].beta).squeeze(0)
                    query_sample = Sample(new_x, new_y, -1, seq_list[i].beta)
                    sub_test_seq.get_input(query_sample=query_sample)
                    test_seq_list.append(sub_test_seq)

                # test_seq = copy.deepcopy(seq_list[i])
                # test_seq.add_sample(sample_list[select_index])
                # test_seq.get_input(last_prompt=True)
                test_loss, test_hessian = evaluate_loss(model, test_seq_list, compute_hessian=True)

                gap = test_loss - train_loss

                print(f"train loss {train_loss}, test loss {test_loss}, gap {gap}")
                print(f"train hessian {train_hessian}, test hessian {test_hessian}")
                dist = {
                    "k": k,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "gap": gap,
                    "train_hessian": train_hessian,
                    "test_hessian": test_hessian
                }
                generalize_sample.append(dist)
                #overfit_sample.append(dist)
                # if overfit:
                #     overfit_sample.append(dist)
                # else:
                #     generalize_sample.append(dist)
                train_seq = copy.deepcopy(seq_list[i])
                train_seq.add_sample(sample_list[overfit_idx])
                train_seq.get_input(query_index=-2)
                # compute loss and hessian
                train_loss, train_hessian = evaluate_loss(model, [train_seq], compute_hessian=True)

                # generate test samples
                test_seq_list = []
                for j in range(1):
                    sub_test_seq = copy.deepcopy(seq_list[i])
                    sub_test_seq.add_sample(sample_list[overfit_idx])
                    new_x = torch.randn(n_dims).to(device)
                    new_y = (new_x @ beta[i]).squeeze(0)
                    query_sample = Sample(new_x, new_y, -1, beta[i])
                    sub_test_seq.get_input(query_sample=query_sample)
                    test_seq_list.append(sub_test_seq)

                # test_seq = copy.deepcopy(seq_list[i])
                # test_seq.add_sample(sample_list[overfit_idx])
                # test_seq.get_input(last_prompt=True)
                test_loss, test_hessian = evaluate_loss(model, test_seq_list, compute_hessian=True)

                gap = test_loss - train_loss

                print(f"train loss {train_loss}, test loss {test_loss}, gap {gap}")
                print(f"train hessian {train_hessian}, test hessian {test_hessian}")
                dist = {
                    "k": k,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "gap": gap,
                    "train_hessian": train_hessian,
                    "test_hessian": test_hessian
                }
                overfit_sample.append(dist)
        # if k in set_size_list:
        #     # get the input tensor
        #     xs, ys = sequence_to_tensor(seq_list)
        #     with torch.no_grad():
        #         pred = model(xs, ys)
        #     metric = task.get_metric()
        #     loss = metric(pred, ys).cpu().numpy()
        #     loss_query = loss.mean(axis=0)[-1]
        #     #print(loss_query)
        #     loss_list.append(loss_query)

    return loss_list, overfit_sample, generalize_sample

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
    b_size = 1
    seq_list = seq_list[:b_size]
    for k in range(1, max_points+1):
        for i in range(b_size):
            sub_seq_list = []
            i_seq = copy.deepcopy(seq_list[i])
            i_seq.add_sample(sample_list[i])
            i_seq.get_input(last_prompt=True)
            #sub_seq.get_input(query_range=n_val)
            xs, ys = sequence_to_tensor([i_seq])
            start_grad = time.time()
            pred_0, embeds_0 = model.forward_with_embeds(xs, ys)
            grad = torch.autograd.grad(pred_0[0, -1], embeds_0, retain_graph=True, create_graph=True)[0]
            grad = grad
            grad_0 = grad
            end_grad = time.time()

            metric = task.get_metric()

            delta_x = []
            for j in range(n):
                if j in sub_sample_index[i]:
                    continue
                sub_seq = copy.deepcopy(seq_list[i])

                sub_seq.add_sample(sample_list[j])
                #sub_seq.get_input(last_prompt=True)
                sub_seq.get_input()

                sub_seq_list.append(sub_seq)

            sub_xs, sub_ys = sequence_to_tensor(sub_seq_list)
            embeds = model.embed(sub_xs, sub_ys)
            delta_embeds = embeds - embeds_0

            delta_term = torch.sum(grad_0 * delta_embeds)

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


def fs_select_hessian(model, xs, ys, beta, sample_list, b_size, n_labeled, set_size_list, use_approx=True):
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

n_classes = 2
#anchor_points = torch.randn(n_classes, n_dims).to(device)
anchor_points = generate_orthogonal_matrix(n_classes, n_dims).to(device)
beta = torch.randn(b_size, n_dims, 1).to(device)
alpha = 1
for i in range(beta.shape[0]):
    i_class = torch.randint(0, n_classes, (1,))
    beta[i] = ((1-alpha)*beta[i] + anchor_points[i_class].T)
runs = 30
for run in tqdm(range(runs)):
    task = task_sampler()
    xs = generate_synthetic_data(num_sequences=b_size, n_points=n_total, x_dim=n_dims, diff_diftribution=False, alpha=0.0)
    xs = xs.to(device)
    #n_classes = b_size // 10
    

    ys = (xs @ beta)[:, :, 0]
    #print(xs)
    #ys += torch.randn_like(ys).to(device) * noise
    #print(ys.shape)
    labels = ys.clone()
    #ys[:, n_labeled:] = torch.zeros_like(ys[:, n_labeled:])
    #print(labels[0])
    #print(ys[0])
    with torch.no_grad():
        embedding = model.encoder(xs, ys)
    print(embedding.shape)

    # split sequences
    sample_list = []
    for i in range(xs.shape[0]):
        # -1 for remaining the last sample as query
        for j in range(xs.shape[1]-1):
            s = Sample(xs[i, j], ys[i, j], i, beta[i])
            sample_list.append(s)
    
    #print(sample_list)

    set_size_list = [1,2,3,4,5,6,7,8,9,10]
    #print(xs)
    #print(ys)
    pred_full_label, loss_full_label = predict_full_label(model, xs, ys=labels, labels=labels)
    loss_full_label_list.append(loss_full_label)
    
    # loss_fs_estimate = fs_select(model, xs, ys, beta, sample_list, b_size, n_labeled, set_size_list, use_approx=True)
    # #loss_fs_estimate = fs_select_debug2(model, xs, ys, beta, sample_list, b_size, n_labeled, set_size_list)
    # print(loss_fs_estimate)
    # loss_fs_estimate_list.append(loss_fs_estimate)
    loss_fs_inference, overfit_sample, generalize_sample = fs_inference_select(model, xs, ys, beta, sample_list, b_size, n_labeled, set_size_list)
    # print("Overfit samples: ")
    # for s in overfit_sample:
    #     print(s)
    # print("Generalize samples: ")
    # for s in generalize_sample:
    #     print(s)
    overfit_file = 'overfit_samples.json'
    generalize_file = 'generalize_samples.json'
    if os.path.exists(overfit_file):
        with open(overfit_file, 'r') as f:
            existing_overfit_samples = json.load(f)
    else:
        existing_overfit_samples = []
    if os.path.exists(generalize_file):
        with open(generalize_file, 'r') as f:
            existing_generalize_samples = json.load(f)
    else:
        existing_generalize_samples = []
    # Append new samples to existing ones
    #existing_overfit_samples = []
    #existing_generalize_samples = []
    existing_overfit_samples.extend(overfit_sample)
    existing_generalize_samples.extend(generalize_sample)
    # Save the combined list back to the file
    with open(overfit_file, 'w') as f:
        json.dump(existing_overfit_samples, f, indent=4)
    with open(generalize_file, 'w') as f:
        json.dump(existing_generalize_samples, f, indent=4)



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

# plt.plot(x, np.mean(loss_random_list, axis=0), lw=2, label="random")
# plt.fill_between(x, np.mean(loss_random_list, axis=0)-np.std(loss_random_list, axis=0), np.mean(loss_random_list, axis=0)+np.std(loss_random_list, axis=0), alpha=0.2)
#plt.plot(x, np.mean(loss_beta_list, axis=0), lw=2, label="beta")
#plt.fill_between(x, np.mean(loss_beta_list, axis=0)-np.std(loss_beta_list, axis=0), np.mean(loss_beta_list, axis=0)+np.std(loss_beta_list, axis=0), alpha=0.2)
#plt.plot(x, np.mean(loss_contrastive_list, axis=0), lw=2, label="contrastive")
#plt.fill_between(x, np.mean(loss_contrastive_list, axis=0)-np.std(loss_contrastive_list, axis=0), np.mean(loss_contrastive_list, axis=0)+np.std(loss_contrastive_list, axis=0), alpha=0.2)

plt.plot(x, np.mean(loss_fs_inference_list, axis=0), lw=2, label="inference")
plt.fill_between(x, np.mean(loss_fs_inference_list, axis=0)-np.std(loss_fs_inference_list, axis=0), np.mean(loss_fs_inference_list, axis=0)+np.std(loss_fs_inference_list, axis=0), alpha=0.2)
# plt.plot(x, np.mean(loss_fs_estimate_list, axis=0), lw=2, label="estimate")
# plt.fill_between(x, np.mean(loss_fs_estimate_list, axis=0)-np.std(loss_fs_estimate_list, axis=0), np.mean(loss_fs_estimate_list, axis=0)+np.std(loss_fs_estimate_list, axis=0), alpha=0.2)

#np.savez("./results/LR.npz", x=x, loss_full_label_list=loss_full_label_list, loss_fs_estimate_list=loss_fs_estimate_list, loss_fs_inference_list=loss_fs_inference_list, loss_random_list=loss_random_list)

#plt.plot(loss_full_label, lw=2, label="Full label")
#plt.plot(loss_unlabeled_once, lw=2, label="Unlabeled once")
#plt.plot(loss_unlabeled_iter, lw=2, label="Unlabeled iter")
#plt.plot(loss_unlabeled_stepbystep, lw=2, label="Unlabeled step by step")
plt.xlabel("# in-context examples")
plt.ylabel("squared error")
plt.legend()
plt.savefig("LR.png")