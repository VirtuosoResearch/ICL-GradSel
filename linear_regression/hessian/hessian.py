#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICUnp.linalgR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*

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

#from pyhessian.utils import group_product, group_add, normalization, get_params_grad, hessian_vector_product, orthnormal

from scipy.stats import norm
from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from scipy.stats import entropy
from scipy.stats import pearsonr


def load_batch_func(batch, device='cpu'):
    batch = batch[0].to(device)
    inputs = batch[:, :-1]
    targets = batch
    batch_size = batch.shape[0]
    return inputs, targets, batch_size

def filter_eigenvalues(eigen_list, weight_list, threshold=None):
    filtered_eigen = []
    filtered_weight = []
    #print(np.max(weight_list))
    for eig, w in zip(eigen_list, weight_list):
        if threshold is not None:
            if eig >= threshold and w >= 1e-7:
                filtered_eigen.append(eig)
                filtered_weight.append(w)
        else:
            if w >= 1e-10:
                filtered_eigen.append(eig)
                filtered_weight.append(w)
    #print(filtered_eigen)
    return filtered_eigen, filtered_weight

def renormalize_weights(filtered_weight, epsilon=1e-12):
    total = sum(filtered_weight)
    if total > 0:
        renormalized_weight = [w / (total + epsilon) for w in filtered_weight]
    else:
        # Handle case where all weights are zero
        renormalized_weight = [0.0 for _ in filtered_weight]
    return renormalized_weight

def construct_spectral_density(flat_eigen, flat_weight, lambdas, sigma=0.1):
    density = np.zeros_like(lambdas)
    for eig, w in zip(flat_eigen, flat_weight):
        density += w * norm.pdf(lambdas, loc=eig, scale=sigma)
    
    # Normalize the density
    density_sum = np.sum(density) * (lambdas[1] - lambdas[0])
    density /= density_sum + 1e-12  # Avoid division by zero
    return density

def flat_list(original_list):
    return [float(e) for run in original_list for e in run]

def list_aggregate(list_a, list_b, batch_size):
    if len(list_a) == 0:
        list_a = [float(b) * batch_size for b in list_b]
    else:
        list_a = [float(a + b * batch_size) for a, b in zip(list_a, list_b)]
    return list_a

def compute_spectral_divergences(
    eigen_list_1, weight_list_1, 
    eigen_list_2, weight_list_2,
    measure = 'kl',
    sigma=0.1, grid_size=100
):
    # Step 1: Validate Inputs
    if not (len(eigen_list_1) == len(weight_list_1)):
        raise ValueError("eigen_list_1 and weight_list_1 must have the same number of SLQ runs.")
    if not (len(eigen_list_2) == len(weight_list_2)):
        raise ValueError("eigen_list_2 and weight_list_2 must have the same number of SLQ runs.")
    
    # Step 2: Flatten the eigenvalues and weights
    
    # Step 3: Determine the global min and max eigenvalues for the common grid
    all_eigen = eigen_list_1 + eigen_list_2
    lambda_min = min(all_eigen) - 1.0  # Padding to ensure coverage
    lambda_max = max(all_eigen) + 1.0
    
    # Step 4: Create a common lambda grid
    common_lambdas = np.linspace(lambda_min, lambda_max, grid_size)
    delta_lambda = common_lambdas[1] - common_lambdas[0]
    
    # Step 5: Construct spectral densities using Gaussian kernels
    def construct_density(eigen_list, weight_list, lambdas, sigma):
        density = np.zeros_like(lambdas)
        for eig, w in zip(eigen_list, weight_list):
            density += w * norm.pdf(lambdas, loc=eig, scale=sigma)
        # Normalize the density
        density_sum = np.sum(density) * (lambdas[1] - lambdas[0])
        density /= density_sum + 1e-12  # Avoid division by zero
        return density
    
    density_1 = construct_density(eigen_list_1, weight_list_1, common_lambdas, sigma)
    density_2 = construct_density(eigen_list_2, weight_list_2, common_lambdas, sigma)
    
    # Step 6: Compute KL Divergence (D_KL(P || Q))
    # To ensure numerical stability, add a small epsilon where necessary
    #print(density_1)
    #print(density_2)
    epsilon = 1e-12
    p = density_1 + epsilon
    q = density_2 + epsilon
    if measure == 'kl':
        divergence = np.sum(p * np.log(p / q)) * delta_lambda
    elif measure == 'js':
        # Step 7: Compute Jensen-Shannon Divergence (D_JS(P || Q))
        m = 0.5 * (p + q)
        d_kl_p_m = np.sum(p * np.log(p / m)) * delta_lambda
        d_kl_q_m = np.sum(q * np.log(q / m)) * delta_lambda
        divergence = 0.5 * (d_kl_p_m + d_kl_q_m)
        
    return divergence, common_lambdas


def create_spectral_density(eigen_list_full, weight_list_full, sigma=0.1, grid_size=1000):
    eigen_values = [eig for run in eigen_list_full for eig in run]
    weights = [w for run in weight_list_full for w in run]
    lambda_min = min(eigen_values) - 1
    lambda_max = max(eigen_values) + 1
    print("min, max: ", lambda_min, lambda_max)
    lambdas = np.linspace(lambda_min, lambda_max, grid_size)
    density = np.zeros_like(lambdas)
    for eig, w in zip(eigen_values, weights):
        density += w * norm.pdf(lambdas, loc=eig, scale=sigma)
    density /= np.sum(density) * (lambdas[1] - lambdas[0])
    return lambdas, density

def kl_divergence(p, q, lambdas):
    epsilon = 1e-12
    p = p + epsilon
    q = q + epsilon
    return np.sum(p * np.log(p / q)) * (lambdas[1] - lambdas[0])

def js_divergence(p, q, lambdas):
    m = 0.5 * (p + q)
    epsilon = 1e-12
    p = p + epsilon
    q = q + epsilon
    m = m + epsilon
    D_KL_P_M = np.sum(p * np.log(p / m)) * (lambdas[1] - lambdas[0])
    D_KL_Q_M = np.sum(q * np.log(q / m)) * (lambdas[1] - lambdas[0])
    D_JS = 0.5 * (D_KL_P_M + D_KL_Q_M)
    return D_JS

def total_variation(p, q, lambdas):
    return 0.5 * np.sum(np.abs(p - q)) * (lambdas[1] - lambdas[0])


def normalization_(vs, epsilon=1e-6):
    """
    normalization of a list of vectors
    return: normalized vectors v
    """
    """
    norms = [torch.sum(v*v) for v in vs]
    norms = [(norm**0.5).cpu().item() for norm in norms]
    vs = [vi / (norms[i] + 1e-6) for (i, vi) in enumerate(vs)]
    return vs
    """
    return [v / (torch.norm(v) + epsilon) for v in vs]

def orthnormal_(ws, vs_list):
    """
    make vector w orthogonal to each vector in v_list.
    afterwards, normalize the output w
    """
    for vs in vs_list:
        for w, v in zip(ws, vs):
            w.data.add_(-v*(torch.sum(w*v)))
    return normalization(ws)


def sqrt_with_neg_handling(arr):
    result = np.where(arr < 0, 0, np.sqrt(arr))
    return result

class Hessian_Calculator():
    def __init__(self, model, loss_fn, p, dataloader=None, external_load_batch_func=None, device='cpu'):
        self.p = p
        self.num_classes = p+2
        self.model = model.eval()  # make model is in evaluation model
        self.loss_fn = loss_fn
        self.aggregate_method = 'mean'

        if external_load_batch_func is not None:
            self.load_batch_func = external_load_batch_func
        else:
            self.load_batch_func = load_batch_func
        
        self.dataloader = dataloader
        self.device = device

        # get splited weights
        self.layers = get_layers(self.model)
        self.weights, self.layer_names, self.grouped_layer_weights, self.grouped_layer_names = get_grouped_layer_weights(self.model)
        print(self.layer_names)
        #print(self.weights[0].grad)

        self.hessian_norms = []
        self.layer_trace = []
        self.lambda_max_list = []

        
        self.spectrum_divergence_list = []
        self.spectrum_entropy_list = []
        self.weighted_entropy_list = []
        self.centroid_list = []
        self.spread_list = []
        self.effective_rank_list = []
        self.stable_rank_list = []
        self.lambda_max_list = []
        self.condition_list = []

        self.max_eigenvector_1 = None
        self.lambda_1 = 0

        self.noise_sensitivity = 0

        self.sample_layer = ['head.weight']
        self.total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def list_aggregate(self, list_a, list_b, batch_size=0, aggregate_method='mean'):
        if aggregate_method == 'mean':
            if len(list_a) == 0:
                list_result = [float(b) * batch_size for b in list_b]
            else:
                list_result = [float(a + b * batch_size) for a, b in zip(list_a, list_b)]
            return list_result

        elif aggregate_method == 'max':
            if len(list_a) == 0:
                list_result = list_b
            else:
                list_result = [max(a, b) for a, b in zip(list_a, list_b)]
            # Return element-wise maximum from list_a and list_b
            return list_result

        elif aggregate_method == 'min':
            if len(list_a) == 0:
                list_result = list_b
            else:
                list_result = [min(a, b) for a, b in zip(list_a, list_b)]
            # Return element-wise minimum from list_a and list_b
            return list_result

        else:
            raise ValueError(f"Unknown method: {aggregate_method}")
    
    def group_div_const(self, X, c):
        return [x/c for x in X]
    
    def collect(self, train_num):
        self.trace_based_measure()
        #self.batch_collect()
        self.batch_aggregate(train_num)

    def trace_based_measure(self, device = "cpu", maxIter=100, tol=1e-3):
        self.layer_trace = []
        self.hessian_norms = []
        with sdpa_kernel(SDPBackend.MATH):
            device = self.device
            model = self.model
            loss_fn = self.loss_fn
            
            for batch in self.dataloader:
                # Specific data process, in order to fit the loss input
                data, target, batch_size = self.load_batch_func(batch, device)
                output = model(data)
                loss = loss_fn(output, target, 'none')
                model.zero_grad()

                # Trace of Hessian. Save in ./results/hessian
                grouped_layer_trace = self.compute_hessians_trace(model, loss.mean(), batch_size, device)
                self.layer_trace = self.list_aggregate(self.layer_trace, grouped_layer_trace, batch_size, aggregate_method='mean')
                #print("method 0: ", self.layer_trace)

                # stable rank
                stable_rank_list, lambda_max_list = self.compute_stable_rank(loss.mean(), batch_size)
                #print("method 1: ", lambda_max_list)
                self.stable_rank_list = self.list_aggregate(self.stable_rank_list, stable_rank_list, batch_size, aggregate_method='mean')
                self.lambda_max_list = self.list_aggregate(self.lambda_max_list, lambda_max_list, batch_size, aggregate_method='mean')

                # max eigenvalue
                #self.compute_eigenvalues(loss.mean(), batch_size)

                # Hessian bound
                layer_hessian_quantities = self.compute_generalization_bound(model, loss.mean(), self.device)
                self.hessian_norms = self.list_aggregate(self.hessian_norms, layer_hessian_quantities, batch_size, aggregate_method='max')

                #self.batch_spectral_density(loss.mean(), batch_size)
                self.compute_sensitivity(loss.mean(), data, target, batch_size)
                
            # Compute the spectral density
            #self.spectral_density(n_iter=100, n_v=5)

            #self.compute_generalization_bound_2(model)

            for param in model.parameters():
                if param.grad is not None:
                    param.grad = None   

    def batch_collect(self):
        self.layer_trace = []
        with sdpa_kernel(SDPBackend.MATH):
            device = self.device
            model = self.model
            loss_fn = self.loss_fn
            for batch in self.dataloader:
                # Specific data process, in order to fit the loss input
                #batch = batch[0].to(device)
                #data = batch[:, :-1]
                #target = batch
                #batch_size = batch.shape[0]
                data, target, batch_size = self.load_batch_func(batch, device)
                output = model(data)
                loss = loss_fn(output, target, 'none')
                model.zero_grad()
                # item_1: f(x)f(x)^T, item_2: diag(p) - pp^T. Save in ./results/prob
                #self.compute_item(model, data, target, batch_size)

                # sensitivity: inject noise to input, and estimate the difference of loss. Save in ./results/input
                #self.compute_sensitivity(model, loss_fn, data, target, batch_size)

                # The ratio of the first singular value and the second singular value of loss. Save in ./results/distance
                #self.compute_singular_ratio(model, loss.mean(), batch_size)

                # Compute the trace of Hessian of weight decay
                #self.compute_wd_hessians_trace()

                # Compute the eigenvalues
                #self.compute_eigenvalues(model, loss.mean(), batch_size)

                # Trace of Hessian. Save in ./results/hessian
                self.compute_hessians_trace(model, loss.mean(), batch_size, device)

                #self.compute_stable_rank_2(model, loss.mean(), batch_size, device)
                self.compute_stable_rank_per_group(loss.mean(), batch_size)

                # Hessian bound
                #self.compute_generalization_bound(model, loss.mean(), self.device)

                #self.batch_spectral_density(loss.mean(), batch_size)
                
            # Compute the spectral density
            #self.spectral_density(n_iter=100, n_v=5)

            self.compute_generalization_bound_2(model)

            for param in model.parameters():
                if param.grad is not None:
                    param.grad = None   
    
    def batch_aggregate(self, train_num):
        # trace based measure
        self.layer_trace = self.group_div_const(self.layer_trace, train_num)
        #self.stable_rank_list = self.group_div_const(self.stable_rank_list, train_num)
        self.lambda_max_list = self.group_div_const(self.lambda_max_list, train_num)


        #self.stable_rank_per_group = self.group_div_const(self.stable_rank_per_group, train_num)
        #print(self.layer_trace)
        #print(self.effective_rank_list)
        #self.shapescale = (self.layer_trace*np.array(self.effective_rank_list)).tolist()
        #print(self.shapescale)
        #print(self.layer_trace)
        #for i in range(len(self.layer_trace)):
        #    self.layer_trace[i]  = float(self.layer_trace[i])
        #self.spectral_density = self.group_div_const(self.spectral_density, train_num)
        #print(self.spectrum_divergence_list)
        #print(self.spectrum_entropy_list)
        #self.spectrum_divergence_list = self.group_div_const(self.spectrum_divergence_list, train_num)
        #self.centroid_list = self.group_div_const(self.centroid_list, train_num)
        #self.spread_list = self.group_div_const(self.spread_list, train_num)
        #self.weighted_entropy_list = self.group_div_const(self.weighted_entropy_list, train_num)
        #self.spectrum_entropy_list = self.group_div_const(self.spectrum_entropy_list, train_num)

        #self.item_1 /= train_num
        #self.item_2 /= train_num
        #self.cosine_similarities = np.mean(self.cosine_similarities / train_num)
        #self.grad_norms = np.mean(self.grad_norms / train_num)
        #self.wd_grad_norms = np.mean(self.wd_grad_norms / train_num)
        #self.l2_distance = np.mean(self.l2_distance / train_num)
        #print(self.cosine_similarities)
        #self.trace = self.layer_trace / train_num
        #print(self.trace)
        #self.lambda_1 = self.lambda_1 / train_num
        #self.condition = self.lambda_1 / self.trace
        #print("condition: ", self.condition)
        """
        self.lambda_1 = self.lambda_1 / train_num
        self.lambda_2 = self.lambda_2 / train_num
        self.lambda_1 = np.mean(self.lambda_1)
        self.lambda_2 = np.mean(self.lambda_2)
        #self.lambda_n = np.mean(self.lambda_n/train_num)
        self.condition = self.lambda_1 / self.lambda_2
        eigen_distance = []
        for i in range(len(self.max_eigenvector_1)):
            eigen_distance.append(torch.norm(self.max_eigenvector_1[i] - self.max_eigenvector_2[i]).item())
        self.max_eigenvector_1 = [eigenvector / train_num for eigenvector in self.max_eigenvector_1]
        self.max_eigenvector_2 = [eigenvector / train_num for eigenvector in self.max_eigenvector_2]
        for i in range(len(self.max_eigenvector_1)):
            eigen_distance.append(torch.norm(self.max_eigenvector_1[i] - self.max_eigenvector_2[i]).item())
        #self.condition = np.mean(self.lambda_1 / ((self.trace - self.lambda_1)/(self.trace_num-1)))
        self.eigenvector_distance = np.mean(eigen_distance)
        """
        #self.wd_trace = np.mean(self.layer_wd_trace)
        #self.trace = np.mean(self.trace)

        self.noise_sensitivity /= train_num
        #print(self.noise_sensitivity)
        
        #self.loss_singularvector_distance = np.mean(self.loss_singularvector_distance / train_num)
        #self.loss_singularvalue_distance = np.mean(self.loss_singularvalue_distance / train_num)

        hessian_quantities = np.sum(sqrt_with_neg_handling(np.array(self.hessian_norms))) / np.sqrt(train_num)
        self.train_hessianmeasurement = (hessian_quantities).item()


    def compute_hessians_trace(self, model, loss, batch_size, aggregate_method='mean', device = "cpu", maxIter=100, tol=1e-8):
        # Get parameters and gradients of corresponding layer
        grouped_layer_trace = []
        for weights in self.grouped_layer_weights:
            gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)

            layer_traces = []
            trace_vhv = []
            trace = 0.
            # Start Iterations
            for _ in range(maxIter):
                vs = [torch.randint_like(weight, high=2) for weight in weights]
                    
                # generate Rademacher random variables
                for v in vs:
                    v[v == 0] = -1

                model.zero_grad()  
                Hvs = torch.autograd.grad(gradients, weights, grad_outputs=vs, retain_graph=True)
                tmp_layer_traces = np.array([torch.sum(Hv*v).cpu().item() for (Hv, v) in zip(Hvs, vs)])
                #tmp_layer_traces = sum([torch.sum(Hv*v).cpu().item() for (Hv, v) in zip(Hvs, vs)])

                layer_traces.append(tmp_layer_traces)
                trace_vhv.append(sum(tmp_layer_traces))

                if abs(np.mean(trace_vhv) - trace) / (abs(trace) + 1e-6) < tol:
                    break
                else:
                    trace = np.mean(trace_vhv)
            layer_trace = np.mean(np.array(layer_traces), axis=0)
            #grouped_layer_trace.append(np.sum(layer_trace, axis=0))
            grouped_layer_trace.append(trace)
        #print(grouped_layer_trace)
        return grouped_layer_trace
        #self.layer_trace = self.list_aggregate(self.layer_trace, grouped_layer_trace, batch_size, aggregate_method=aggregate_method)
        """
        if len(self.layer_trace) == 0:
            #avg_layer_trace = np.mean(np.array(layer_traces), axis=0) / trace_num
            self.layer_trace = grouped_layer_trace
        else:
            self.layer_trace = np.maximum(grouped_layer_trace, self.layer_trace).tolist()
        """
        #print(self.layer_trace)

    
    # only support top_n=1
    def compute_eigenvalues(self, loss, batch_size, top_n=1, maxIter=100, tol=1e-8):
        model = self.model
        #weights = self.weights
        
        topn_eigenvalues_list = []
        eigenvectors_list = []
        for weights in self.grouped_layer_weights:
            gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)

            topn_eigenvalues = []
            eigenvectors = []
            computed_dim = 0
            while computed_dim < top_n:
                
                eigenvalues = None
                vs = [torch.randn_like(weight) for weight in weights]  # generate random vector
                vs = normalization(vs)  # normalize the vector

                for _ in range(maxIter):
                    #vs = orthnormal(vs, eigenvectors)
                    #model.zero_grad()

                    Hvs = torch.autograd.grad(gradients, weights, grad_outputs=vs, retain_graph=True)
                    tmp_eigenvalues = sum([ torch.sum(Hv*v).cpu().item() for (Hv, v) in zip(Hvs, vs)])

                    vs = normalization(Hvs)

                    if eigenvalues == None:
                        eigenvalues = tmp_eigenvalues
                    else:
                        if abs(eigenvalues - tmp_eigenvalues) / (abs(eigenvalues) + 1e-8) < tol:
                            break
                        else:
                            eigenvalues = tmp_eigenvalues
                topn_eigenvalues.append(eigenvalues)
                eigenvectors.append(vs)
                computed_dim += 1
            topn_eigenvalues_list.append(topn_eigenvalues[0])
            eigenvectors_list.append(eigenvectors)
        """
        topn_eigenvalues_list = []
        eigenvectors_list = []

        for weights in self.grouped_layer_weights:
            gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)
            topn_eigenvalues = []
            eigenvectors = []
            computed_dim = 0

            while computed_dim < top_n:
                eigenvalue_estimate = None
                vs = [torch.randn_like(w) for w in weights]
                vs = normalization_list(vs)

                for _ in range(maxIter):
                    vs = orthnormal_list(vs, eigenvectors)
                    self.model.zero_grad()
                    Hvs = torch.autograd.grad(gradients, weights, grad_outputs=vs, retain_graph=True)
                    tmp_eigenvalue = sum(torch.sum(Hv * v).item() for Hv, v in zip(Hvs, vs))
                    vs = normalization_list(Hvs)

                    if eigenvalue_estimate is None:
                        eigenvalue_estimate = tmp_eigenvalue
                    else:
                        rel_change = abs(eigenvalue_estimate - tmp_eigenvalue) / (abs(eigenvalue_estimate) + 1e-8)
                        eigenvalue_estimate = tmp_eigenvalue
                        if rel_change < tol:
                            break

                topn_eigenvalues.append(eigenvalue_estimate)
                eigenvectors.append(vs)
                computed_dim += 1

            topn_eigenvalues_list.append(topn_eigenvalues[0])
            eigenvectors_list.append(eigenvectors)
        """
        #print(topn_eigenvalues_list)
        # Max eigenvalue. In process
        max_eigenvalue, max_eigenvector = compute_eigenvalue(self.model, loss, self.device, top_n=1)
        max_eigenvector = max_eigenvector[0]
        #print(max_eigenvector[0].shape)
        #block_sim_1 = F.cosine_similarity(max_eigenvector[0], torch.eye(max_eigenvector[0].shape[1]).to(device))
        #block_sim_2 = F.cosine_similarity(max_eigenvector[1], torch.eye(max_eigenvector[1].shape[1]).to(device))
        #block_sim_3 = F.cosine_similarity(max_eigenvector[2], torch.eye(max_eigenvector[2].shape[1]).to(device))
        #block_sim_1 = max_eigenvector[0][:, 0]
        #block_sim_2 = max_eigenvector[1][:, 0]
        #block_sim_3 = max_eigenvector[2][:, 0]
        #self.block_sim_1 += block_sim_1.mean()*batch_size
        #self.block_sim_2 += block_sim_2.mean()*batch_size
        #self.block_sim_3 += block_sim_3.mean()*batch_size
        #print(block_sim_1.shape)
        #print(block_sim_1, block_sim_2, block_sim_3)
        #min_eigenvalue, min_eigenvector = compute_eigenvalue(model, -loss, device, top_n=1)
        #print(max_eigenvalue, min_eigenvalue)
        
        #max_eigenvalue, max_eigenvector = compute_eigenvalue(model, loss, device, top_n=1)
        max_eigenvector_1 = max_eigenvector[0]
        #max_eigenvector_2 = max_eigenvector[1]
        if self.max_eigenvector_1 is None:
            self.max_eigenvector_1 = max_eigenvector_1 * batch_size
            #self.max_eigenvector_2 = max_eigenvector_2 * batch_size
        else:
            self.max_eigenvector_1 += max_eigenvector_1 * batch_size
            #self.max_eigenvector_2 += max_eigenvector_2 * batch_size
        lambda_1 = np.array(max_eigenvalue[0])
        #lambda_2 = np.array(max_eigenvalue[1])
        #print("method 2: ", topn_eigenvalues_list)
        self.lambda_1 += lambda_1 * batch_size
        #self.lambda_2 += lambda_2 * batch_size
        #print(layer_trace, lambda_1)
        #print(estimate_trace, estimate_eigen)
        #self.lambda_n += lambda_n * batch_size

    def compute_generalization_bound(self, model, loss, device="cpu", state_dict = None):
        # Get parameters and gradients of corresponding layer
        weights = self.weights
        gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)
        
        vs = []
        for name, module in self.layers.items():
            weight = module.weight
            v = weight.detach().clone() - model.init_state[name+".weight"].to(weight.device)
            vs.append(v)

        model.zero_grad()    
        Hvs = torch.autograd.grad(gradients, weights, grad_outputs=vs, retain_graph=True)

        layer_hessian_quantities = [torch.sum(Hv*v).cpu().item() for (Hv, v) in zip(Hvs, vs)]
        
        layer_hessian_quantities = np.array(layer_hessian_quantities)
        
        return layer_hessian_quantities
        #layer_hessian_quantities = np.sum(layer_hessian_quantities) / np.sqrt(config.train.hessian_log_size)
        if len(self.hessian_norms) == 0:
            self.hessian_norms = layer_hessian_quantities
        else:
            self.hessian_norms = np.maximum(self.hessian_norms, layer_hessian_quantities)

    def compute_sensitivity(self, loss, data, target, batch_size):
        # noise_sensitivity: Estimate the sensetivity of input. Save in ./results/input
        for i in range(50):
            #noisy_output, noise_norm = model.add_noise_forward(data)
            
            #noise_sensitivity = torch.norm(noisy_output[:, -1] - output[:, -1]) / noise_norm
            #noisy_loss = F.cross_entropy(noisy_output[:, -1], target[:, -1], reduction='none')
            #noise_sensitivity = (noisy_loss - loss) / noise_norm
            noisy_output_1, noisy_output_2, noise_norm = self.model.add_bi_noise_forward(data)
            noisy_loss_1 = F.cross_entropy(noisy_output_1[:, -1], target[:, -1], reduction='none')
            noisy_loss_2 = F.cross_entropy(noisy_output_2[:, -1], target[:, -1], reduction='none')
            noise_sensitivity = (noisy_loss_1 + noisy_loss_2 - 2*loss)
            #noise_sensitivity = (noisy_output_1 + noisy_output_2 - output)[:, -1]

        self.noise_sensitivity += noise_sensitivity.mean().item() * batch_size

    # copy from pyhessian
    def dataloader_hv_product(self, v, weights):

        device = self.device
        num_data = 0  # count the number of datum points in the dataloader

        THv = [torch.zeros(weight.size()).to(device) for weight in weights]  # accumulate result
        for batch in self.dataloader:
            self.model.zero_grad()
            # Specific data process, in order to fit the loss input
            data, target, batch_size = self.load_batch_func(batch, device)
            output = self.model(data)
            loss = self.loss_fn(output, target, 'mean')
            loss.backward(create_graph=True)
            #params, gradsH = get_params_grad(self.model) # TODO check
            gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)
            self.model.zero_grad()
            Hv = torch.autograd.grad(gradients,
                                     weights,
                                     grad_outputs=v,
                                     only_inputs=True,
                                     retain_graph=False)
            THv = [
                THv1 + Hv1 * float(batch_size) + 0.
                for THv1, Hv1 in zip(THv, Hv)
            ]
            num_data += float(batch_size)

        THv = [THv1 / float(num_data) for THv1 in THv]

        return THv
        eigenvalue = group_product(THv, v).cpu().item()
        return eigenvalue, THv
    

    def compute_generalization_bound_2(self, model, device="cpu", state_dict = None):
        # Get parameters and gradients of corresponding layer
        weights = self.weights
        #gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)
        
        vs = []
        for name, module in self.layers.items():
            weight = module.weight
            v = weight.detach().clone() - model.init_state[name+".weight"].to(weight.device)
            vs.append(v)

        model.zero_grad()    
        #Hvs = torch.autograd.grad(gradients, weights, grad_outputs=vs, retain_graph=True)
        Hvs = self.dataloader_hv_product(vs, weights)

        layer_hessian_quantities = [torch.sum(Hv*v).cpu().item() for (Hv, v) in zip(Hvs, vs)]
        
        layer_hessian_quantities = np.array(layer_hessian_quantities)
        
        #layer_hessian_quantities = np.sum(layer_hessian_quantities) / np.sqrt(config.train.hessian_log_size)
        if len(self.hessian_norms) == 0:
            self.hessian_norms = layer_hessian_quantities
        else:
            self.hessian_norms = np.maximum(self.hessian_norms, layer_hessian_quantities)

    def compute_stable_rank_2(self, model, loss, batch_size, device = "cpu", maxIter=100, tol=1e-3):
        # Get parameters and gradients of corresponding layer
        grouped_layer_trace = []
        for weights in self.grouped_layer_weights:
            gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)

            layer_traces = []
            trace_vhv = []
            trace = 0.
            # Start Iterations
            for _ in range(maxIter):
                vs = [torch.randint_like(weight, high=2) for weight in weights]
                    
                # generate Rademacher random variables
                for v in vs:
                    v[v == 0] = -1

                model.zero_grad()  
                Hvs = torch.autograd.grad(gradients, weights, grad_outputs=vs, retain_graph=True)
                HHv = torch.autograd.grad(gradients, weights, grad_outputs=Hvs, retain_graph=True)
                tmp_layer_traces = np.array([torch.sum(Hv*v).cpu().item() for (Hv, v) in zip(Hvs, vs)])
                #tmp_layer_traces = sum([torch.sum(Hv*v).cpu().item() for (Hv, v) in zip(Hvs, vs)])

                layer_traces.append(tmp_layer_traces)
                trace_vhv.append(sum(tmp_layer_traces))

                if abs(np.mean(trace_vhv) - trace) / (abs(trace) + 1e-6) < tol:
                    break
                else:
                    trace = np.mean(trace_vhv)
            layer_trace = np.mean(np.array(layer_traces), axis=0)
            #grouped_layer_trace.append(np.sum(layer_trace, axis=0))
            grouped_layer_trace.append(trace)
        #print(grouped_layer_trace)
        
        if len(self.layer_trace) == 0:
            #avg_layer_trace = np.mean(np.array(layer_traces), axis=0) / trace_num
            self.layer_trace = grouped_layer_trace
        else:
            self.layer_trace = np.maximum(grouped_layer_trace, self.layer_trace).tolist()

    def approximate_trace_h2(self, loss, weights, maxIter=100, tol=1e-3):
        """
        Approximates trace(H^2) for the given group of weights, 
        where H is the Hessian of 'loss' wrt 'weights'.
        
        Returns: A float approximation of trace(H^2) for that group.
        """
        # 1. First-order gradient wrt 'weights'
        gradients = torch.autograd.grad(loss, weights, 
                                        retain_graph=True, 
                                        create_graph=True)
        
        trace_estimates = []
        trace_running_list = []
        prev_trace_mean = 0.0
        
        for _ in range(maxIter):
            # Generate Rademacher random vectors 'v' for each tensor
            vs = [torch.randint_like(w, high=2) for w in weights]
            for v in vs:
                v[v == 0] = -1  # convert {0,1} to {-1, +1}

            # Step 1: compute H v
            Hv = torch.autograd.grad(gradients, weights, 
                                     grad_outputs=vs,
                                     retain_graph=True)
            
            # Step 2: compute H(Hv) = H^2 v
            HHv = torch.autograd.grad(gradients, weights, 
                                      grad_outputs=Hv,
                                      #grad_outputs=vs,
                                      retain_graph=True)
            
            # Now estimate v^T (H^2 v) = sum( (HHv_i * v_i).sum() ) for each i
            # Each i corresponds to a parameter tensor in this group
            vHv_sum = 0.0
            for (HHv_i, v_i) in zip(HHv, vs):
                vHv_sum += torch.sum(HHv_i * v_i).item()
            
            trace_estimates.append(vHv_sum)
            trace_running_list.append(vHv_sum)
            
            # Early stopping check
            new_mean = np.mean(trace_running_list)
            rel_change = abs(new_mean - prev_trace_mean) / (abs(prev_trace_mean) + 1e-6)
            if rel_change < tol:
                break
            prev_trace_mean = new_mean

        # Final approximate trace(H^2) is the mean of all samples
        return np.mean(trace_estimates)

    def approximate_lambda_max(self, loss, weights, power_iter=20):
        """
        Approximates the largest eigenvalue of the Hessian wrt 'weights'
        using power iteration. 
        Only works well if the Hessian is PSD.
        
        Returns: float lambda_max
        """
        # 1. Compute first-order gradient
        gradients = torch.autograd.grad(loss, weights, 
                                        retain_graph=True, 
                                        create_graph=True)
        
        # Initialize a random vector
        # (We'll flatten all group parameters into a single vector approach)
        # But we can keep it separate for each param tensor if we want.
        
        # For simplicity, let's just create a single flattened vector 'v'.
        # We'll need to define our own 'apply_H' that does H*v across the group.
        
        # Flatten each parameter for power iteration
        vecs = []
        shapes = []
        for w in weights:
            shapes.append(w.shape)
            vecs.append(torch.randn_like(w).flatten())
            #vecs.append(torch.randn_like(w))
        v = torch.cat(vecs).detach()
        #vs = [torch.randn_like(weight) for weight in weights]
        #v = normalization(vs)
        
        # Function to compute H*v for the entire group in a flattened manner
        def apply_H(v_flat):
            # Reshape v_flat back into each param shape
            offset = 0
            vs_unflat = []
            for w, shp in zip(weights, shapes):
                size = w.numel()
                vs_unflat.append(v_flat[offset:offset+size].view(shp))
                offset += size
            
            # Now compute Hessian-vector product: H vs_unflat
            Hv_unflat = torch.autograd.grad(gradients, weights,
                                            grad_outputs=vs_unflat,
                                            #grad_outputs=v,
                                            retain_graph=True)
            # Flatten the result
            Hv_list = []
            for Hv_i in Hv_unflat:
                Hv_list.append(Hv_i.flatten())
            return torch.cat(Hv_list)
        
        v_norm = v.norm(p=2)
        if v_norm < 1e-12:
            return 0.0
        v = v / v_norm
        
        # Power iteration
        for _ in range(power_iter):
            Hv = apply_H(v)
            norm_Hv = Hv.norm(p=2)
            if norm_Hv < 1e-8:
                return 0.0
            v = Hv / norm_Hv
        
        # Rayleigh quotient approximation for final eigenvalue
        Hv = apply_H(v)
        lambda_max_approx = torch.dot(v, Hv).item()
        return lambda_max_approx

    def compute_stable_rank(self, loss, batch_size, aggregate_method='mean'):
        """
        High-level function that:
        1) Computes the loss
        2) For each group:
           a) Approximates trace(H^2)
           b) Approximates largest eigenvalue
           c) stable_rank = trace(H^2) / (lambda_max^2)
        3) Stores results in self.stable_rank_per_group
        """
        # Compute the loss wrt all grouped weights (retain graph for second derivatives)
        
        stable_ranks = []
        lambda_max_list = []
        for weights in self.grouped_layer_weights:
            # 1. trace(H^2)
            trace_h2 = self.approximate_trace_h2(loss, weights)
            #print("method 1: ", trace_h2)
            # 2. largest eigenvalue (power iteration)
            lambda_max = self.approximate_lambda_max(loss, weights, power_iter=100)
            
            # 3. stable rank = trace(H^2) / (lambda_max^2)
            epsilon = 1e-12
            srank = trace_h2 / (lambda_max**2 + epsilon)
            
            stable_ranks.append(srank)
            lambda_max_list.append(lambda_max)
        
        #self.stable_rank_per_group = self.list_aggregate(self.stable_rank_per_group, stable_ranks, batch_size, aggregate_method)
        #self.lambda_max_list = self.list_aggregate(self.lambda_max_list, lambda_max_list, batch_size, aggregate_method)
        #print(self.stable_rank_per_group)
        return stable_ranks, lambda_max_list

    def compute_spectral_entropy(self, eigen_list, weight_list, sigma=0.01, grid=1000):
        # Step 1: Filter near-zero eigenvalues
        filtered_eigen, filtered_weight = filter_eigenvalues(eigen_list, weight_list)
        
        # Step 2: Renormalize weights
        renormalized_weight = renormalize_weights(filtered_weight)
        #print("renorm: ", sum(renormalized_weight))
        
        # Step 3: Define lambda grid
        if len(filtered_eigen) == 0:
            raise ValueError("No eigenvalues remain after filtering. Adjust the threshold.")
        lambda_min = min(filtered_eigen) - 1.0  # Adding padding
        lambda_max = max(filtered_eigen) + 1.0  # Adding padding
        lambdas = np.linspace(lambda_min, lambda_max, grid)
        delta_lambda = lambdas[1] - lambdas[0]
        
        # Step 4: Construct spectral density
        density = construct_spectral_density(filtered_eigen, renormalized_weight, lambdas, sigma)
        
        # Step 5: Compute spectral entropy
        epsilon = 1e-12
        p = density + epsilon  # Avoid log(0)
        #spectral_entropy = -np.sum(p * np.log(p))
        p = np.array(renormalized_weight) + epsilon
        spectral_entropy = -np.sum(p * np.log(p))
        weighted_entropy = -np.sum(p * np.log(p) * np.array(filtered_eigen))
        centroid = np.sum(np.array(renormalized_weight) * np.array(filtered_eigen)) 
        spread = np.sum(np.array(renormalized_weight) * (np.array(filtered_eigen) - centroid)**2)
        
        return spectral_entropy, weighted_entropy, centroid, spread

    def compute_effective_rank(self, eigen_list, weight_list):
        epsilon = 1e-12
        filtered_eigen, filtered_weight = filter_eigenvalues(eigen_list, weight_list, threshold=0)
        #print(filtered_eigen)
        weighted_eigen = np.array(filtered_eigen) * np.array(filtered_weight)
        #print(weighted_eigen)
        normalization = np.sum(weighted_eigen) + epsilon

        #print(normalization)
        p = weighted_eigen / (normalization + epsilon)
        p = np.array(p) + epsilon
        #print(p)
        entropy = -np.sum(p * np.log(p))

        effective_rank_entropy = np.exp(entropy)

        return effective_rank_entropy

    def spectral_density(self, n_iter=10, n_v=5, sigma=0.01, grid=100, threshold=1e-10):
        """
        Compute estimated eigenvalue density using the stochastic Lanczos algorithm (SLQ). First compute the Hessian of all batches, then compute the values using the avearge Hessian.
        Parameters:
        -----------
        loss : torch.Tensor
            The loss tensor of the batch for which the Hessian is computed.
        batch_size : int
            The size of the batch.
        n_iter : int, optional (default=10)
            Number of iterations used to compute the trace.
        n_v : int, optional (default=5)
            Number of SLQ runs.
        sigma : float, optional (default=0.01)
            Standard deviation for Gaussian smoothing.
        grid : int, optional (default=100)
            Number of grid points for density estimation.
        threshold : float, optional (default=1e-10)
            Threshold for numerical stability.
        
        Saves:
        --------
        self.spectrum_divergence_list: 
            List of spectral divergences between the eigenvalue densities of each layer and the final layer.
        self.spectrum_entropy_list:
            List of spectral entropies of each layer.
        self.weighted_entropy_list:
            List of weighted entropies of each layer.
        self.centroid_list:
            List of centroids of each layer.
        self.spread_list:
            List of spreads of each layer.
        """

        self.sigma = sigma
        self.grid = grid
        self.threshold = threshold
        def group_product(xs, ys):
            return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])
            #return [torch.sum(x * y).cpu().item() for (x, y) in zip(xs, ys)]
        def group_add(params, update, alpha=1):
            """
            params = params + update*alpha
            :param params: list of variable
            :param update: list of data
            :return:
            """
            for i, p in enumerate(params):
                params[i].data.add_(update[i] * alpha)
            return params
        device = self.device
        layer_eigenvalues = []
        layer_eigenweights = []
        layer_lambdas = []
        layer_density = []
        for weights in self.grouped_layer_weights:
            eigen_list_full, weight_list_full = [], []
            #gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)
            #print("grad shape: ", len(gradients))
            for k in range(n_v):
                v = [torch.randint_like(weight, high=2, device=device) for weight in weights]
                # generate Rademacher random variables
                for v_i in v:
                    v_i[v_i == 0] = -1
                v = normalization(v)

                # standard lanczos algorithm initlization
                v_list = [v]
                w_list = []
                alpha_list = []
                beta_list = []
                ############### Lanczos
                for i in range(n_iter):
                    self.model.zero_grad()
                    w_prime = [torch.zeros(weight.size()).to(device) for weight in weights]
                    if i == 0:
                        #w_prime = torch.autograd.grad(gradients, weights, grad_outputs=v, only_inputs=True, retain_graph=True)
                        w_prime = self.dataloader_hv_product(v, weights)
            
                        alpha = group_product(w_prime, v)
                        alpha_list.append(alpha)
                        w = group_add(w_prime, v, alpha=-alpha)
                        #print("w shape: ", len(w))
                        w_list.append(w)
                    else:
                        beta = torch.sqrt(group_product(w, w))
                        beta_list.append(beta.cpu().item())
                        if beta_list[-1] != 0.:
                            # We should re-orth it
                            v = orthnormal(w, v_list)
                            v_list.append(v)
                        else:
                            # generate a new vector
                            w = [torch.randn(weight.size()).to(device) for weight in weights]
                            v = orthnormal(w, v_list)
                            v_list.append(v)
                        #w_prime = torch.autograd.grad(gradients, weights, grad_outputs=v, only_inputs=True, retain_graph=True)
                        w_prime = self.dataloader_hv_product(v, weights)
                        alpha = group_product(w_prime, v)
                        alpha_list.append(alpha.cpu().item())
                        w_tmp = group_add(w_prime, v, alpha=-alpha)
                        w = group_add(w_tmp, v_list[-2], alpha=-beta)

                T = torch.zeros(n_iter, n_iter).to(device)
                for i in range(len(alpha_list)):
                    T[i, i] = alpha_list[i]
                    if i < len(alpha_list) - 1:
                        T[i + 1, i] = beta_list[i]
                        T[i, i + 1] = beta_list[i]
                a_, b_ = torch.linalg.eig(T)
                #print(a_)
                #print(b_)

                eigen_list = a_.real
                weight_list = b_[0, :].real**2
                eigen_list_full.append(list(eigen_list.cpu().numpy()))
                weight_list_full.append(list(weight_list.cpu().numpy()))
            
            layer_eigenvalues.append(flat_list(eigen_list_full))
            layer_eigenweights.append(flat_list(weight_list_full))

        spectrum_divergence_list = []
        for i in range(len(layer_eigenvalues)-1):
            divergence, _ = compute_spectral_divergences(layer_eigenvalues[i], layer_eigenweights[i], layer_eigenvalues[-1], layer_eigenweights[-1], measure='js')
            spectrum_divergence_list.append(divergence)


        #dis_1, _ = compute_spectral_divergences(layer_eigenvalues[0], layer_eigenweights[0], layer_eigenvalues[-1], layer_eigenweights[-1], measure='kl')
        #dis_2, _ = compute_spectral_divergences(layer_eigenvalues[1], layer_eigenweights[1], layer_eigenvalues[-1], layer_eigenweights[-1], measure='kl')
        #dis_3, _ = compute_spectral_divergences(layer_eigenvalues[0], layer_eigenweights[0], layer_eigenvalues[1], layer_eigenweights[1], measure='js')
        #print("dis: ", dis_1, dis_2, dis_3)
        #print(layer_lambdas[0])
        #print(layer_lambdas[1])
        #D_KL_test = kl_divergence(layer_density[0], layer_density[-1], layer_lambdas[0])
        #D_JS_test = js_divergence(density_test_1, density_test_2, lambdas_test)

        self.spectrum_entropy_list = []
        self.weighted_entropy_list = []
        self.centroid_list = []
        self.spread_list = []
        self.effective_rank_list = []
        for i in range(len(layer_eigenvalues)):
            spectral_entropy, weighted_entropy, centroid, spread = self.compute_spectral_entropy(layer_eigenvalues[i], layer_eigenweights[i], sigma=self.sigma, grid=self.grid)
            #print(spectral_entropy)
            self.spectrum_entropy_list.append(spectral_entropy)
            self.weighted_entropy_list.append(weighted_entropy)
            self.centroid_list.append(centroid)
            self.spread_list.append(spread)
            effective_rank = self.compute_effective_rank(layer_eigenvalues[i], layer_eigenweights[i])
            self.effective_rank_list.append(effective_rank)

        self.spectrum_divergence_list = spectrum_divergence_list
        self.layer_eigenvalues = layer_eigenvalues
        self.layer_eigenweights = layer_eigenweights

        return layer_lambdas, layer_density
    
    def batch_spectral_density(self, loss, batch_size, n_iter=10, n_v=5, sigma=0.01, grid=100, threshold=1e-10):
        """
        Compute estimated eigenvalue density using the stochastic Lanczos algorithm (SLQ). First compute the Hessian of the batch, then take the average over the batch in batch_aggregate.
        Parameters:
        -----------
        loss : torch.Tensor
            The loss tensor of the batch for which the Hessian is computed.
        batch_size : int
            The size of the batch.
        n_iter : int, optional (default=10)
            Number of iterations used to compute the trace.
        n_v : int, optional (default=5)
            Number of SLQ runs.
        sigma : float, optional (default=0.01)
            Standard deviation for Gaussian smoothing.
        grid : int, optional (default=100)
            Number of grid points for density estimation.
        threshold : float, optional (default=1e-10)
            Threshold for numerical stability.
        
        Saves:
        --------
        self.spectrum_divergence_list: 
            List of spectral divergences between the eigenvalue densities of each layer and the final layer.
        self.spectrum_entropy_list:
            List of spectral entropies of each layer.
        self.weighted_entropy_list:
            List of weighted entropies of each layer.
        self.centroid_list:
            List of centroids of each layer.
        self.spread_list:
            List of spreads of each layer.
        """
        self.sigma = sigma
        self.grid = grid
        self.threshold = threshold

        def group_product(xs, ys):
            return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])
            #return [torch.sum(x * y).cpu().item() for (x, y) in zip(xs, ys)]
        def group_add(params, update, alpha=1):
            """
            params = params + update*alpha
            :param params: list of variable
            :param update: list of data
            :return:
            """
            for i, p in enumerate(params):
                params[i].data.add_(update[i] * alpha)
            return params
        def group_div(params, alpha=1):
            """
            params = params + update*alpha
            :param params: list of variable
            :param update: list of data
            :return:
            """
            for i, p in enumerate(params):
                params[i].data.div_(alpha)
            return params
        def list_aggregate(list_a, list_b, batch_size):
            if len(list_a) == 0:
                list_a = [float(b) * batch_size for b in list_b]
            else:
                list_a = [float(a + b * batch_size) for a, b in zip(list_a, list_b)]
            return list_a
        device = self.device
        layer_eigenvalues = []
        layer_eigenweights = []
        layer_lambdas = []
        layer_density = []
        for weights in self.grouped_layer_weights:
            eigen_list_full, weight_list_full = [], []
            self.model.zero_grad()
            v = [torch.randn_like(weight, device=device) for weight in weights]
            v = torch.randn(sum([w.numel() for w in weights]))
            
            #n_param = sum([w.numel for w in weights])
            gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)
            #print(gradients[0].shape)
            #print("grad shape: ", len(gradients))
            for k in range(n_v):
                #v = [torch.randint_like(weight, high=2, device=device) for weight in weights]
                v = [torch.randn_like(weight, device=device) for weight in weights]

                # standard lanczos algorithm initlization
                v_list = [v]
                w_list = []
                alpha_list = []
                beta_list = []
                ############### Lanczos
                #for i in range(n_iter):
                self.model.zero_grad()
                alpha_list, beta_list = lanczos_gradient_single(self.model, loss, weights, n_iter)
                    
                alpha_tensor = torch.tensor(alpha_list, device=self.device)
                beta_tensor = torch.tensor(beta_list, device=self.device)

                T = torch.diag(alpha_tensor) + torch.diag(beta_tensor, 1) + torch.diag(beta_tensor, -1)
                a_, b_ = torch.linalg.eig(T)
                #print(a_)
                #print(b_)

                eigen_list = a_.real
                weight_list = b_[0, :].real**2
                eigen_list_full.append(list(eigen_list.cpu().numpy()))
                weight_list_full.append(list(weight_list.cpu().numpy()))
            
            layer_eigenvalues.append(flat_list(eigen_list_full))
            layer_eigenweights.append(flat_list(weight_list_full))

        spectrum_divergence_list = []
        for i in range(len(layer_eigenvalues)-1):
            divergence, _ = compute_spectral_divergences(layer_eigenvalues[i], layer_eigenweights[i], layer_eigenvalues[-1], layer_eigenweights[-1], measure='js')
            spectrum_divergence_list.append(divergence)
        #divergence, _ = compute_spectral_divergences(layer_eigenvalues[0], layer_eigenweights[0], layer_eigenvalues[-2], layer_eigenweights[-2], measure='js')
        #spectrum_divergence_list.append(divergence)
        #dis_1, _ = compute_spectral_divergences(layer_eigenvalues[0], layer_eigenweights[0], layer_eigenvalues[-1], layer_eigenweights[-1], measure='kl')
        #dis_2, _ = compute_spectral_divergences(layer_eigenvalues[1], layer_eigenweights[1], layer_eigenvalues[-1], layer_eigenweights[-1], measure='kl')
        #dis_3, _ = compute_spectral_divergences(layer_eigenvalues[0], layer_eigenweights[0], layer_eigenvalues[1], layer_eigenweights[1], measure='js')
        #print("dis: ", dis_1, dis_2, dis_3)
        #print(layer_lambdas[0])
        #print(layer_lambdas[1])
        #D_KL_test = kl_divergence(layer_density[0], layer_density[-1], layer_lambdas[0])
        #D_JS_test = js_divergence(density_test_1, density_test_2, lambdas_test)

        spectrum_entropy_list = []
        weighted_entropy_list = []
        centroid_list = []
        spread_list = []
        effective_rank_list = []
        stable_rank_list = []
        for i in range(len(layer_eigenvalues)):
            spectral_entropy, weighted_entropy, centroid, spread = self.compute_spectral_entropy(layer_eigenvalues[i], layer_eigenweights[i], sigma=self.sigma, grid=self.grid)
            #print(spectral_entropy)
            spectrum_entropy_list.append(spectral_entropy * batch_size)
            weighted_entropy_list.append(weighted_entropy * batch_size)
            centroid_list.append(centroid * batch_size)
            spread_list.append(spread * batch_size)

            effective_rank = self.compute_effective_rank(layer_eigenvalues[i], layer_eigenweights[i])
            effective_rank_list.append(effective_rank * batch_size)

            #stable_rank = self.compute_stable_rank(layer_eigenvalues[i], layer_eigenweights[i])
            #stable_rank_list.append(stable_rank * batch_size)


        self.spectrum_entropy_list = list_aggregate(self.spectrum_entropy_list, spectrum_entropy_list, batch_size)
        self.weighted_entropy_list = list_aggregate(self.weighted_entropy_list, weighted_entropy_list, batch_size)
        self.centroid_list = list_aggregate(self.centroid_list, centroid_list, batch_size)
        self.spread_list = list_aggregate(self.spread_list, spread_list, batch_size)
        self.spectrum_divergence_list = list_aggregate(self.spectrum_divergence_list, spectrum_divergence_list, batch_size)

        self.effective_rank_list = list_aggregate(self.effective_rank_list, effective_rank_list, batch_size)

        #self.stable_rank_list = list_aggregate(self.stable_rank_list, stable_rank_list, batch_size)
        #print(len(self.spectrum_divergence_list))
        self.layer_eigenvalues = layer_eigenvalues * batch_size
        self.layer_eigenweights = layer_eigenweights * batch_size

        return layer_eigenvalues, layer_eigenweights

    def batch_spectral_density_old(self, loss, batch_size, n_iter=10, n_v=5, sigma=0.01, grid=100, threshold=1e-10):
        """
        Compute estimated eigenvalue density using the stochastic Lanczos algorithm (SLQ). First compute the Hessian of the batch, then take the average over the batch in batch_aggregate.
        Parameters:
        -----------
        loss : torch.Tensor
            The loss tensor of the batch for which the Hessian is computed.
        batch_size : int
            The size of the batch.
        n_iter : int, optional (default=10)
            Number of iterations used to compute the trace.
        n_v : int, optional (default=5)
            Number of SLQ runs.
        sigma : float, optional (default=0.01)
            Standard deviation for Gaussian smoothing.
        grid : int, optional (default=100)
            Number of grid points for density estimation.
        threshold : float, optional (default=1e-10)
            Threshold for numerical stability.
        
        Saves:
        --------
        self.spectrum_divergence_list: 
            List of spectral divergences between the eigenvalue densities of each layer and the final layer.
        self.spectrum_entropy_list:
            List of spectral entropies of each layer.
        self.weighted_entropy_list:
            List of weighted entropies of each layer.
        self.centroid_list:
            List of centroids of each layer.
        self.spread_list:
            List of spreads of each layer.
        """
        self.sigma = sigma
        self.grid = grid
        self.threshold = threshold

        def group_product(xs, ys):
            return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])
            #return [torch.sum(x * y).cpu().item() for (x, y) in zip(xs, ys)]
        def group_add(params, update, alpha=1):
            """
            params = params + update*alpha
            :param params: list of variable
            :param update: list of data
            :return:
            """
            for i, p in enumerate(params):
                params[i].data.add_(update[i] * alpha)
            return params
        def group_div(params, alpha=1):
            """
            params = params + update*alpha
            :param params: list of variable
            :param update: list of data
            :return:
            """
            for i, p in enumerate(params):
                params[i].data.div_(alpha)
            return params
        def list_aggregate(list_a, list_b, batch_size):
            if len(list_a) == 0:
                list_a = [float(b) * batch_size for b in list_b]
            else:
                list_a = [float(a + b * batch_size) for a, b in zip(list_a, list_b)]
            return list_a
        device = self.device
        layer_eigenvalues = []
        layer_eigenweights = []
        layer_lambdas = []
        layer_density = []
        for weights in self.grouped_layer_weights:
            eigen_list_full, weight_list_full = [], []
            self.model.zero_grad()
            gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)
            #print("grad shape: ", len(gradients))
            for k in range(n_v):
                #v = [torch.randint_like(weight, high=2, device=device) for weight in weights]
                v = [torch.randn_like(weight, device=device) for weight in weights]
                # generate Rademacher random variables
                #for v_i in v:
                #    v_i[v_i == 0] = -1
                v = normalization(v)
                #v /= torch.norm(v)

                # standard lanczos algorithm initlization
                v_list = [v]
                w_list = []
                alpha_list = []
                beta_list = []
                ############### Lanczos
                for i in range(n_iter):
                    self.model.zero_grad()
                    w_prime = [torch.zeros(weight.size()).to(device) for weight in weights]
                    if i == 0:
                        w_prime = torch.autograd.grad(gradients, weights, grad_outputs=v, only_inputs=True, retain_graph=True)
                        #w_prime = self.dataloader_hv_product(v, weights)
            
                        alpha = group_product(w_prime, v)
                        alpha_list.append(alpha)
                        w = group_add(w_prime, v, alpha=-alpha)
                        #print("w shape: ", len(w))
                        w_list.append(w)
                    else:
                        beta = torch.sqrt(group_product(w, w))
                        beta_list.append(beta.cpu().item())
                        if beta_list[-1] != 0.:
                            # We should re-orth it
                            #v = orthnormal(w, v_list)
                            #v = w / beta_list[-1]
                            v = group_div(w, beta_list[-1])
                            v_list.append(v)
                        else:
                            # generate a new vector
                            w = [torch.randn(weight.size()).to(device) for weight in weights]
                            v = orthnormal(w, v_list)
                            v_list.append(v)
                        w_prime = torch.autograd.grad(gradients, weights, grad_outputs=v, only_inputs=True, retain_graph=True)
                        #w_prime = self.dataloader_hv_product(v, weights)
                        alpha = group_product(w_prime, v)
                        alpha_list.append(alpha.cpu().item())
                        w_tmp = group_add(w_prime, v, alpha=-alpha)
                        w = group_add(w_tmp, v_list[-2], alpha=-beta)

                T = torch.zeros(n_iter, n_iter).to(device)
                for i in range(len(alpha_list)):
                    T[i, i] = alpha_list[i]
                    if i < len(alpha_list) - 1:
                        T[i + 1, i] = beta_list[i]
                        T[i, i + 1] = beta_list[i]
                a_, b_ = torch.linalg.eig(T)
                #print(a_)
                #print(b_)

                eigen_list = a_.real
                weight_list = b_[0, :].real**2
                eigen_list_full.append(list(eigen_list.cpu().numpy()))
                weight_list_full.append(list(weight_list.cpu().numpy()))
            
            layer_eigenvalues.append(flat_list(eigen_list_full))
            layer_eigenweights.append(flat_list(weight_list_full))

        spectrum_divergence_list = []
        for i in range(len(layer_eigenvalues)-1):
            divergence, _ = compute_spectral_divergences(layer_eigenvalues[i], layer_eigenweights[i], layer_eigenvalues[-1], layer_eigenweights[-1], measure='js')
            spectrum_divergence_list.append(divergence)
        #divergence, _ = compute_spectral_divergences(layer_eigenvalues[0], layer_eigenweights[0], layer_eigenvalues[-2], layer_eigenweights[-2], measure='js')
        #spectrum_divergence_list.append(divergence)
        #dis_1, _ = compute_spectral_divergences(layer_eigenvalues[0], layer_eigenweights[0], layer_eigenvalues[-1], layer_eigenweights[-1], measure='kl')
        #dis_2, _ = compute_spectral_divergences(layer_eigenvalues[1], layer_eigenweights[1], layer_eigenvalues[-1], layer_eigenweights[-1], measure='kl')
        #dis_3, _ = compute_spectral_divergences(layer_eigenvalues[0], layer_eigenweights[0], layer_eigenvalues[1], layer_eigenweights[1], measure='js')
        #print("dis: ", dis_1, dis_2, dis_3)
        #print(layer_lambdas[0])
        #print(layer_lambdas[1])
        #D_KL_test = kl_divergence(layer_density[0], layer_density[-1], layer_lambdas[0])
        #D_JS_test = js_divergence(density_test_1, density_test_2, lambdas_test)

        spectrum_entropy_list = []
        weighted_entropy_list = []
        centroid_list = []
        spread_list = []
        effective_rank_list = []
        stable_rank_list = []
        for i in range(len(layer_eigenvalues)):
            spectral_entropy, weighted_entropy, centroid, spread = self.compute_spectral_entropy(layer_eigenvalues[i], layer_eigenweights[i], sigma=self.sigma, grid=self.grid)
            #print(spectral_entropy)
            spectrum_entropy_list.append(spectral_entropy * batch_size)
            weighted_entropy_list.append(weighted_entropy * batch_size)
            centroid_list.append(centroid * batch_size)
            spread_list.append(spread * batch_size)

            effective_rank = self.compute_effective_rank(layer_eigenvalues[i], layer_eigenweights[i])
            effective_rank_list.append(effective_rank * batch_size)

            #stable_rank = self.compute_stable_rank(layer_eigenvalues[i], layer_eigenweights[i])
            #stable_rank_list.append(stable_rank * batch_size)


        self.spectrum_entropy_list = list_aggregate(self.spectrum_entropy_list, spectrum_entropy_list, batch_size)
        self.weighted_entropy_list = list_aggregate(self.weighted_entropy_list, weighted_entropy_list, batch_size)
        self.centroid_list = list_aggregate(self.centroid_list, centroid_list, batch_size)
        self.spread_list = list_aggregate(self.spread_list, spread_list, batch_size)
        self.spectrum_divergence_list = list_aggregate(self.spectrum_divergence_list, spectrum_divergence_list, batch_size)

        self.effective_rank_list = list_aggregate(self.effective_rank_list, effective_rank_list, batch_size)

        #self.stable_rank_list = list_aggregate(self.stable_rank_list, stable_rank_list, batch_size)
        #print(len(self.spectrum_divergence_list))
        self.layer_eigenvalues = layer_eigenvalues * batch_size
        self.layer_eigenweights = layer_eigenweights * batch_size

        return layer_eigenvalues, layer_eigenweights

    # From 'Why TF needs Adam'
    def get_layer_spectrum(self, n_v, n_iter):
        weights_dic, values_dic = {}, {}

        for name, param in self.model.named_parameters():
            if name not in self.sample_layer:
                continue
            if param.requires_grad:

                zeros = np.zeros((n_v, n_iter))
                weights_dic[name] = [row.tolist() for row in zeros]
                values_dic[name] =  [row.tolist() for row in zeros]

    
        t_s = time.time()
        for k in range(n_v): 
            #print('current k' , k)

            'wiki version'
            T_dic = self.tridiagonalize_by_lanzcos_layer_by_layer(n_iter, k) #returns a dic: {'name': T}
            
            for name, T in T_dic.items():
                eigenvalues, U  = np.linalg.eigh(T)
                values_dic[name][k] = eigenvalues.tolist() #array to list
                weights_dic[name][k] = (U[0]**2).tolist()

            #print("===values: ", eigenvalues)
            #'we also save the inter-medium results'
            #self.save_curve(total_time= time.time() - t_s, weights_layer = weights_dic, values_layer = values_dic)
        #print(weights_dic.keys())
        for name in values_dic.keys():
            values_dic[name] = np.concatenate(values_dic[name])
            weights_dic[name] = np.concatenate(weights_dic[name])

        return values_dic, weights_dic

        total_time = time.time() - t_s

        self.save_curve(total_time= total_time, weights_layer = weights_dic, values_layer = values_dic)

    def tridiagonalize_by_lanzcos_layer_by_layer(self, n_iter, k):
        v_dic = {} # value: list
        alpha_dic = {} # value: scaler
        w_dic = {} # value: #parameters*1 tensor
        beta_dic = {} # value: scaler
        T_dic = {} # value: m*m tensor 
        'initialize'
        for name, params in self.model.named_parameters():
            if name not in self.sample_layer:
                continue
            if params.requires_grad:
                v = torch.randn_like(params, dtype = torch.float64) 
                v /= torch.norm(v)
                v_dic[name] = [v.cpu()]
                T_dic[name] = np.zeros((n_iter, n_iter), dtype= np.float64)
        #print(v_dic)

        w_prime_dic = self.hessian_vector_product_with_dic_input(v_dic, k,0) 

        'orthogonalize wprime'
        for name in T_dic.keys():
            alpha_dic[name] = torch.sum(w_prime_dic[name] * v_dic[name][-1])  
            w_dic[name] = w_prime_dic[name] - alpha_dic[name] * v_dic[name][-1]
            T_dic[name][0, 0] = alpha_dic[name] 

        'iteration'
        for j in range(1, n_iter):

            for name in T_dic.keys(): 
                beta = torch.norm(w_dic[name])
                beta_dic[name] = beta
                if beta >1e-8:
                    v_dic[name].append( w_dic[name] / beta )
                else:
                    #print('The value of beta is 0')
                    v_dic[name].append( w_dic[name] / 1e-8 )
                    #raise ZeroDivisionError('The value of beta is 0')
                if len(v_dic[name]) > 2:
                    del v_dic[name][0]  # keep this list short to save memory

            t_hessian = time.time()
  
            w_prime_dic = self.hessian_vector_product_with_dic_input(v_dic, k,j) 

            'orthogonalize wprime'
            for name in T_dic.keys():
                alpha_dic[name] = torch.sum(w_prime_dic[name] * v_dic[name][-1])  
                w_dic[name] = w_prime_dic[name] - alpha_dic[name] * v_dic[name][-1] - beta_dic[name] * v_dic[name][-2]
                T_dic[name][j, j] = alpha_dic[name] 
                T_dic[name][j-1, j ] = beta_dic[name] 
                T_dic[name][j , j-1] = beta_dic[name]

        return  T_dic

    def get_full_spectrum(self, n_v, n_iter):
        weights = np.zeros((n_v, n_iter))
        values = np.zeros((n_v, n_iter))

        for k in range(n_v): 
            'wiki version'
            T = self.tridiagonalize_by_lanzcos(n_iter, k)
            eigenvalues, U  = np.linalg.eigh(T)
            values[k,:] = eigenvalues
            weights[k,:] = U[0]**2
        
        all_values = np.concatenate(values)
        all_weights = np.concatenate(weights)
        return all_values, all_weights
   
        grid, curve = self.interpolate(weights, values)
    
    def tridiagonalize_by_lanzcos(self, n_iter, k):
        'set up'
        v_list = []
        T = np.zeros((n_iter, n_iter), dtype= np.float64)

        'initialization'
        v = torch.randn(self.total_params, dtype = torch.float64) 
        v /= torch.norm(v)
        v_list.append(v.cpu())


        w_prime = self.hessian_vector_product_with_tensor_input(v_list[-1], k,0)
        'orthogonalize wprime'
        alpha = torch.sum(w_prime * v_list[-1])
        w = w_prime - alpha * v_list[-1]
        T[0, 0] = alpha

        'iteration'
        #t_s = time.time()
        #print('runing lanczos')
        for j in range(1, n_iter):
            beta = torch.norm(w)
            if beta >1e-8:
                v_list.append(w / beta)

            else:
                v_list.append(w / 1e-8)

                # print(f' since beta = {beta}, generate v that orthogonal to all previous v')
                # # Generate a random vector orthogonal to previous ones
                # v = torch.randn(self.total_params) *(1/self.total_params)**0.5
                # for i in range(j):
                #     vi = v_list[i]
                #     v -= torch.sum(vi * v) * vi
                # v /= torch.norm(v)
                if len(v_list) > 2:
                    del v_list[0]  # keep this list short to save memory


            w_prime = self.hessian_vector_product_with_tensor_input(v_list[-1], k,j)
            alpha = torch.sum(w_prime* v_list[-1])
            w = w_prime - alpha * v_list[-1] - beta * v_list[-2]
            T[j, j] = alpha
            T[j-1, j ] = beta
            T[j , j-1] = beta
         
        return  T

    def hessian_vector_product_with_tensor_input(self, d_tensor, v_step, l_step):
        'comput hessian_vector product, takes a flattened tensors as input (with shape (total parameters, ) )'
        d_tensor = d_tensor.cuda()
        self.model.eval()
        self.model.zero_grad(set_to_none = True)
        total_hd_tensor = 0

        t_hd = time.time()
        for batch in self.dataloader:
            # Specific data process, in order to fit the loss input
            data, target, batch_size = self.load_batch_func(batch, self.device)
            output = self.model(data)
            loss = self.loss_fn(output, target, 'mean')
            #self.model.zero_grad()

            loss.backward(create_graph= True)
            g_list = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    g_list.append(torch.flatten(param.grad.double()))

            g_tensor = torch.cat(g_list, dim = 0)
            
            self.model.zero_grad(set_to_none = True)
            g_tensor = g_tensor.cuda()
            l = torch.sum(g_tensor*d_tensor)
            l.backward(retain_graph = True)

            hd_list = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    hd_list.append(torch.flatten(param.grad.double().data.clone()))

            hd_tensor = torch.cat(hd_list, dim = 0)
            self.model.zero_grad(set_to_none = True)
            hd_tensor = hd_tensor.cpu()
            total_hd_tensor += hd_tensor

            #if self.use_minibatch == True and batch_idx == self.gradient_accumulation_steps-1:
            #    break
        return total_hd_tensor

    def hessian_vector_product_with_dic_input(self, d_dic, v_step, l_step):

        'comput hessian_vector product, takes a dictionary as input, the values of dic is a list of historical lanscoz directions: d_dic = {name, [history v..]}'
        self.model.eval()
        self.model.zero_grad(set_to_none = True)

        'initialize'
        hd_dic = {}
        for name, param in self.model.named_parameters():
            if name not in self.sample_layer:
                continue
            if param.requires_grad:
                hd_dic[name]  = torch.zeros_like(param.data).cpu()


        t_hd = time.time()
        for batch in self.dataloader:
            # Specific data process, in order to fit the loss input
            data, target, batch_size = self.load_batch_func(batch, self.device)
            output = self.model(data)
            loss = self.loss_fn(output, target, 'mean')
            loss.backward(create_graph= True)
            g_dic = {}
            for name, param in self.model.named_parameters():
                if name not in self.sample_layer:
                    continue
                if param.requires_grad:
                    g_dic[name] = param.grad.double()

        
            self.model.zero_grad(set_to_none = True)
            for name, param in self.model.named_parameters():
                if name not in self.sample_layer:
                    continue
                if param.requires_grad:
                    l = torch.sum(g_dic[name].cuda() * d_dic[name][-1].cuda())
                    l.backward(retain_graph = True)
                    hd = param.grad.double().data.clone()
                    hd_dic[name]  += hd.cpu()   
                    self.model.zero_grad(set_to_none = True)
            break
       
        return hd_dic

    # Compute the spectrum only
    def compute_spectrum(self, train_num, n_iter=10, n_v=5, method=1):
        with sdpa_kernel(SDPBackend.MATH):
            if method == 1:
                # First compute the Hessian for all batches. Then compute the spectral density
                self.spectral_density(n_iter=n_iter, n_v=n_v, sigma=0.01)
            elif method == 2:
                # First compute the Hessian and spectral density for each batch. Then aggregate the avearge for all batches.
                device = self.device
                model = self.model
                loss_fn = self.loss_fn
                for batch in self.dataloader:
                    data, target, batch_size = self.load_batch_func(batch, device)
                    output = model(data)
                    loss = loss_fn(output, target, 'none')
                    model.zero_grad()

                    self.batch_spectral_density(loss.mean(), batch_size, n_iter=n_iter, n_v=n_v)

                self.spectrum_divergence_list = self.group_div_const(self.spectrum_divergence_list, train_num)
                self.centroid_list = self.group_div_const(self.centroid_list, train_num)
                self.spread_list = self.group_div_const(self.spread_list, train_num)
                self.weighted_entropy_list = self.group_div_const(self.weighted_entropy_list, train_num)
                self.spectrum_entropy_list = self.group_div_const(self.spectrum_entropy_list, train_num)
                self.effective_rank_list = self.group_div_const(self.effective_rank_list, train_num)
                self.stable_rank_list = self.group_div_const(self.stable_rank_list, train_num)
            else:
                print("=======> SLQ for full model")
                values_full, weights_full = self.get_full_spectrum(n_v=n_v, n_iter=n_iter)
                self.values_full = values_full.tolist()
                self.weights_full = weights_full.tolist()
                spectral_entropy, weighted_entropy, centroid, spread = self.compute_spectral_entropy(values_full, weights_full, sigma=0.01, grid=1000)
                self.spectrum_entropy_list.append(spectral_entropy)
                self.weighted_entropy_list.append(weighted_entropy)
                self.centroid_list.append(centroid)
                self.spread_list.append(spread)
                effective_rank = self.compute_effective_rank(values_full, weights_full)
                self.effective_rank_list.append(effective_rank)
                filtered_eigens, _ = filter_eigenvalues(values_full, weights_full)
                self.condition_list.append(np.abs(np.max(filtered_eigens)) / np.abs(np.min(filtered_eigens)))
                #print(values_full)
                
                """
                print("SLQ for layers")
                values_dic, weights_dic = self.get_layer_spectrum(n_v=n_v, n_iter=n_iter)
                self.values_head = values_dic['head.weight'].tolist()
                self.weights_head = weights_dic['head.weight'].tolist()

                for name in values_dic.keys():
                    spectral_entropy, weighted_entropy, centroid, spread = self.compute_spectral_entropy(values_dic[name], weights_dic[name], sigma=0.01, grid=1000)
                    self.spectrum_entropy_list.append(spectral_entropy)
                    self.weighted_entropy_list.append(weighted_entropy)
                    self.centroid_list.append(centroid)
                    self.spread_list.append(spread)
                    effective_rank = self.compute_effective_rank(values_dic[name], weights_dic[name])
                    self.effective_rank_list.append(effective_rank)
                    filtered_eigens, _ = filter_eigenvalues(values_dic[name], weights_dic[name])
                    self.condition_list.append(np.abs(np.max(filtered_eigens)) / np.abs(np.min(filtered_eigens)))

                for name in values_dic.keys():
                    values_dic[name] = values_dic[name].tolist()
                    weights_dic[name] = weights_dic[name].tolist()
                self.values_dic = values_dic
                self.weights_dic = weights_dic
                """

        for param in self.model.parameters():
            if param.grad is not None:
                param.grad = None   

                


    def log(self, logger, i):
        logger.log("values_full", self.values_full, i)
        logger.log("weights_full", self.weights_full, i)
        #logger.log("values_head", self.values_head, i)
        #logger.log("weights_head", self.weights_head, i)
        #logger.log("values_dic", self.values_dic, i)
        #logger.log("weights_dic", self.weights_dic, i)
        logger.log("spectral_entropy", self.spectrum_entropy_list, i)
        logger.log("weighted_entropy", self.weighted_entropy_list, i)
        logger.log("centroid", self.centroid_list, i)
        logger.log("spread", self.spread_list, i)
        #logger.log("spectrum_divergence", self.spectrum_divergence_list, i)
        logger.log("effective_rank", self.effective_rank_list, i)
        #logger.log("shapescale", self.shapescale, i)
        #logger.log("stable_rank", self.stable_rank_list, i)
        logger.log("condition", self.condition_list, i)


def compute_eigenvalue(model, loss, device, maxIter=100, tol=1e-10, top_n=1):
    model.zero_grad()
    layers = model.get_layers()
    weights = [module.weight for name, module in layers.items()]
    
    model.zero_grad()
    gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)

    topn_eigenvalues = []
    eigenvectors = []
    computed_dim = 0
    while computed_dim < top_n:
        
        eigenvalues = None
        vs = [torch.randn_like(weight) for weight in weights]  # generate random vector
        vs = normalization(vs)  # normalize the vector

        for _ in range(maxIter):
            vs = orthnormal(vs, eigenvectors)
            model.zero_grad()

            Hvs = torch.autograd.grad(gradients, weights, grad_outputs=vs, retain_graph=True)
            tmp_eigenvalues = [ torch.sum(Hv*v).cpu().item() for (Hv, v) in zip(Hvs, vs)]

            vs = normalization(Hvs)

            if eigenvalues == None:
                eigenvalues = tmp_eigenvalues
            else:
                if abs(sum(eigenvalues) - sum(tmp_eigenvalues)) / (abs(sum(eigenvalues)) +
                                                        1e-6) < tol:
                    break
                else:
                    eigenvalues = tmp_eigenvalues
        topn_eigenvalues.append(eigenvalues)
        eigenvectors.append(vs)
        computed_dim += 1

    return topn_eigenvalues, eigenvectors


def get_grouped_layer_weights(model):
    layers = get_layers(model)
    weights = []
    trace_num = []
    layer_names = []
    for name, module in layers.items():
        layer_names.append(name)
        weights.append(module.weight)

    grouped_layer_weights = []
    grouped_layer_names = []
    # Embedding layer
    #grouped_layer_weights.append([weights[0], weights[1]])
    #grouped_layer_names.append('Embedding')
    # Transformer layers
    #grouped_layer_weights.append(weights[:-1])
    #grouped_layer_names.append('Entire model')
    # Entire model
    grouped_layer_weights.append(weights)
    grouped_layer_names.append('Entire model')
    # Head
    grouped_layer_weights.append([weights[-1]])
    grouped_layer_names.append('Head')

    return weights, layer_names, grouped_layer_weights, grouped_layer_names

def get_layers(model):
    layers = OrderedDict()
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and any(p.requires_grad for p in module.parameters(recurse=False)):
            #if (type(module) == torch.nn.Linear) and "LayerNorm" not in name and "ln" not in name and "embeddings" not in name and "pooler" not in name:
            if "LayerNorm" not in name and "ln" not in name and "pooler" not in name:
            #print(f"Layer: {name}, Module: {module}")
                layers[name] = module
    return layers

def normalization_list(vs):
    total_norm_sq = sum((v**2).sum() for v in vs)
    norm = total_norm_sq.sqrt()
    if norm < 1e-12:
        return [v.clone() for v in vs]
    return [v / norm for v in vs]

def orthnormal_list(vs, eigenvectors):
    if len(eigenvectors) == 0:
        return normalization_list(vs)
    for e in eigenvectors:
        # e is a list of tensors; compute dot(vs, e)
        dot = sum(torch.sum(v * e_part) for v, e_part in zip(vs, e))
        vs = [v - dot * e_part for v, e_part in zip(vs, e)]
    return normalization_list(vs)

def orthnormal(ws, vs_list):
    """
    make vector w orthogonal to each vector in v_list.
    afterwards, normalize the output w
    """
    for vs in vs_list:
        for w, v in zip(ws, vs):
            w.data.add_(-v*(torch.sum(w*v)))
    return normalization(ws)

# copy from pyhessian
def normalization(v):
    """
    normalization of a list of vectors
    return: normalized vectors v
    """
    s = group_product(v, v)
    s = s**0.5
    s = s.cpu().item()
    v = [vi / (s + 1e-6) for vi in v]
    return v

def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])

def normalization_(vs, epsilon=1e-6):
    """
    normalization of a list of vectors
    return: normalized vectors v
    """
    """
    norms = [torch.sum(v*v) for v in vs]
    norms = [(norm**0.5).cpu().item() for norm in norms]
    vs = [vi / (norms[i] + 1e-6) for (i, vi) in enumerate(vs)]
    return vs
    """
    return [v / (torch.norm(v) + epsilon) for v in vs]

def lanczos(A, n_iter):
    n = A.shape[0]
    v = torch.randn(n)

    v_list = []
    alpha_list = []
    beta_list = []

    v = v / torch.linalg.norm(v)
    v_list.append(v)

    # Hessian product vector
    w = A @ v
    alpha_list.append(torch.dot(v, w).item())
    w = w - alpha_list[0] * v

    for j in range(1, n_iter):
        beta_tmp = torch.linalg.norm(w)
        if beta_tmp < 1e-10:
            print("break!")
            # or?
            return alpha_list, beta_list
        beta_list.append(beta_tmp.item())
        v = w / beta_list[-1]
        v_list.append(v)
        w = A @ v
        alpha_list.append(torch.dot(w, v).item())
        w = w - alpha_list[-1] * v - beta_list[-1] * v_list[-2]
    
    return alpha_list, beta_list

def flatten_tensors(tensor_list):
    flats = []
    for t in tensor_list:
        flats.append(t.contiguous().view(-1))
    return torch.cat(flats)

def unflatten_tensors(flat_tensor, tensor_list):
    new_tensors = []
    offset = 0
    for t in tensor_list:
        numel = t.numel()
        new_tensors.append(flat_tensor[offset : offset + numel].view_as(t))
        offset += numel
    return new_tensors

def lanczos_gradient_single(model, loss, weights, n_iter):
    n = sum([w.numel() for w in weights])
    v = torch.randn(n)
    #v = torch.randn_like(weight[0])
    # This gives you a single tensor holding all the gradients
    v_list = []
    alpha_list = []
    beta_list = []

    v = v / torch.linalg.norm(v)
    v_list.append(v)

    # Hessian product vector
    #w = torch.autograd.grad(gradient, weight, grad_outputs=v_list, only_inputs=True, retain_graph=True)
    w = hessian_vector_product_with_tensor_input(model, loss, weights, v)

    alpha_list.append(torch.dot(v, w).item())
    w = w - alpha_list[0] * v

    for j in range(1, n_iter):
        beta_tmp = torch.linalg.norm(w)
        if beta_tmp < 1e-10:
            print("break!")
            # or?
            return alpha_list, beta_list
        beta_list.append(beta_tmp.item())
        v = w / beta_list[-1]
        v_list.append(v)
        #w = torch.autograd.grad(grads, weight, grad_outputs=v, only_inputs=True, retain_graph=True)
        model.zero_grad()
        w = hessian_vector_product_with_tensor_input(model, loss, weights, v)
        alpha_list.append(torch.dot(w, v).item())
        w = w - alpha_list[-1] * v - beta_list[-1] * v_list[-2]
    
    return alpha_list, beta_list

def hessian_vector_product_with_tensor_input(model, loss, weights, d_tensor):
    'comput hessian_vector product, takes a flattened tensors as input (with shape (total parameters, ) )'

    d_tensor = torch.tensor(d_tensor).cuda()
    total_hd_tensor = 0

    t_hd = time.time()

    loss.backward(create_graph= True)
    g_list = []

    for w in weights:
        if w.requires_grad:
            #print(w)
            g_list.append(torch.flatten(w.grad))

    g_tensor = torch.cat(g_list, dim = 0)
    
    model.zero_grad(set_to_none = True)
    g_tensor = g_tensor.cuda()
    l = torch.sum(g_tensor*d_tensor)
    l.backward(retain_graph = True)

    hd_list = []
    for w in weights:
        if w.requires_grad:
            #print(w)
            hd_list.append(torch.flatten(w.grad.data.clone()))

    hd_tensor = torch.cat(hd_list, dim = 0)
    model.zero_grad(set_to_none = True)
    hd_tensor = hd_tensor.cpu()
    total_hd_tensor += hd_tensor
    #print("===;", total_hd_tensor.shape)

    return total_hd_tensor
