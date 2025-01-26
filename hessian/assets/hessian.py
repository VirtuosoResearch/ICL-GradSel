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

from pyhessian.utils import group_product, group_add, normalization, get_params_grad, hessian_vector_product, orthnormal

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

def filter_eigenvalues(eigen_list, weight_list, threshold=1e-8):
    filtered_eigen = []
    filtered_weight = []
    #print(np.max(weight_list))
    for eig, w in zip(eigen_list, weight_list):
        if eig >= 1e-8 and w >= threshold:
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


def normalization(vs, epsilon=1e-6):
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

def orthnormal(ws, vs_list):
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

class hessian():
    """
    The class used to compute :
        i) the top 1 (n) eigenvalue(s) of the neural network
        ii) the trace of the entire neural network
        iii) the estimated eigenvalue density
    """

    def __init__(self, model, criterion, data=None, dataloader=None, device='cpu'):
        """
        model: the model that needs Hessain information
        criterion: the loss function
        data: a single batch of data, including inputs and its corresponding labels
        dataloader: the data loader including bunch of batches of data
        """

        # make sure we either pass a single batch or a dataloader
        assert (data != None and dataloader == None) or (data == None and
                                                         dataloader != None)

        self.model = model.eval()  # make model is in evaluation model
        self.criterion = criterion

        if data != None:
            self.data = data
            self.full_dataset = False
        else:
            self.data = dataloader
            self.full_dataset = True

        self.device = device

        # pre-processing for single batch case to simplify the computation.
        if not self.full_dataset:
            self.inputs, self.targets = self.data
            self.inputs = self.inputs.to(device)
            self.targets = self.targets.to(device)

            # if we only compute the Hessian information for a single batch data, we can re-use the gradients.
            outputs = self.model(self.inputs)
            loss = self.criterion(outputs, self.targets)
            loss.backward(create_graph=True)

        # this step is used to extract the parameters from the model
        layers = self.model.get_layers()
        weights = [module.weight for name, module in layers.items()]
        self.params = weights
        params, gradsH = get_params_grad(self.model)
        self.params = params
        self.gradsH = gradsH  # gradient used for Hessian computation

    def dataloader_hv_product(self, v):

        device = self.device
        num_data = 0  # count the number of datum points in the dataloader
        THv = [torch.zeros(p.size()).to(device) for p in self.params
              ]  # accumulate result
        #for inputs, targets in self.data:
        for batch in self.data:
            batch = batch[0].to(device)
            inputs = batch[:, :-1]
            targets = batch
            self.model.zero_grad()
            tmp_num_data = inputs.size(0)
            outputs = self.model(inputs.to(device))
            loss = self.criterion(outputs, targets.to(device))
            loss.backward(create_graph=True)
            params, gradsH = get_params_grad(self.model)
            layers = self.model.get_layers()
            weights = [module.weight for name, module in layers.items()]
            gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)            
            self.model.zero_grad()
            Hv = torch.autograd.grad(gradients, weights, grad_outputs=v, retain_graph=True)
            #Hv = torch.autograd.grad(gradsH,
            #                         params,
            #                         grad_outputs=v,
            #                         only_inputs=True,
            #                         retain_graph=False)
            THv = [
                THv1 + Hv1 * float(tmp_num_data) + 0.
                for THv1, Hv1 in zip(THv, Hv)
            ]
            num_data += float(tmp_num_data)

        THv = [THv1 / float(num_data) for THv1 in THv]
        eigenvalue = group_product(THv, v).cpu().item()
        return eigenvalue, THv
    
    def eigenvalues(self, maxIter=100, tol=1e-3, top_n=1):
        """
        compute the top_n eigenvalues using power iteration method
        maxIter: maximum iterations used to compute each single eigenvalue
        tol: the relative tolerance between two consecutive eigenvalue computations from power iteration
        top_n: top top_n eigenvalues will be computed
        """

        assert top_n >= 1

        device = self.device

        eigenvalues = []
        eigenvectors = []

        computed_dim = 0

        while computed_dim < top_n:
            eigenvalue = None
            v = [torch.randn(p.size()).to(device) for p in self.params
                ]  # generate random vector
            v = normalization(v)  # normalize the vector

            for i in range(maxIter):
                v = orthnormal(v, eigenvectors)
                self.model.zero_grad()

                if self.full_dataset:
                    tmp_eigenvalue, Hv = self.dataloader_hv_product(v)
                else:
                    Hv = hessian_vector_product(self.gradsH, self.params, v)
                    tmp_eigenvalue = group_product(Hv, v).cpu().item()

                v = normalization(Hv)

                if eigenvalue == None:
                    eigenvalue = tmp_eigenvalue
                else:
                    if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) +
                                                           1e-6) < tol:
                        break
                    else:
                        eigenvalue = tmp_eigenvalue

            eigenvalues.append(eigenvalue)
            eigenvectors.append(v)
            computed_dim += 1

        #return eigenvalues
        return eigenvalues, eigenvectors

    def trace(self, maxIter=100, tol=1e-3):
        """
        compute the trace of hessian using Hutchinson's method
        maxIter: maximum iterations used to compute trace
        tol: the relative tolerance
        """

        device = self.device
        trace_vhv = []
        trace = 0.

        for i in range(maxIter):
            self.model.zero_grad()
            v = [
                torch.randint_like(p, high=2, device=device)
                for p in self.params
            ]
            # generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1

            if self.full_dataset:
                _, Hv = self.dataloader_hv_product(v)
            else:
                Hv = hessian_vector_product(self.gradsH, self.params, v)
            trace_vhv.append(group_product(Hv, v).cpu().item())
            if abs(np.mean(trace_vhv) - trace) / (trace + 1e-6) < tol:
                return trace_vhv
            else:
                trace = np.mean(trace_vhv)
        
        return trace_vhv

    def density(self, n_iter=10, n_v=5):
        """
        compute estimated eigenvalue density using stochastic lanczos algorithm (SLQ)
        n_iter: number of iterations used to compute trace
        n_v: number of SLQ runs
        """

        device = self.device
        eigen_list_full = []
        weight_list_full = []

        for k in range(n_v):
            v = [
                torch.randint_like(p, high=2, device=device)
                for p in self.params
            ]
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
                w_prime = [torch.zeros(p.size()).to(device) for p in self.params]
                if i == 0:
                    if self.full_dataset:
                        _, w_prime = self.dataloader_hv_product(v)
                    else:
                        w_prime = hessian_vector_product(
                            self.gradsH, self.params, v)
                    alpha = group_product(w_prime, v)
                    alpha_list.append(alpha.cpu().item())
                    w = group_add(w_prime, v, alpha=-alpha)
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
                        w = [torch.randn(p.size()).to(device) for p in self.params]
                        v = orthnormal(w, v_list)
                        v_list.append(v)
                    if self.full_dataset:
                        _, w_prime = self.dataloader_hv_product(v)
                    else:
                        w_prime = hessian_vector_product(
                            self.gradsH, self.params, v)
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

        return eigen_list_full, weight_list_full

class Hessian_Calculator():
    def __init__(self, model, loss_fn, p, dataloader=None, external_load_batch_func=None, device='cpu'):
        self.p = p
        self.num_classes = p+2
        self.model = model.eval()  # make model is in evaluation model
        self.loss_fn = loss_fn 

        if external_load_batch_func is not None:
            self.load_batch_func = external_load_batch_func
        else:
            self.load_batch_func = load_batch_func
        
        self.dataloader = dataloader

        self.device = device

        # this step is used to extract the parameters from the model
        #self.layers = self.model.get_layers()
        self.layers = get_layers(self.model)
        self.weights = []
        self.trace_num = []
        self.layer_names = []
        for name, module in self.layers.items():
            self.layer_names.append(name)
            self.weights.append(module.weight)
            #self.trace_num.append(module.weight.shape[0] * module.weight.shape[1])
        print(self.layer_names)

        #params, gradsH = get_params_grad(model)
        #self.weights = params
        
        # TODO automatically group layer weights
        self.grouped_layer_weights = []
        #for i in range(len(self.weights)):
        #    self.grouped_layer_weights.append([self.weights[i]])
        #self.grouped_layer_weights.append([self.weights[2], self.weights[3]])
        #self.grouped_layer_weights.append([self.weights[-3], self.weights[-2]])
        #self.grouped_layer_weights.append([self.weights[0]])
        #self.grouped_layer_weights.append([self.weights[1]])

        # Embedding layer & attention
        self.grouped_layer_weights.append([self.weights[0], self.weights[1]])
        #self.grouped_layer_weights.append([self.weights[2], self.weights[4]])
        #self.grouped_layer_weights.append([self.weights[3], self.weights[5], self.weights[6]])
        #self.grouped_layer_weights.append([self.weights[2]])
        # Blocks in transformer layers
        #for i in range(len(self.model.layers)):
            #self.grouped_layer_weights.append([self.weights[i*3+2]])
        #    self.grouped_layer_weights.append([self.weights[i*2+3], self.weights[i*2+4]])
        # All transformer layers
        #self.grouped_layer_weights.append(self.weights[:-1])
        self.grouped_layer_weights.append(self.weights)
        # Head
        self.grouped_layer_weights.append([self.weights[-1]])
        for w in self.grouped_layer_weights:
            print(len(w))
        self.grouped_layer_names = []
        self.grouped_layer_names.append('Transformer-1')
        self.grouped_layer_names.append('Head')

        self.params = self.weights

        self.hessian_norms = []
        self.layer_trace = []
        self.max_eigenvalues = 0
        self.max_eigenvector_1 = None
        self.max_eigenvector_2 = None
        self.lambda_1 = 0
        self.lambda_2 = 0
        self.lambda_n = 0
        self.layer_wd_trace = None
        self.cosine_similarities = 0
        self.grad_norms = 0
        self.wd_grad_norms = 0
        self.l2_distance = 0
        self.loss_singularvalue_distance = 0
        self.loss_singularvector_distance = 0
        self.item_1 = 0
        self.item_2 = 0
        self.noise_sensitivity = 0
        
        self.spectrum_divergence_list = []
        self.spectrum_entropy_list = []
        self.weighted_entropy_list = []
        self.centroid_list = []
        self.spread_list = []
        self.effective_rank_list = []
        self.stable_rank_list = []
        self.stable_rank_per_group = []
        
    
    def group_div_const(self, X, c):
        return [x/c for x in X]
    
    def collect(self, train_num):
        self.batch_collect()
        self.batch_aggregate(train_num)

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
        self.stable_rank_per_group = self.group_div_const(self.stable_rank_per_group, train_num)
        #print(self.layer_trace)
        #print(self.effective_rank_list)
        self.shapescale = (self.layer_trace*np.array(self.effective_rank_list)).tolist()
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

        #self.noise_sensitivity /= train_num
        #print(self.noise_sensitivity)
        
        #self.loss_singularvector_distance = np.mean(self.loss_singularvector_distance / train_num)
        #self.loss_singularvalue_distance = np.mean(self.loss_singularvalue_distance / train_num)

        hessian_quantities = np.sum(sqrt_with_neg_handling(np.array(self.hessian_norms))) / np.sqrt(train_num)
        self.train_hessianmeasurement = (hessian_quantities).item()
    
    def compute_item(self, model, data, target, batch_size):
        output = model(data)
        rep = model.get_rep(data)[:, -1]

        logits = output[:, -1]
        y = target[:, -1]
        P = F.softmax(logits)
        item_1 = torch.trace(torch.mm(rep, rep.t()))
        diag_P = torch.diag_embed(P) 
        outer_P = P.unsqueeze(2) * P.unsqueeze(1)
        H_G = diag_P - outer_P
        item_2 = torch.trace(H_G.mean(dim=0))
        self.item_1 += item_1.item()
        self.item_2 += item_2.item() * batch_size

    def compute_sensitivity(self, model, loss_fn, data, target, batch_size):
        output = model(data)
        loss = loss_fn(output, target, 'none')

        # noise_sensitivity: Estimate the sensetivity of input. Save in ./results/input
        for i in range(50):
            #noisy_output, noise_norm = model.add_noise_forward(data)
            
            #noise_sensitivity = torch.norm(noisy_output[:, -1] - output[:, -1]) / noise_norm
            #noisy_loss = F.cross_entropy(noisy_output[:, -1], target[:, -1], reduction='none')
            #noise_sensitivity = (noisy_loss - loss) / noise_norm
            noisy_output_1, noisy_output_2, noise_norm = model.add_bi_noise_forward(data)
            noisy_loss_1 = F.cross_entropy(noisy_output_1[:, -1], target[:, -1], reduction='none')
            noisy_loss_2 = F.cross_entropy(noisy_output_2[:, -1], target[:, -1], reduction='none')
            noise_sensitivity = (noisy_loss_1 + noisy_loss_2 - 2*loss)
            #noise_sensitivity = (noisy_output_1 + noisy_output_2 - output)[:, -1]

        self.noise_sensitivity += noise_sensitivity.mean().item() * batch_size

    def compute_singular_ratio(self, model, loss, batch_size):
        with torch.set_grad_enabled(True), sdpa_kernel(SDPBackend.MATH):
            loss_grads = torch.autograd.grad(loss, self.weights, retain_graph=True, create_graph=True)
            
            # Compute weight decay gradients
            lam = 1
            weight_decay_grads = [lam * w for w in self.weights]

            # \sigma1/\sigma2
            cosine_similarities = []
            l2_distance = []
            grad_norms = []
            wd_grad_norms = []
            loss_max_singularvector_1 = []
            loss_max_singularvector_2 = []
            loss_max_singularvalue_1 = []
            loss_max_singularvalue_2 = []
            loss_singularvector_distance = []
            loss_singularvalue_distance = []
            
            for name, grad, wd_grad in zip(self.layer_names, loss_grads, weight_decay_grads):
                U, S, Vh = torch.linalg.svd(grad, full_matrices=False)

                # Extract the first and second singular values
                first_singular_value = S[0].item()
                second_singular_value = S[1].item()
                loss_max_singularvalue_1.append(first_singular_value)
                loss_max_singularvalue_2.append(second_singular_value)

                # Extract the corresponding singular vectors
                # U and Vh are orthogonal matrices. Columns of U are left singular vectors.
                # Rows of Vh are the right singular vectors (V transposed).
                first_left_singular_vector = U[:, 0]
                first_right_singular_vector = Vh[0, :]
                second_left_singular_vector = U[:, 1]
                second_right_singular_vector = Vh[1, :]
                loss_max_singularvector_1.append(first_right_singular_vector)
                loss_max_singularvector_2.append(second_right_singular_vector)

                loss_singularvalue_distance.append(first_singular_value/second_singular_value)
                loss_singularvector_distance.append(torch.norm(Vh[0, :] - Vh[1, :]).item())
                # Flatten the gradients into 1D tensors
                grad_flat = grad.view(-1)
                wd_grad_flat = wd_grad.view(-1)

                # Compute the dot product and norms
                dot_product = torch.dot(grad_flat, wd_grad_flat)
                norm_grad = torch.norm(grad_flat)
                norm_wd_grad = torch.norm(wd_grad_flat)

                grad_norms.append(norm_grad.item())
                #print(norm_grad.item(), norm_wd_grad.item())
                wd_grad_norms.append(norm_wd_grad.item())

                # Compute cosine similarity with numerical stability
                cos_sim = dot_product / (norm_grad * norm_wd_grad + 1e-8)

                # Optionally, detach the cosine similarity from the graph and convert to float
                cosine_similarities.append(cos_sim.item())

                l2_distance.append(torch.norm(grad_flat - wd_grad_flat).item())

            
            self.cosine_similarities += np.array(cosine_similarities) * batch_size
            #print(self.cosine_similarities)
            self.grad_norms += np.array(grad_norms) * batch_size
            self.wd_grad_norms += np.array(wd_grad_norms) * batch_size
            self.l2_distance += np.array(l2_distance) * batch_size

            self.loss_singularvalue_distance += np.array(loss_singularvalue_distance) * batch_size
            self.loss_singularvector_distance += np.array(loss_singularvector_distance) *batch_size

    def compute_hessians_trace(self, model, loss, batch_size, device = "cpu", maxIter=100, tol=1e-3):
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
        
        if len(self.layer_trace) == 0:
            #avg_layer_trace = np.mean(np.array(layer_traces), axis=0) / trace_num
            self.layer_trace = grouped_layer_trace
        else:
            self.layer_trace = np.maximum(grouped_layer_trace, self.layer_trace).tolist()
        #print(self.layer_trace)

    def compute_wd_hessians_trace(self, lam=1):        
        self.layer_wd_trace = np.array(self.trace_num) * lam
    
    def compute_eigenvalues(self, model, loss, batch_size):
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
        #print(lambda_1)
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
        
        #layer_hessian_quantities = np.sum(layer_hessian_quantities) / np.sqrt(config.train.hessian_log_size)
        if len(self.hessian_norms) == 0:
            self.hessian_norms = layer_hessian_quantities
        else:
            self.hessian_norms = np.maximum(self.hessian_norms, layer_hessian_quantities)

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
        v = torch.cat(vecs).detach()
        
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
            if norm_Hv < 1e-12:
                return 0.0
            v = Hv / norm_Hv
        
        # Rayleigh quotient approximation for final eigenvalue
        Hv = apply_H(v)
        lambda_max_approx = torch.dot(v, Hv).item()
        return lambda_max_approx

    def compute_stable_rank_per_group(self, loss, batch_size):
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
        for weights in self.grouped_layer_weights:
            # 1. trace(H^2)
            trace_h2 = self.approximate_trace_h2(loss, weights)
            
            # 2. largest eigenvalue (power iteration)
            lambda_max = self.approximate_lambda_max(loss, weights, power_iter=20)
            
            # 3. stable rank = trace(H^2) / (lambda_max^2)
            epsilon = 1e-12
            srank = trace_h2 / (lambda_max**2 + epsilon)
            
            stable_ranks.append(srank)
        
        self.stable_rank_per_group = list_aggregate(self.stable_rank_per_group, stable_ranks, batch_size)
        #print(self.stable_rank_per_group)
        return stable_ranks

    def compute_spectral_entropy(self, eigen_list, weight_list, sigma=0.1, grid=100):
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
        filtered_eigen, filtered_weight = filter_eigenvalues(eigen_list, weight_list)
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
    
    def compute_stable_rank(self, eigen_list, weight_list):
        epsilon = 1e-12
        filtered_eigen, filtered_weight = filter_eigenvalues(eigen_list, weight_list)
        F_norm_value = np.sum(np.array(filtered_eigen)**2 * np.array(filtered_weight))
        #print("F: ", F_norm_value)
        largest_eigenvalue = np.max(filtered_eigen)
        #print("largest eig: ",largest_eigenvalue)

        stable_rank = F_norm_value / (largest_eigenvalue**2 + epsilon)

        return stable_rank

    # copy from pyhessian
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
        
        divergence, _ = compute_spectral_divergences(layer_eigenvalues[0], layer_eigenweights[0], layer_eigenvalues[-2], layer_eigenweights[-2], measure='js')
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
        for i in range(len(layer_eigenvalues)):
            spectral_entropy, weighted_entropy, centroid, spread = self.compute_spectral_entropy(layer_eigenvalues[i], layer_eigenweights[i], sigma=self.sigma, grid=self.grid)
            #print(spectral_entropy)
            self.spectrum_entropy_list.append(spectral_entropy)
            self.weighted_entropy_list.append(weighted_entropy)
            self.centroid_list.append(centroid)
            self.spread_list.append(spread)

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
                            v = orthnormal(w, v_list)
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

            stable_rank = self.compute_stable_rank(layer_eigenvalues[i], layer_eigenweights[i])
            stable_rank_list.append(stable_rank * batch_size)


        self.spectrum_entropy_list = list_aggregate(self.spectrum_entropy_list, spectrum_entropy_list, batch_size)
        self.weighted_entropy_list = list_aggregate(self.weighted_entropy_list, weighted_entropy_list, batch_size)
        self.centroid_list = list_aggregate(self.centroid_list, centroid_list, batch_size)
        self.spread_list = list_aggregate(self.spread_list, spread_list, batch_size)
        self.spectrum_divergence_list = list_aggregate(self.spectrum_divergence_list, spectrum_divergence_list, batch_size)

        self.effective_rank_list = list_aggregate(self.effective_rank_list, effective_rank_list, batch_size)

        self.stable_rank_list = list_aggregate(self.stable_rank_list, stable_rank_list, batch_size)
        #print(len(self.spectrum_divergence_list))
        self.layer_eigenvalues = layer_eigenvalues * batch_size
        self.layer_eigenweights = layer_eigenweights * batch_size

        return layer_lambdas, layer_density


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
                self.spectral_density_full_params(n_iter=n_iter, n_v=n_v, sigma=0.01)

    def log(self, logger, i):
        logger.log("spectral_entropy", self.spectrum_entropy_list, i)
        logger.log("weighted_entropy", self.weighted_entropy_list, i)
        logger.log("centroid", self.centroid_list, i)
        logger.log("spread", self.spread_list, i)
        logger.log("spectrum_divergence", self.spectrum_divergence_list, i)
        logger.log("effective_rank", self.effective_rank_list, i)
        logger.log("shapescale", self.shapescale, i)
        logger.log("stable_rank", self.stable_rank_per_group, i)

class Hessian_Calculator_old():
    def __init__(self):
        self.hessian_norms = []
        self.layer_trace = []
        self.max_eigenvalues = 0
        self.max_eigenvector_1 = None
        self.max_eigenvector_2 = None
        self.lambda_1 = 0
        self.lambda_2 = 0
        self.lambda_n = 0
        self.layer_trace = 0
        self.layer_wd_trace = None
        self.cosine_similarities = 0
        self.grad_norms = 0
        self.wd_grad_norms = 0
        self.l2_distance = 0
        self.loss_singularvalue_distance = 0
        self.loss_singularvector_distance = 0
        self.item_1 = 0
        self.item_2 = 0
        self.noise_sensitivity = 0
        self.block_sim_1 = 0
        self.block_sim_2 = 0
        self.block_sim_3 = 0

    def batch_collect(self, model, loss_fn, lam, data, target, batch_size, device='cpu'):
        model.eval()

        # item_1: f(x)f(x)^T, item_2: diag(p) - pp^T. Save in ./results/prob
        model = model.to(device)
        self.compute_item(model, data, target, batch_size)

        # sensitivity: inject noise to input, and estimate the difference of loss. Save in ./results/input
        self.compute_sensitivity(model, loss_fn, data, target, batch_size)

        # The ratio of the first singular value and the second singular value of loss. Save in ./results/distance
        self.compute_singular_ratio(model, loss_fn, data, target, batch_size)

        output = model(data)
        loss = loss_fn(output, target)
        if self.layer_wd_trace is None:
            layer_wd_trace = compute_wd_hessians_trace(model, lam, device)
            self.layer_wd_trace = layer_wd_trace
        #input_trace = compute_input_hessians_trace(model, loss, data, device)

        # Trace of Hessian. Save in ./results/hessian
        #layer_trace, self.trace_num = compute_hessians_trace(model, loss, device)
        #print(layer_trace)
        #self.layer_trace += np.array(layer_trace) * batch_size

        # Max eigenvalue. In process
        max_eigenvalue, max_eigenvector = compute_eigenvalue(model, loss, device, top_n=1)
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
        #print(lambda_1)
        self.lambda_1 += lambda_1 * batch_size
        #self.lambda_2 += lambda_2 * batch_size
        #print(layer_trace, lambda_1)
        #print(estimate_trace, estimate_eigen)
        #self.lambda_n += lambda_n * batch_size
        
        # Randomly sample part of the training data to compute the Hessian over

        # Hessian bound
        layer_hessian_quantities = compute_hessians_quantity(model, loss, device)

        #layer_hessian_quantities = np.sum(layer_hessian_quantities) / np.sqrt(config.train.hessian_log_size)
        if len(self.hessian_norms) == 0:
            self.hessian_norms = layer_hessian_quantities
        else:
            self.hessian_norms = np.maximum(self.hessian_norms, layer_hessian_quantities)
        
    
    def batch_aggregate(self, train_num):
        #self.block_sim_1 /= train_num
        #self.block_sim_2 /= train_num
        #self.block_sim_3 /= train_num
        #print(f"block sim: {self.block_sim_1}, {self.block_sim_2}, {self.block_sim_3}")
        self.item_1 /= train_num
        self.item_2 /= train_num
        self.cosine_similarities = np.mean(self.cosine_similarities / train_num)
        self.grad_norms = np.mean(self.grad_norms / train_num)
        self.wd_grad_norms = np.mean(self.wd_grad_norms / train_num)
        self.l2_distance = np.mean(self.l2_distance / train_num)
        #print(self.cosine_similarities)
        self.trace = self.layer_trace / train_num
        print(self.trace)
        self.lambda_1 = self.lambda_1 / train_num
        self.condition = self.lambda_1 / self.trace
        print("condition: ", self.condition)
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
        self.wd_trace = np.mean(self.layer_wd_trace)
        self.trace = np.mean(self.trace)

        self.noise_sensitivity /= train_num
        print(self.noise_sensitivity)
        
        self.loss_singularvector_distance = np.mean(self.loss_singularvector_distance / train_num)
        self.loss_singularvalue_distance = np.mean(self.loss_singularvalue_distance / train_num)

        hessian_quantities = np.sum(sqrt_with_neg_handling(np.array(self.hessian_norms))) / np.sqrt(train_num)
        self.train_hessianmeasurement = (hessian_quantities).item()
    
    def compute_item(self, model, data, target, batch_size):
        output = model(data)
        rep = model.get_rep(data)[:, -1]

        logits = output[:, -1]
        y = target[:, -1]
        P = F.softmax(logits)
        item_1 = torch.trace(torch.mm(rep, rep.t()))
        diag_P = torch.diag_embed(P) 
        outer_P = P.unsqueeze(2) * P.unsqueeze(1)
        H_G = diag_P - outer_P
        item_2 = torch.trace(H_G.mean(dim=0))
        self.item_1 += item_1.item()
        self.item_2 += item_2.item() * batch_size

    def compute_sensitivity(self, model, loss_fn, data, target, batch_size):
        output = model(data)
        loss = loss_fn(output, target, 'none')

        # noise_sensitivity: Estimate the sensetivity of input. Save in ./results/input
        for i in range(50):
            #noisy_output, noise_norm = model.add_noise_forward(data)
            
            #noise_sensitivity = torch.norm(noisy_output[:, -1] - output[:, -1]) / noise_norm
            #noisy_loss = F.cross_entropy(noisy_output[:, -1], target[:, -1], reduction='none')
            #noise_sensitivity = (noisy_loss - loss) / noise_norm
            noisy_output_1, noisy_output_2, noise_norm = model.add_bi_noise_forward(data)
            noisy_loss_1 = F.cross_entropy(noisy_output_1[:, -1], target[:, -1], reduction='none')
            noisy_loss_2 = F.cross_entropy(noisy_output_2[:, -1], target[:, -1], reduction='none')
            noise_sensitivity = (noisy_loss_1 + noisy_loss_2 - 2*loss)
            #noise_sensitivity = (noisy_output_1 + noisy_output_2 - output)[:, -1]

        self.noise_sensitivity += noise_sensitivity.mean().item() * batch_size

    def compute_singular_ratio(self, model, loss_fn, data, target, batch_size):
        with torch.set_grad_enabled(True), sdpa_kernel(SDPBackend.MATH):
            output = model(data)
            loss = loss_fn(output, target)
            layers = model.get_layers()
            weights = []
            trace_num = []
            layer_names = []
            for name, module in layers.items():
                weights.append(module.weight)
                layer_names.append(name)
                trace_num.append(module.weight.shape[0] * module.weight.shape[1])
            model.zero_grad()
            loss_grads = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)
            
            # Compute weight decay gradients
            lam = 1
            weight_decay_grads = [lam * w for w in weights]

            # \sigma1/\sigma2
            cosine_similarities = []
            l2_distance = []
            grad_norms = []
            wd_grad_norms = []
            loss_max_singularvector_1 = []
            loss_max_singularvector_2 = []
            loss_max_singularvalue_1 = []
            loss_max_singularvalue_2 = []
            loss_singularvector_distance = []
            loss_singularvalue_distance = []
            
            for name, grad, wd_grad in zip(layer_names, loss_grads, weight_decay_grads):
                U, S, Vh = torch.linalg.svd(grad, full_matrices=False)

                # Extract the first and second singular values
                first_singular_value = S[0].item()
                second_singular_value = S[1].item()
                loss_max_singularvalue_1.append(first_singular_value)
                loss_max_singularvalue_2.append(second_singular_value)

                # Extract the corresponding singular vectors
                # U and Vh are orthogonal matrices. Columns of U are left singular vectors.
                # Rows of Vh are the right singular vectors (V transposed).
                first_left_singular_vector = U[:, 0]
                first_right_singular_vector = Vh[0, :]
                second_left_singular_vector = U[:, 1]
                second_right_singular_vector = Vh[1, :]
                loss_max_singularvector_1.append(first_right_singular_vector)
                loss_max_singularvector_2.append(second_right_singular_vector)

                loss_singularvalue_distance.append(first_singular_value/second_singular_value)
                loss_singularvector_distance.append(torch.norm(Vh[0, :] - Vh[1, :]).item())
                # Flatten the gradients into 1D tensors
                grad_flat = grad.view(-1)
                wd_grad_flat = wd_grad.view(-1)

                # Compute the dot product and norms
                dot_product = torch.dot(grad_flat, wd_grad_flat)
                norm_grad = torch.norm(grad_flat)
                norm_wd_grad = torch.norm(wd_grad_flat)

                grad_norms.append(norm_grad.item())
                #print(norm_grad.item(), norm_wd_grad.item())
                wd_grad_norms.append(norm_wd_grad.item())

                # Compute cosine similarity with numerical stability
                cos_sim = dot_product / (norm_grad * norm_wd_grad + 1e-8)

                # Optionally, detach the cosine similarity from the graph and convert to float
                cosine_similarities.append(cos_sim.item())

                l2_distance.append(torch.norm(grad_flat - wd_grad_flat).item())

            
            self.cosine_similarities += np.array(cosine_similarities) * batch_size
            #print(self.cosine_similarities)
            self.grad_norms += np.array(grad_norms) * batch_size
            self.wd_grad_norms += np.array(wd_grad_norms) * batch_size
            self.l2_distance += np.array(l2_distance) * batch_size

            self.loss_singularvalue_distance += np.array(loss_singularvalue_distance) * batch_size
            self.loss_singularvector_distance += np.array(loss_singularvector_distance) *batch_size


"""
def get_layers(model):
    layers = OrderedDict()
    for name, module in model.named_modules():
        #if (type(module) == torch.nn.Linear) and \
        #("LayerNorm" not in name and "embeddings" not in name and "pooler" not in name):
        has_params = any(p.requires_grad for p in module.parameters(recurse=False))
        if has_params:
            print(f"Layer: {name}, Module: {module}")
            layers[name] = module
    return layers
"""
def get_layers(model):
    layers = OrderedDict()
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and any(p.requires_grad for p in module.parameters(recurse=False)):
            #if (type(module) == torch.nn.Linear) and "LayerNorm" not in name and "ln" not in name and "embeddings" not in name and "pooler" not in name:
            if "LayerNorm" not in name and "ln" not in name and "pooler" not in name:
            #print(f"Layer: {name}, Module: {module}")
                layers[name] = module
    return layers
""" Calculate Hessian Norms: (W-W^)^T (H) (W - W^s)"""


def compute_hessians_quantity(model, loss, device="cpu", state_dict = None):
    # Get parameters and gradients of corresponding layer
    layers = model.get_layers()
    weights = [module.weight for name, module in layers.items()]
    model.zero_grad()
    gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)
    
    vs = []
    for name, module in layers.items():
        weight = module.weight
        v = weight.detach().clone() - model.init_state[name+".weight"].to(weight.device)
        vs.append(v)

    model.zero_grad()    
    Hvs = torch.autograd.grad(gradients, weights, grad_outputs=vs, retain_graph=True)

    layer_hessian_quantities = [torch.sum(Hv*v).cpu().item() for (Hv, v) in zip(Hvs, vs)]
    
    out = np.array(layer_hessian_quantities)
    return out

def compute_wd_hessians_trace(model, lam, device = "cpu", maxIter=100, tol=1e-3):
    # Get parameters and gradients of corresponding layer

    layers = model.get_layers()
    weights = []
    trace_num = []
    for name, module in layers.items():
        weights.append(module.weight)
        trace_num.append(module.weight.shape[0] * module.weight.shape[1])
    
    layer_wd_trace = np.array(trace_num) * lam

    return layer_wd_trace

def compute_input_hessians_trace(model, loss, data, device = "cpu", maxIter=100, tol=1e-3):
    # Get parameters and gradients of corresponding layer
    #print(data)
    data=data.type(torch.float32)
    data.requires_grad=True
    model.zero_grad()
    gradients = torch.autograd.grad(loss, data, retain_graph=True, create_graph=True)

    layer_traces = []
    trace_vhv = []
    trace = 0.

    # Start Iterations
    for _ in range(maxIter):
        vs = torch.randint_like(data, high=2)
            
        # generate Rademacher random variables
        vs[vs == 0] = -1

        model.zero_grad()  
        Hvs = torch.autograd.grad(gradients, data, grad_outputs=vs, retain_graph=True)
        tmp_layer_traces = np.array([torch.sum(Hv*v).cpu().item() for (Hv, v) in zip(Hvs, vs)])

        layer_traces.append(tmp_layer_traces)
        trace_vhv.append(np.sum(tmp_layer_traces))

        if abs(np.mean(trace_vhv) - trace) / (abs(trace) + 1e-6) < tol:
            break
        else:
            trace = np.mean(trace_vhv)
    layer_trace = np.mean(np.array(layer_traces), axis=0)
    #avg_layer_trace = np.mean(np.array(layer_traces), axis=0) / trace_num
    data.requires_grad=False
    return layer_trace

def compute_hessians_trace(model, loss, device = "cpu", maxIter=100, tol=1e-3):
    # Get parameters and gradients of corresponding layer

    layers = model.get_layers()
    weights = []
    trace_num = []
    for name, module in layers.items():
        weights.append(module.weight)
        trace_num.append(module.weight.shape[0] * module.weight.shape[1])
    model.zero_grad()
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

        layer_traces.append(tmp_layer_traces)
        trace_vhv.append(np.sum(tmp_layer_traces))

        if abs(np.mean(trace_vhv) - trace) / (abs(trace) + 1e-6) < tol:
            break
        else:
            trace = np.mean(trace_vhv)
    layer_trace = np.mean(np.array(layer_traces), axis=0)
    #avg_layer_trace = np.mean(np.array(layer_traces), axis=0) / trace_num
    return layer_trace, np.array(trace_num)

""" Calculate Top Eigenvalue of Hessian """ 
def compute_eigenvalue(model, loss, device, maxIter=100, tol=1e-8, top_n=1):
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

def compute_small_eigenvalue(model, loss, device, maxIter=100, tol=1e-8, top_n=1):
    layers = model.get_layers()
    weights = [module.weight for name, module in layers.items()]
    model.zero_grad()
    """ use negative loss to get the minimum eigenvalue here """
    gradients = torch.autograd.grad(-loss, weights, retain_graph=True, create_graph=True)

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

def compute_hessians_distance(model, hessian_trace, sample_num, device="cpu", state_dict = None):

    layers = model.get_layers()

    model.zero_grad()
    
    vs = []
    for name, module in layers.items():
        weight = module.weight
        v = torch.norm(weight.detach().clone(), p=2)
        vs.append(v.item())

    w_dis = np.array(vs)

    h_dis = np.sum(np.sqrt(hessian_trace * w_dis**2)) / sample_num
    
    return h_dis

def get_params_grad_split(model):
    others_params, head_params = [], []
    others_grads, head_grads = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        grad_value = 0. if param.grad is None else param.grad + 0.

        if name.startswith('head'):  
            head_params.append(param)
            head_grads.append(grad_value)
        else:
            others_params.append(param)
            others_grads.append(grad_value)

    return [others_params, head_params], [others_grads, head_grads]