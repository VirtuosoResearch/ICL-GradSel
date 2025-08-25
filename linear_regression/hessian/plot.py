import json
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from scipy.stats import norm
from datetime import datetime

from src.configs.config import ConfigDict

plot_fig = {
    "block_sim": False,
    "effective_rank": True,
    "spectral_entropy": True,
    "weighted_entropy": True,
    "centroid": True,
    "spread": True,
    "shapescale": False,
    "stable_rank": False,
    "condition": True,
}
    

def hessian_plot(log, config, save_dir="./results/"):
    if config.mark == 'standard':
        save_dir += "standard/"
    # Train accuracy
    train_converge = log["train_converge"]["value"]
    val_converge = log["val_converge"]["value"]
    grok_start = log["grok_start"]['value']
    #print(train_converge, val_converge)
    current_time = datetime.now()
    time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print("Plotting hessian, ", log.label)
    try:
        # Hessian
        plt.plot(log["train_hessian_trace"]["iter"], log["train_hessian_trace"]["value"], label="Hessian")
        # |Train loss - val loss|
        plt.plot(log["train_wd_hessian_trace"]["iter"], log["train_wd_hessian_trace"]["value"], label="wd Hessian")
        # Plot
        if train_converge > 0:
            plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
        if val_converge > 0:
            plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
        plt.legend()
        plt.title(f"{config.task.task}")
        plt.xlabel("Gap")
        plt.ylabel("Hessian")
        plt.xscale("log", base=10)
        plt.yscale("log", base=10)
        #plt.ylim(1e-7, 1e7)
        plt.grid()
        plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
        plt.savefig(f"{save_dir}hessian/hessian_{log.label}.png", dpi=150)
        plt.draw()
        plt.close()
    except Exception as e: 
        print(e)
    try:
        # Hessian bound
        plt.plot(log["train_hessianmeasurement"]["iter"], log["train_hessianmeasurement"]["value"], label="Hessian")
        # |Train loss - val loss|
        plt.plot(log["loss_gap"]["iter"], log["loss_gap"]["value"], label="|Train Loss - Val Loss|")
        # Plot
        if train_converge > 0:
            plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
        if val_converge > 0:
            plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
        plt.legend()
        plt.title(f"{config.task.task}")
        plt.xlabel("Gap")
        plt.ylabel("Hessian")
        plt.xscale("log", base=10)
        plt.yscale("log", base=10)
        #plt.ylim(1e-7, 1e7)
        plt.grid()
        plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
        plt.savefig(f"{save_dir}gap/gap_{log.label}.png", dpi=150)
        plt.draw()
        plt.close()
    except Exception as e: 
        print(e)
    
    try:
        # layer hessian trace
        hessian_layer_trace = np.array(log["hessian_layer_trace"]["value"]).T
        #print(spectrum_divergence)
        for i in range(len(hessian_layer_trace)):
            plt.plot(log["hessian_layer_trace"]["iter"], hessian_layer_trace[i], label=f"TF-{i} and head")
        #plt.plot(log["train_block_sim_2"]["iter"], log["train_block_sim_2"]["value"], label="1 and 2")
        #plt.plot(log["train_block_sim_3"]["iter"], log["train_block_sim_3"]["value"], label="0 and 1")
        # Plot
        if train_converge > 0:
            plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
        if val_converge > 0:
            plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
        if grok_start > 0:
            plt.axvline(x=grok_start, color='yellow', linestyle='--', linewidth=1, label='grokking start')
        plt.legend()
        plt.title(f"{config.task.task}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xscale("log", base=10)
        plt.yscale("log", base=10)
        #plt.ylim(1e-7, 1e7)
        plt.grid()
        plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
        plt.savefig(f"{save_dir}hessian/hessian_{log.label}.png", dpi=150)
        plt.draw()
        plt.close()
    except Exception as e: 
        print(e)
    try:
        # input noisy gradient
        plt.plot(log["train_noise_sensitivity"]["iter"], log["train_noise_sensitivity"]["value"], label="input noisy gradient")
        # Plot
        if train_converge > 0:
            plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
        if val_converge > 0:
            plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
        plt.legend()
        plt.title(f"{config.task.task}")
        plt.xlabel("Gap")
        plt.ylabel("Hessian")
        plt.xscale("log", base=10)
        #plt.yscale("log", base=10)
        #plt.ylim(1e-7, 1e7)
        plt.grid()
        
        plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
        plt.savefig(f"{save_dir}input/input_{log.label}.png", dpi=150)
        plt.draw()
        plt.close()
    except Exception as e: 
        print(e)
    try:
        # block sim
        if plot_fig["block_sim"]:
            spectrum_divergence = np.array(log["spectrum_divergence"]["value"]).T
            #print(spectrum_divergence)
            for i in range(len(spectrum_divergence)):
                plt.plot(log["spectrum_divergence"]["iter"], spectrum_divergence[i], label=f"TF-{i} and head")
            #plt.plot(log["train_block_sim_2"]["iter"], log["train_block_sim_2"]["value"], label="1 and 2")
            #plt.plot(log["train_block_sim_3"]["iter"], log["train_block_sim_3"]["value"], label="0 and 1")
            # Plot
            if train_converge > 0:
                plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
            if val_converge > 0:
                plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
            if grok_start > 0:
                plt.axvline(x=grok_start, color='yellow', linestyle='--', linewidth=1, label='grokking start')
            plt.legend()
            plt.title(f"{config.task.task}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.xscale("log", base=10)
            plt.yscale("log", base=10)
            #plt.ylim(1e-7, 1e7)
            plt.grid()
            plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
            plt.savefig(f"{save_dir}block_sim/block_sim_{log.label}.png", dpi=150)
            plt.draw()
            plt.close()
    except Exception as e: 
        print(e)
    
    try:

        if plot_fig["spectral_entropy"]:
            # spectral entropy
            spectral_entropy = np.array(log["spectral_entropy"]["value"]).T
            #print(spectrum_divergence)
            for i in range(len(spectral_entropy)-1):
                plt.plot(log["spectral_entropy"]["iter"], spectral_entropy[i], label=f"TF-{i}")
            plt.plot(log["spectral_entropy"]["iter"], spectral_entropy[-1], label=f"head")
            #plt.plot(log["train_block_sim_2"]["iter"], log["train_block_sim_2"]["value"], label="1 and 2")
            #plt.plot(log["train_block_sim_3"]["iter"], log["train_block_sim_3"]["value"], label="0 and 1")
            # Plot
            if train_converge > 0:
                plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
            if val_converge > 0:
                plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
            if grok_start > 0:
                plt.axvline(x=grok_start, color='yellow', linestyle='--', linewidth=1, label='grokking start')
            plt.legend()
            plt.title(f"{config.task.task}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.xscale("log", base=10)
            plt.yscale("log", base=10)
            #plt.ylim(1e-7, 1e7)
            plt.grid()
            plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
            plt.savefig(f"{save_dir}entropy/entropy_{log.label}.png", dpi=150)
            plt.draw()
            plt.close()
    except Exception as e: 
        print(e)
    
    try:
        if plot_fig["weighted_entropy"]:
            # weighted entropy
            weighted_entropy = np.array(log["weighted_entropy"]["value"]).T
            #print(spectrum_divergence)
            for i in range(len(weighted_entropy)-1):
                plt.plot(log["weighted_entropy"]["iter"], weighted_entropy[i], label=f"TF-{i}")
            plt.plot(log["weighted_entropy"]["iter"], weighted_entropy[-1], label=f"head")
            #plt.plot(log["train_block_sim_2"]["iter"], log["train_block_sim_2"]["value"], label="1 and 2")
            #plt.plot(log["train_block_sim_3"]["iter"], log["train_block_sim_3"]["value"], label="0 and 1")
            # Plot
            if train_converge > 0:
                plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
            if val_converge > 0:
                plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
            if grok_start > 0:
                plt.axvline(x=grok_start, color='yellow', linestyle='--', linewidth=1, label='grokking start')
            plt.legend()
            plt.title(f"{config.task.task}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.xscale("log", base=10)
            plt.yscale("log", base=10)
            #plt.ylim(1e-7, 1e7)
            plt.grid()
            plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
            plt.savefig(f"{save_dir}entropy/weighted_{log.label}.png", dpi=150)
            plt.draw()
            plt.close()
    except Exception as e: 
        print(e)
    
    try:
        if plot_fig["centroid"]:
            # centroid entropy
            centroid = np.array(log["centroid"]["value"]).T
            #print(spectrum_divergence)
            for i in range(len(centroid)-1):
                plt.plot(log["centroid"]["iter"], centroid[i], label=f"TF-{i}")
            plt.plot(log["centroid"]["iter"], centroid[-1], label=f"head")
            #plt.plot(log["train_block_sim_2"]["iter"], log["train_block_sim_2"]["value"], label="1 and 2")
            #plt.plot(log["train_block_sim_3"]["iter"], log["train_block_sim_3"]["value"], label="0 and 1")
            # Plot
            if train_converge > 0:
                plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
            if val_converge > 0:
                plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
            if grok_start > 0:
                plt.axvline(x=grok_start, color='yellow', linestyle='--', linewidth=1, label='grokking start')
            plt.legend()
            plt.title(f"{config.task.task}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.xscale("log", base=10)
            plt.yscale("log", base=10)
            #plt.ylim(1e-7, 1e7)
            plt.grid()
            plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
            plt.savefig(f"{save_dir}entropy/centroid_{log.label}.png", dpi=150)
            plt.draw()
            plt.close()
    except Exception as e: 
        print(e)

    try:
        if plot_fig["spread"]:
            # centroid entropy
            spread = np.array(log["spread"]["value"]).T
            #print(spectrum_divergence)
            for i in range(len(spread)-1):
                plt.plot(log["spread"]["iter"], spread[i], label=f"TF-{i}")
            plt.plot(log["spread"]["iter"], spread[-1], label=f"head")
            #plt.plot(log["train_block_sim_2"]["iter"], log["train_block_sim_2"]["value"], label="1 and 2")
            #plt.plot(log["train_block_sim_3"]["iter"], log["train_block_sim_3"]["value"], label="0 and 1")
            # Plot
            if train_converge > 0:
                plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
            if val_converge > 0:
                plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
            if grok_start > 0:
                plt.axvline(x=grok_start, color='yellow', linestyle='--', linewidth=1, label='grokking start')
            plt.legend()
            plt.title(f"{config.task.task}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.xscale("log", base=10)
            plt.yscale("log", base=10)
            #plt.ylim(1e-7, 1e7)
            plt.grid()
            plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
            plt.savefig(f"{save_dir}entropy/spread_{log.label}.png", dpi=150)
            plt.draw()
            plt.close()
    except Exception as e: 
        print(e)

    try:
        if plot_fig["effective_rank"]:
            # effective rank
            er = np.array(log["effective_rank"]["value"]).T
            #print(spectrum_divergence)
            for i in range(len(weighted_entropy)-1):
                plt.plot(log["effective_rank"]["iter"], er[i], label=f"TF-{i}")
            plt.plot(log["effective_rank"]["iter"], er[-1], label=f"head")
            #plt.plot(log["train_block_sim_2"]["iter"], log["train_block_sim_2"]["value"], label="1 and 2")
            #plt.plot(log["train_block_sim_3"]["iter"], log["train_block_sim_3"]["value"], label="0 and 1")
            # Plot
            if train_converge > 0:
                plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
            if val_converge > 0:
                plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
            if grok_start > 0:
                plt.axvline(x=grok_start, color='yellow', linestyle='--', linewidth=1, label='grokking start')
            plt.legend()
            plt.title(f"{config.task.task}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.xscale("log", base=10)
            plt.yscale("log", base=10)
            #plt.ylim(1e-7, 1e7)
            plt.grid()
            plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
            plt.savefig(f"{save_dir}entropy/effective_rank_{log.label}.png", dpi=150)
            plt.draw()
            plt.close()
    except Exception as e: 
        print(e)
    
    try:
        if plot_fig["shapescale"]:
            # effective rank
            shapescale = np.array(log["shapescale"]["value"]).T
            #print(spectrum_divergence)
            for i in range(len(weighted_entropy)-1):
                plt.plot(log["shapescale"]["iter"], shapescale[i], label=f"TF-{i}")
            plt.plot(log["shapescale"]["iter"], shapescale[-1], label=f"head")
            #plt.plot(log["train_block_sim_2"]["iter"], log["train_block_sim_2"]["value"], label="1 and 2")
            #plt.plot(log["train_block_sim_3"]["iter"], log["train_block_sim_3"]["value"], label="0 and 1")
            # Plot
            if train_converge > 0:
                plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
            if val_converge > 0:
                plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
            if grok_start > 0:
                plt.axvline(x=grok_start, color='yellow', linestyle='--', linewidth=1, label='grokking start')
            plt.legend()
            plt.title(f"{config.task.task}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.xscale("log", base=10)
            plt.yscale("log", base=10)
            #plt.ylim(1e-7, 1e7)
            plt.grid()
            plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
            plt.savefig(f"{save_dir}entropy/shapescale_{log.label}.png", dpi=150)
            plt.draw()
            plt.close()
    except Exception as e: 
        print(e)

    try:
        if plot_fig["stable_rank"]:
            # effective rank
            stable_rank = np.array(log["stable_rank"]["value"]).T
            #print(spectrum_divergence)
            for i in range(len(weighted_entropy)-1):
                plt.plot(log["stable_rank"]["iter"], stable_rank[i], label=f"TF-{i}")
            plt.plot(log["stable_rank"]["iter"], stable_rank[-1], label=f"head")
            #plt.plot(log["train_block_sim_2"]["iter"], log["train_block_sim_2"]["value"], label="1 and 2")
            #plt.plot(log["train_block_sim_3"]["iter"], log["train_block_sim_3"]["value"], label="0 and 1")
            # Plot
            if train_converge > 0:
                plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
            if val_converge > 0:
                plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
            if grok_start > 0:
                plt.axvline(x=grok_start, color='yellow', linestyle='--', linewidth=1, label='grokking start')
            plt.legend()
            plt.title(f"{config.task.task}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.xscale("log", base=10)
            plt.yscale("log", base=10)
            #plt.ylim(1e-7, 1e7)
            plt.grid()
            plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
            plt.savefig(f"{save_dir}entropy/stable_rank_{log.label}.png", dpi=150)
            plt.draw()
            plt.close()
    except Exception as e: 
        print(e)
    try:
        if plot_fig["condition"]:
            # effective rank
            condition = np.array(log["condition"]["value"]).T
            #print(spectrum_divergence)
            for i in range(len(weighted_entropy)-1):
                plt.plot(log["condition"]["iter"], condition[i], label=f"TF-{i}")
            plt.plot(log["condition"]["iter"], condition[-1], label=f"head")
            #plt.plot(log["train_block_sim_2"]["iter"], log["train_block_sim_2"]["value"], label="1 and 2")
            #plt.plot(log["train_block_sim_3"]["iter"], log["train_block_sim_3"]["value"], label="0 and 1")
            # Plot
            if train_converge > 0:
                plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
            if val_converge > 0:
                plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
            if grok_start > 0:
                plt.axvline(x=grok_start, color='yellow', linestyle='--', linewidth=1, label='grokking start')
            plt.legend()
            plt.title(f"{config.task.task}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.xscale("log", base=10)
            plt.yscale("log", base=10)
            #plt.ylim(1e-7, 1e7)
            plt.grid()
            plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
            plt.savefig(f"{save_dir}entropy/condition_{log.label}.png", dpi=150)
            plt.draw()
            plt.close()
    except Exception as e: 
        print(e)
    try:
        # Determine the range for the lambda grid
        iters = str(log['values_full']['iter'][-1])
        sigma = 0.01
        grid_size = 10000
        flat_eigen = log['values_full']['value'][-1]
        flat_weight = log['weights_full']['value'][-1]
        lambda_min = min(flat_eigen) - 1.0
        lambda_max = max(flat_eigen) + 1.0
        #lambda_min = -4
        #lambda_max = 4
        
        # Create a lambda grid
        lambdas = np.linspace(lambda_min, lambda_max, grid_size)
        delta_lambda = lambdas[1] - lambdas[0]
        
        # Initialize the total density
        total_density = np.zeros_like(lambdas)

        # Sum all contributions without plotting individual ones
        for eig, w in zip(flat_eigen, flat_weight):
            total_density += w * norm.pdf(lambdas, loc=eig, scale=sigma)
        
        # Normalize the total density
        total_density /= np.sum(total_density) * delta_lambda
        
        # Plot the total spectral density
        plt.plot(lambdas, total_density, color='blue', linewidth=2)
        
        plt.xlabel('Eigenvalue (λ)')
        plt.ylabel('Density')
        plt.yscale("log", base=10)
        plt.ylim(bottom=1e-10)
        plt.title(f"Full spec. {iters}")
        plt.legend()
        plt.grid(True)
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        time_str = f"time: {formatted_time}"
        plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
        plt.savefig(f"{save_dir}spectrum/full_spectrum_{log.label}_{iters}.png", dpi=150)
        plt.draw()
        plt.close()
    except Exception as e:
        print(e)
    try:
        # Determine the range for the lambda grid
        iters = str(log['values_head']['iter'][-1])
        sigma = 0.01
        grid_size = 10000
        flat_eigen = log['values_head']['value'][-1]
        flat_weight = log['weights_head']['value'][-1]
        lambda_min = min(flat_eigen) - 1.0
        lambda_max = max(flat_eigen) + 1.0
        #lambda_min = -4
        #lambda_max = 4
        
        # Create a lambda grid
        lambdas = np.linspace(lambda_min, lambda_max, grid_size)
        delta_lambda = lambdas[1] - lambdas[0]
        
        # Initialize the total density
        total_density = np.zeros_like(lambdas)

        # Sum all contributions without plotting individual ones
        for eig, w in zip(flat_eigen, flat_weight):
            total_density += w * norm.pdf(lambdas, loc=eig, scale=sigma)
        
        # Normalize the total density
        total_density /= np.sum(total_density) * delta_lambda
        
        # Plot the total spectral density
        plt.plot(lambdas, total_density, color='blue', linewidth=2)
        
        plt.xlabel('Eigenvalue (λ)')
        plt.ylabel('Density')
        plt.yscale("log", base=10)
        plt.ylim(bottom=1e-10)
        plt.title(f"Full spec. {iters}")
        plt.legend()
        plt.grid(True)
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        time_str = f"time: {formatted_time}"
        plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
        plt.savefig(f"{save_dir}spectrum/head_spectrum_{log.label}_{iters}.png", dpi=150)
        plt.draw()
        plt.close()
    except Exception as e:
        print(e)
    
    try:
        if True:
            # pos and neg trace
            #print(log["trace_pos"]["value"])
            #print(log["trace_pos"]["iter"])
            plt.plot(log["trace_pos"]["iter"], log["trace_pos"]["value"], label=f"pos trace")
            plt.plot(log["trace_neg"]["iter"], log["trace_neg"]["value"], label=f"neg trace")
            #plt.plot(log["train_block_sim_2"]["iter"], log["train_block_sim_2"]["value"], label="1 and 2")
            #plt.plot(log["train_block_sim_3"]["iter"], log["train_block_sim_3"]["value"], label="0 and 1")
            # Plot
            if train_converge > 0:
                plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
            if val_converge > 0:
                plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
            if grok_start > 0:
                plt.axvline(x=grok_start, color='yellow', linestyle='--', linewidth=1, label='grokking start')
            plt.legend()
            plt.title(f"{config.task.task}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.xscale("log", base=10)
            plt.yscale("log", base=10)
            #plt.ylim(1e-7, 1e7)
            plt.grid()
            plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
            plt.savefig(f"{save_dir}hessian/posneg_{log.label}.png", dpi=150)
            plt.draw()
            plt.close()
    except Exception as e: 
        print(e)
    try:
        if True:
            # abs weighted entropy
            plt.plot(log["abs_weighted_entropy"]["iter"], log["abs_weighted_entropy"]["value"], label=f"abs weighted entropy")
            #plt.plot(log["train_block_sim_2"]["iter"], log["train_block_sim_2"]["value"], label="1 and 2")
            #plt.plot(log["train_block_sim_3"]["iter"], log["train_block_sim_3"]["value"], label="0 and 1")
            # Plot
            if train_converge > 0:
                plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
            if val_converge > 0:
                plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
            if grok_start > 0:
                plt.axvline(x=grok_start, color='yellow', linestyle='--', linewidth=1, label='grokking start')
            plt.legend()
            plt.title(f"{config.task.task}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.xscale("log", base=10)
            plt.yscale("log", base=10)
            #plt.ylim(1e-7, 1e7)
            plt.grid()
            plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
            plt.savefig(f"{save_dir}entropy/abs_{log.label}.png", dpi=150)
            plt.draw()
            plt.close()
    except Exception as e: 
        print(e)