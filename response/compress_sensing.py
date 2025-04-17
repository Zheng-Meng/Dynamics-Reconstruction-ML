# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 10:46:42 2025

@author: zmzhai
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from sklearn.linear_model import LassoLars
import os

def load_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data['inputs'], data['targets']

def reconstruct_single_dct(signal, mask, alpha=1e-5):
    N = len(signal)
    Phi = dct(np.eye(N), norm='ortho')
    A = Phi[mask]
    y = signal[mask]
    
    mean_val = y.mean()
    y_centered = y - mean_val
    
    model = LassoLars(alpha=alpha, max_iter=1000)
    model.fit(A, y_centered)
    x_hat = model.coef_
    
    recovered = idct(x_hat, norm='ortho') + mean_val
    return recovered

def run_cs_on_dataset(inputs, targets, alpha=1e-5, plot_example=False, sample_idx=0, dim_idx=0):
    num_samples, T, D = inputs.shape
    reconstructed = np.zeros_like(inputs)

    mse_list = []
    for i in range(num_samples):
        if i > 100: # for testing
            break
        for d in range(D):
            mask = inputs[i, :, d] != 0
            rec = reconstruct_single_dct(inputs[i, :, d], mask, alpha=alpha)
            reconstructed[i, :, d] = rec

        mse = np.mean((reconstructed[i, :, :] - targets[i, :, :]) ** 2)
        mse_list.append(mse)

    mse = np.mean(mse_list)
    std = np.std(mse_list)

    if plot_example:
        inp = inputs[sample_idx, :, dim_idx]
        tgt = targets[sample_idx, :, dim_idx]
        rec = reconstructed[sample_idx, :, dim_idx]

        plt.figure(figsize=(12, 4))
        plt.plot(tgt, label="Ground Truth", linewidth=2)
        plt.plot(inp, 'ko', markersize=2, label="Observed")
        plt.plot(rec, '--', label="CS-Reconstruction")
        plt.legend()
        plt.title(f"Compressed Sensing Reconstruction (Sample {sample_idx}, Dim {dim_idx})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"cs_reconstruction_{sample_idx}_{dim_idx}.png")
        plt.close()

    return mse, std, reconstructed

def run_on_file(file_path, alpha=1e-5, name="System", plot=False):
    print(f"\n=== Running CS on: {name} ===")
    inputs, targets = load_data(file_path)

    num_samples, T, D = inputs.shape
    flattened = inputs.reshape(-1, D)  # Shape becomes (133*3000, 3)
    targets_flattened = targets.reshape(-1, D)

    # Calculate how many complete chunks of 2000 we can get
    total_timesteps = flattened.shape[0]
    new_T = 2000
    num_chunks = total_timesteps // new_T

    # Truncate to use only complete chunks
    truncated = flattened[:num_chunks * new_T]
    # Reshape to the desired format
    reshaped = truncated.reshape(-1, new_T, D)
    targets_reshaped = targets_flattened[:num_chunks * new_T].reshape(-1, new_T, D)

    mse, std, _ = run_cs_on_dataset(reshaped, targets_reshaped, alpha=alpha, plot_example=plot)
    print(f"CS MSE: {mse:.6f} (std: {std:.6f})")
    
    return mse, std

def scan_mask_ratios(systems, mask_ratios, alpha=1e-5):
    # you need to read/provide the inputs with missing data here
    save_dir = '../irregular_time_series_1.3/save_data/transformer_iter'
    results_dir = './save_data/cs_results'
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    for system in systems:
        print(f"\n========== Processing {system} system ==========")
        
        # Initialize results for this system
        system_results = {'mask_ratios': [], 'mse': [], 'std': []}
        
        for mask_ratio in mask_ratios:
            filename = f"system_{system}_mask_{round(mask_ratio,2)}.pkl"
            file_path = os.path.join(save_dir, filename)
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"File {file_path} not found. Skipping mask_ratio = {mask_ratio}")
                continue
            
            # Run CS reconstruction for this mask ratio
            plot = False  # Only plot example for mask_ratio = 0.6
            mse, std = run_on_file(file_path, alpha=alpha, name=f"{system}_{mask_ratio}", plot=plot)
            
            # Store results
            system_results['mask_ratios'].append(mask_ratio)
            system_results['mse'].append(mse)
            system_results['std'].append(std)
        
        # Save results for this system to a pickle file
        with open(os.path.join(results_dir, f'cs_performance_{system}.pkl'), 'wb') as f:
            pickle.dump(system_results, f)
        
        # Plot MSE vs mask ratio for this system
        plt.figure(figsize=(10, 6))
        plt.errorbar(system_results['mask_ratios'], system_results['mse'], 
                     yerr=system_results['std'], marker='o', capsize=5)
        plt.xlabel('Mask Ratio')
        plt.ylabel('MSE')
        plt.title(f'Compressed Sensing Performance - {system.capitalize()} System')
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, f'cs_performance_{system}.png'))
        plt.close()

# Main execution
if __name__ == "__main__":
    # Define systems and mask ratios to scan
    systems = ['lorenz', 'foodchain', 'lotka']
    mask_ratios = np.arange(0.0, 1.0, 0.05)
    mask_ratios = [round(r, 2) for r in mask_ratios]  # Round to 2 decimal places
    
    # Run the scan
    scan_mask_ratios(systems, mask_ratios, alpha=1e-5) 