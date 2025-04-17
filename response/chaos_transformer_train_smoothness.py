# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:37:50 2024

@author: zmzhai
"""

import numpy as np
import os
import random
import matplotlib.pyplot as plt
import pickle
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import transformer_encoder
import utils
import gc
from tqdm import tqdm
import argparse

torch.cuda.empty_cache()
gc.collect()

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

data_read_length = 500000
save_file_name = 'train_smoothness_iter'
print(save_file_name)
index_name = 0
# read and preprocess all data from the directory 
directory_path = './chaos_data/'

# set Transformer hyperparameters
parser = argparse.ArgumentParser('Train transformer on chaotic systems')
parser.add_argument('--logdir', default='logdir', help='Folder to store everything/load')

parser.add_argument('--dim', default=3, type=int, help='Dimension of the chaotic systems')
parser.add_argument('--input-size', default=3, type=int, help='Transformer input dimension')
parser.add_argument('--output-size', default=3, type=int, help='Transformer output dimension')
parser.add_argument('--hidden-size', default=512, type=int, help='Transformer hidden layer dimension')
parser.add_argument('--nhead', default=4, type=int, help='Transformer number of heads')
parser.add_argument('--num-layers', default=4, type=int, help='Transformer number of layers')
parser.add_argument('--d-model', default=128, type=int, help='Transformer projection dimension')
parser.add_argument('--dropout', default=0.2, type=float, help='Transformer drop out ratio')
parser.add_argument('--noise-level', default=0.05, type=float, help='Noise level added to the training data')

parser.add_argument('--seq-start', default=1, type=int, help='Minimum input sequence length')
parser.add_argument('--sequence-length', default=3000, type=int, help='Maximum input sequence length')
parser.add_argument('--batch-size', default=16, type=int, help='Batch size')
parser.add_argument('--num-epochs', default=50, type=int, help='Epochs')

parser.add_argument('--lr', default=0.001, type=float, help='learning rate')

args = parser.parse_args()
print(args)

# Define smoothness weights to test
smoothness_weights = [0, 0.1, 0.5, 1]
mask_ratio_set = np.arange(0.0, 1.01, 0.02)
num_iterations = 5  # Number of iterations for each configuration

attractors_all = ['aizawa', 'bouali', 'chua', 'dadras', 'foodchain', 'four_wing', 'hastings', 'lorenz',
                  'lotka', 'rikitake', 'rossler', 'sprott_0', 'sprott_1', 'sprott_10','sprott_11',
                  'sprott_12','sprott_13', 'sprott_14', 'sprott_15', 'sprott_16', 'sprott_17', 'sprott_18',
                  'sprott_2', 'sprott_3',
                  'sprott_4', 'sprott_5', 'sprott_6', 'sprott_7','sprott_8','sprott_9', 'wang']

attractors_test = ['foodchain', 'lorenz', 'lotka']
attractors_train = [i for i in attractors_all if i not in attractors_test]

# Initialize results matrix
mse_results = np.zeros((len(smoothness_weights), len(mask_ratio_set), len(attractors_test), num_iterations))
std_results = np.zeros((len(smoothness_weights), len(mask_ratio_set), len(attractors_test), num_iterations))

# Create save directories
save_dir = './save_data/smoothness_test'
model_save_dir = './save_data/smoothness_models'
os.makedirs(save_dir, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)

# Main training and testing loop
for smoothness_idx, smoothness_weight in enumerate(smoothness_weights):
    print(f"\nTesting smoothness weight: {smoothness_weight}")
    
    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}/{num_iterations}")
        
        # Train the model
        train_data = []
        for train_filename in attractors_train:
            file_path = os.path.join(directory_path, 'data_' + train_filename + '.pkl')
            normalized_data, _ = utils.read_and_normalize_chaos(file_path, data_read_length)
            train_data.append(normalized_data[:, :args.dim].reshape(-1, args.dim))

        train_data_reshape = []
        for i in range(len(attractors_train)):
            data_return = utils.reshape_data(train_data[i], args.sequence_length, args.dim)
            train_data_reshape.append(data_return)
            
        combined_data = np.concatenate(train_data_reshape, axis=0)

        # create train dataLoader
        train_dataset = TensorDataset(torch.tensor(combined_data, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

        # create transformer model
        model = transformer_encoder.TimeSeriesTransformer(args.input_size, args.output_size, args.d_model, 
                                                        args.nhead, args.num_layers, args.hidden_size, 
                                                        args.dropout).to(device)

        criterion = nn.MSELoss()
        smoothness_loss_fn = transformer_encoder.SmoothnessLoss(alpha=0.5, beta=0.5)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # train the model
        for epoch in range(int(args.num_epochs)):
            for data in tqdm(train_loader):
                inputs = data[0].to(device)
                
                random_seq_length = random.randint(args.seq_start, args.sequence_length)
                inputs = inputs[:, :random_seq_length, :]
                targets = inputs.clone()
                
                noise = torch.normal(mean=0.0, std=args.noise_level, size=inputs.shape).to(device)
                noisy_inputs = inputs + inputs * noise
                inputs = noisy_inputs
                
                inputs_new = torch.zeros((inputs.shape[0], inputs.shape[1], args.input_size)).to(device)
                
                for i in range(inputs.shape[0]):
                    mask_ratio = random.uniform(0.0, 1.0)
                    numpy_input, temp_mask = utils.mask_data_transformer(inputs[i].cpu().numpy(), mask_ratio)
                    inputs_new[i, :, :args.dim] = torch.from_numpy(numpy_input).to(device)

                outputs = model(inputs_new)
                loss = transformer_encoder.combined_loss_function(outputs, targets, smoothness_loss_fn, 
                                                               mse_weight=1.0, smoothness_weight=smoothness_weight)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Save the first trained model for each smoothness weight
        if iteration == 0:
            # Save model state dict
            model_save_name = f'save_chaos_transformer_smoothness_{smoothness_weight}_{index_name}'
            torch.save(model.state_dict(), f'./save_data/smoothness_models/{model_save_name}.pth')
            
            # Save related parameters
            pkl_file = open(f'./save_data/smoothness_models/{model_save_name}.pkl', 'wb')
            pickle.dump(attractors_test, pkl_file)
            pickle.dump(model_save_name, pkl_file)
            pickle.dump(args, pkl_file)
            pkl_file.close()

        # Test the model with different mask ratios
        for mask_idx, mask_ratio in enumerate(mask_ratio_set):
            for system_idx, test_filename in enumerate(attractors_test):
                test_losses = []
                
                file_path = os.path.join(directory_path, 'data_' + test_filename + '.pkl')
                normalized_data, _ = utils.read_and_normalize_chaos(file_path, data_read_length)
                test_data = normalized_data[:, :args.dim].reshape(-1, args.dim)
                
                test_data_reshape = utils.reshape_data(test_data, args.sequence_length, args.dim)
                test_dataset = TensorDataset(torch.tensor(test_data_reshape, dtype=torch.float32))
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
                
                with torch.no_grad():
                    for data in test_loader:
                        inputs = data[0].to(device)
                        targets = inputs.clone()
                        
                        inputs_new = torch.zeros((inputs.shape[0], inputs.shape[1], args.input_size)).to(device)
                        for i in range(inputs.shape[0]):
                            numpy_input, temp_mask = utils.mask_data_transformer(inputs[i].cpu().numpy(), mask_ratio)
                            inputs_new[i, :, :args.dim] = torch.from_numpy(numpy_input).to(device)
                        
                        outputs = model(inputs_new)
                        loss = criterion(outputs, targets)
                        test_losses.append(loss.item())
                
                # Calculate MSE and STD for this mask ratio, system, and iteration
                mse = np.mean(test_losses)
                std = np.std(test_losses)
                
                mse_results[smoothness_idx, mask_idx, system_idx, iteration] = mse
                std_results[smoothness_idx, mask_idx, system_idx, iteration] = std
                
                print(f"System: {test_filename}, Mask ratio: {mask_ratio:.2f}, MSE: {mse:.6f}, STD: {std:.6f}")

# Save results
results = {
    'mse_results': mse_results,
    'std_results': std_results,
    'smoothness_weights': smoothness_weights,
    'mask_ratio_set': mask_ratio_set,
    'attractors_test': attractors_test,
    'num_iterations': num_iterations
}

with open(os.path.join(save_dir, f'smoothness_test_results_{index_name}.pkl'), 'wb') as f:
    pickle.dump(results, f)

