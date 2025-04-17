# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 13:33:05 2025

@author: zmzhai
"""

import time
import numpy as np
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import transformer_encoder
import utils
import gc
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import pickle
import copy

torch.cuda.empty_cache()
gc.collect()

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

data_read_length = 500000
save_file_name = 'train_system_number'
print(save_file_name)

directory_path = './chaos_data/'

parser = argparse.ArgumentParser('Train transformer with incremental training datasets')
parser.add_argument('--dim', default=3, type=int, help='Dimension of the chaotic systems')
parser.add_argument('--input-size', default=3, type=int, help='Transformer input dimension')
parser.add_argument('--output-size', default=3, type=int, help='Transformer output dimension')
parser.add_argument('--batch-size', default=16, type=int, help='Batch size')
parser.add_argument('--num-epochs', default=50, type=int, help='Epochs') 
parser.add_argument('--sequence-length', default=3000, type=int, help='Maximum input sequence length')
parser.add_argument('--seq-start', default=1, type=int, help='Minimum input sequence length')
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')

parser.add_argument('--num-repeats', default=5, type=int, help='Number of times to repeat training per setup')
parser.add_argument('--mask-ratio-step', default=0.05, type=float, help='Testing mask ratio step size')
parser.add_argument('--hidden-size', default=512, type=int, help='Transformer hidden layer dimension')
parser.add_argument('--nhead', default=4, type=int, help='Transformer number of heads')
parser.add_argument('--num-layers', default=4, type=int, help='Transformer number of layers')
parser.add_argument('--d-model', default=128, type=int, help='Transformer projection dimension')
parser.add_argument('--dropout', default=0.2, type=float, help='Transformer dropout ratio')
parser.add_argument('--noise-level', default=0.05, type=float, help='Noise level added to the training data')
args = parser.parse_args()

testing_batch_size = 1

attractors_all = ['aizawa', 'bouali', 'chua', 'dadras', 'four_wing', 'hastings', 'rikitake', 
                  'rossler', 'sprott_0', 'sprott_1', 'sprott_2', 'sprott_3', 
                  'sprott_4', 'sprott_5', 'sprott_6', 'sprott_7', 'sprott_8', 
                  'sprott_9', 'sprott_10', 'sprott_11', 'sprott_12', 'sprott_13', 
                  'sprott_14', 'sprott_15', 'sprott_16', 'sprott_17', 'sprott_18', 'wang']

attractors_test = ['foodchain', 'lorenz', 'lotka']
train_sizes = range(1, len(attractors_all) + 1)

def train_and_evaluate(train_set, mask_ratios):
    print(f"Training on {len(train_set)} systems: {train_set}")
    train_data = []
    for train_filename in train_set:
        file_path = os.path.join(directory_path, 'data_' + train_filename + '.pkl')
        normalized_data, _ = utils.read_and_normalize_chaos(file_path, data_read_length)
        train_data.append(normalized_data[:, :args.dim].reshape(-1, args.dim))
        
    train_data_reshape = []
    for i in range(len(train_set)):
        data_return = utils.reshape_data(train_data[i], args.sequence_length, args.dim)
        train_data_reshape.append(data_return)
        
    test_data = []
    for test_filename in attractors_test:
        file_path = os.path.join(directory_path, 'data_' + test_filename + '.pkl')
        normalized_data, _ = utils.read_and_normalize_chaos(file_path, data_read_length)
        test_data.append(normalized_data[:, :args.dim].reshape(-1, args.dim))

    test_data_reshape = []
    for i in range(len(attractors_test)):
        data_return = utils.reshape_data(test_data[i], args.sequence_length, args.dim)
        test_data_reshape.append(data_return)
        
    combined_data = np.concatenate(train_data_reshape, axis=0)
    
    # create train dataLoader
    train_dataset = TensorDataset(torch.tensor(combined_data, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    # create transformer model
    model = transformer_encoder.TimeSeriesTransformer(args.input_size, args.output_size, args.d_model, args.nhead, args.num_layers, args.hidden_size, args.dropout).to(device)

    criterion = nn.MSELoss()
    smoothness_loss_fn = transformer_encoder.SmoothnessLoss(alpha=0.5, beta=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_time = time.time()
    for epoch in range(args.num_epochs):
        for data in train_loader:
            inputs = data[0].to(device)  # Shape should be [batch_size, sequence_length, input_size]
            # choose a random sequence_legnth 
            random_seq_length = random.randint(args.seq_start, args.sequence_length)
            inputs = inputs[:, :random_seq_length, :]
            targets = inputs.clone()
            # Add Gaussian noise to the inputs
            noise = torch.normal(mean=0.0, std=args.noise_level, size=inputs.shape).to(device)
            noisy_inputs = inputs + inputs * noise
            inputs = noisy_inputs
            
            inputs_new = torch.zeros((inputs.shape[0], inputs.shape[1], args.input_size)).to(device)
            
            for i in range(inputs.shape[0]):
                # generate a random number between 0.0~1.0 as the mask ratio
                mask_ratio = random.uniform(0.0, 1.0)
                numpy_input, temp_mask = utils.mask_data_transformer(inputs[i].cpu().numpy(), mask_ratio)
                
                inputs_new[i, :, :args.dim] = torch.from_numpy(numpy_input).to(device)
    
            outputs = model(inputs_new)
            loss = transformer_encoder.combined_loss_function(outputs, targets, smoothness_loss_fn, mse_weight=1.0, smoothness_weight=0.1)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    training_time = time.time() - start_time
    
    model.eval()
    test_losses = np.zeros((len(mask_ratios), len(attractors_test)))
    with torch.no_grad():
        for mask_ratio_i in range(len(mask_ratios)):
            mask_ratio = mask_ratios[mask_ratio_i]
            for file_i in range(len(attractors_test)):
                data_test = test_data_reshape[file_i]
                test_dataset = TensorDataset(torch.tensor(data_test, dtype=torch.float32))
                test_loader = DataLoader(test_dataset, batch_size=testing_batch_size, shuffle=False)
                
                total_loss = 0
                for data in test_loader:
                    inputs = data[0].to(device)
                    # choose a random sequence_legnth to train
                    random_seq_length = random.randint(args.seq_start, args.sequence_length)
                    inputs = inputs[:, :random_seq_length, :]
                    targets = copy.deepcopy(inputs)

                    inputs_new = torch.zeros((inputs.shape[0], inputs.shape[1], args.input_size)).to(device)
                    for i in range(inputs.shape[0]):
                        numpy_input, temp_mask = utils.mask_data_transformer(inputs[i].cpu().numpy(), mask_ratio)
                        
                        inputs_new[i, :, :args.dim] = torch.from_numpy(numpy_input).to(device)

                    outputs = model(inputs_new)
                    
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
                    
                avg_loss = total_loss / len(test_loader)
                test_losses[mask_ratio_i, file_i] = avg_loss
    
    print(f'training time:{training_time}')
                
    return training_time, test_losses


mask_ratios = np.arange(0.0, 1.0, args.mask_ratio_step)
results = np.zeros((len(train_sizes), args.num_repeats, len(mask_ratios), 3))
train_times = np.zeros((len(train_sizes), args.num_repeats))

for train_size_i in range(len(train_sizes)):
    train_size = train_sizes[train_size_i]
    
    for repeat in range(args.num_repeats):
        train_subset = random.sample(attractors_all, train_size)
        training_time, test_performance = train_and_evaluate(train_subset, mask_ratios)
        
        results[train_size_i, repeat, :, :] = test_performance
        train_times[train_size_i, repeat] = training_time

# Save results
with open(f'./save_results/results_{save_file_name}.pkl', 'wb') as f:
    pickle.dump(results, f)
    pickle.dump(train_times, f)

print("Experiment completed and results saved.")
