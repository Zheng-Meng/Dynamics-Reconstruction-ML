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
save_file_name = 'train_by_yourself'
print(save_file_name)
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

attractors_all = ['aizawa', 'bouali', 'chua', 'dadras', 'foodchain', 'four_wing', 'hastings', 'lorenz',
                  'lotka', 'rikitake', 'rossler', 'sprott_0', 'sprott_1', 'sprott_10','sprott_11',
                  'sprott_12','sprott_13', 'sprott_14', 'sprott_15', 'sprott_16', 'sprott_17', 'sprott_18',
                  'sprott_2', 'sprott_3',
                  'sprott_4', 'sprott_5', 'sprott_6', 'sprott_7','sprott_8','sprott_9', 'wang']

attractors_test = ['foodchain', 'lorenz', 'lotka']
attractors_train = [i for i in attractors_all if i not in attractors_test]

# read training and testing systems separately, and preprocess the data.
train_data = []
for train_filename in attractors_train:
    file_path = os.path.join(directory_path, 'data_' + train_filename + '.pkl')
    normalized_data, _ = utils.read_and_normalize_chaos(file_path, data_read_length)
    train_data.append(normalized_data[:, :args.dim].reshape(-1, args.dim))

test_data = []
for test_filename in attractors_test:
    file_path = os.path.join(directory_path, 'data_' + test_filename + '.pkl')
    normalized_data, _ = utils.read_and_normalize_chaos(file_path, data_read_length)
    test_data.append(normalized_data[:, :args.dim].reshape(-1, args.dim))

train_data_reshape = []
for i in range(len(attractors_train)):
    data_return = utils.reshape_data(train_data[i], args.sequence_length, args.dim)
    train_data_reshape.append(data_return)

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

# train the model
train_losses = []
for epoch in range(int(args.num_epochs)):
    for data in tqdm(train_loader):
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
        
    print(f'Epoch [{epoch+1}/{args.num_epochs}], Loss: {loss.item()}')
    train_losses.append(loss)


# Test the model on testing systems, which were not accessed during training.
model.eval()

# Dictionary to store samples from each system
system_samples = {}
system_test_losses = []

# testing_seq_length = args.sequence_length
testing_seq_length = 1500 # testing sequence length
testing_batch_size = 1

mask_ratio = 0.8 # testing missing ratio

data_all_testing = []

with torch.no_grad():
    test_losses = []
    for file_i in range(len(attractors_test)):
        data_test = test_data_reshape[file_i]
        test_dataset = TensorDataset(torch.tensor(data_test, dtype=torch.float32))
        test_loader = DataLoader(test_dataset, batch_size=testing_batch_size, shuffle=False)
        
        sample_for_system = {}
        
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
            test_losses.append(loss.item())
            # save data
            if not sample_for_system:
                sample_for_system['input'] = inputs_new[:, :, :args.dim].cpu().numpy()
                sample_for_system['target'] = targets.cpu().numpy()
                
                reconstructed = outputs.cpu().numpy()
                
                sample_for_system['reconstructed'] = reconstructed
        
        system_samples[attractors_test[file_i]] = sample_for_system
        avg_test_loss = sum(test_losses) / len(test_losses)
        system_test_losses.append(avg_test_loss)
        print(f'Test Loss: {avg_test_loss:.4f}')


train_losses_cpu = [loss.item() for loss in train_losses]
# Plot the training loss
plt.plot(train_losses_cpu)
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('./figures/Training_loss_{}.png'.format(save_file_name))
plt.show()

# Plot the testing loss
test_system_names = [name for name in attractors_test]
plt.bar(test_system_names, system_test_losses)
plt.xticks(rotation=45)
plt.title('Test Loss')
plt.xlabel('System')
plt.ylabel('Loss')
plt.savefig('./figures/Testing_loss_{}.png'.format(save_file_name))
plt.show()

plot_length = testing_seq_length  
linewidth = 3
count = 0

for filename, sample in system_samples.items():
    count += 1
    # Assuming each sample dictionary contains 3-dimensional data for input, target, and reconstructed
    input_data = sample['input'].reshape(-1, args.dim)
    reconstructed_data = sample['reconstructed'].reshape(-1, args.dim)
    target_data = sample['target'].reshape(-1, args.dim)
    
    plot_length = np.shape(input_data)[0]
    # Convert 0 in input_data to NaN for plotting
    input_data[input_data == 0] = np.nan
    fig, axes = plt.subplots(args.dim, 1, figsize=(10, 15))  
    # Plot data for each feature
    for i in range(args.dim):  # Loop through each feature
        ax = axes[i]
        ax.plot(target_data[:plot_length, i], label='Target', linestyle='--', linewidth=2.5)
        ax.plot(input_data[:plot_length, i], label='Input', marker='o')
        ax.plot(reconstructed_data[:plot_length, i], label='Reconstructed', linewidth=linewidth, alpha=0.5)
        ax.set_title(f'{filename} - Feature {i+1}')
        ax.legend()

    # Save the figure
    plt.savefig(f'./figures/{count}_{filename}.png')
    plt.show()
    plt.close(fig)

# # save the model and related parameters
# torch.save(model.state_dict(), './save_model/save_chaos_transformer_{}.pth'.format(save_file_name))

# pkl_file = open('./save_model/' + 'save_chaos_transformer_{}'.format(save_file_name)+ '.pkl', 'wb')
# pickle.dump(attractors_test, pkl_file)
# pickle.dump(save_file_name, pkl_file)
# pickle.dump(args, pkl_file)
# pkl_file.close()
















