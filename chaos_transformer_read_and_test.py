# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 11:11:23 2024

@author: zmzhai
"""


import numpy as np
import os
import pickle
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import transformer_encoder
import utils
import gc
import matplotlib.pyplot as plt

torch.cuda.empty_cache()
gc.collect()

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# read and test length for each target system: 400000
data_read_length = 400000
directory_path = './chaos_data/'
save_file_name = 'trained' 
save_dir = './save_data/transformer_iter_{}'.format(save_file_name)
os.makedirs(save_dir, exist_ok=True)

# whether we want to save the data of inputs, targets, and outputs or not.
save_data_to_file = True

pkl_file = open('./save_model/' + 'save_chaos_transformer_{}'.format(save_file_name)+ '.pkl', 'rb')
attractors_test = pickle.load(pkl_file)
save_file_name = pickle.load(pkl_file)
args = pickle.load(pkl_file)
pkl_file.close()

# read testing systems
test_data = []
for test_filename in attractors_test:
    file_path = os.path.join(directory_path, 'data_' + test_filename + '.pkl')
    normalized_data, _ = utils.read_and_normalize_chaos(file_path, data_read_length)
    test_data.append(normalized_data)

# add the test of counter-example: stochastic signal
stochastic_signal = utils.generate_stochastic_signals(data_read_length)
attractors_test.append('stochastic')
test_data.append(stochastic_signal)

test_num = len(attractors_test)
def save_data(inputs, targets, outputs, sequence_length, mask_ratio, system_name):
    filename = f"system_{system_name}_seq_{sequence_length}_mask_{round(mask_ratio,2)}.pkl"
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump({'inputs': inputs, 'targets': targets, 'outputs': outputs}, f)

# create model
model = transformer_encoder.TimeSeriesTransformer(args.input_size, args.output_size, args.d_model, args.nhead, args.num_layers, args.hidden_size, args.dropout).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

model.load_state_dict(torch.load('./save_model/save_chaos_transformer_{}.pth'.format(save_file_name)))

# grid scan
# seq_length_set = range(25, 3010, 25)
# mask_ratio_set = np.arange(0.0, 1.01, 0.01)

seq_length_set = [2000]
mask_ratio_set = [0.8]

loss_ml = np.zeros((len(seq_length_set), len(mask_ratio_set), test_num))
loss_std_ml = np.zeros((len(seq_length_set), len(mask_ratio_set), test_num))

for seq_length_idx in range(len(seq_length_set)):
    sequence_length = seq_length_set[seq_length_idx]

    test_data_reshape = []
    for i in range(len(attractors_test)):
        data_return = utils.reshape_data(test_data[i], sequence_length, args.dim)
        test_data_reshape.append(data_return)
    
    testing_seq_length = sequence_length
    testing_batch_size = 1
    
    for mask_ratio_idx in range(len(mask_ratio_set)):
        mask_ratio = mask_ratio_set[mask_ratio_idx]
        print(f'start for seq length {sequence_length} and mask ratio {mask_ratio}')
        
        model.eval()
        # recovery task
        with torch.no_grad():
            for system_i in range(test_num):
                data_test = test_data_reshape[system_i]
                test_dataset = TensorDataset(torch.tensor(data_test, dtype=torch.float32))
                test_loader = DataLoader(test_dataset, batch_size=testing_batch_size, shuffle=False)
                
                accumulated_inputs, accumulated_targets, accumulated_outputs = [], [], []
                
                loss_file_ml = []
                loss_std_file_ml = []

                for data in test_loader:
                    inputs = data[0].to(device)
                    targets = copy.deepcopy(inputs)
        
                    inputs_new = torch.zeros((inputs.shape[0], inputs.shape[1], args.input_size)).to(device)
                    for i in range(inputs.shape[0]):
                        numpy_input, temp_mask = utils.mask_data_transformer(inputs[i].cpu().numpy(), mask_ratio)
                        
                        inputs_new[i, :, :args.dim] = torch.from_numpy(numpy_input).to(device)

                    outputs = model(inputs_new)
                    loss = criterion(outputs, targets)
                    loss_file_ml.append(loss.cpu())

                    # Accumulate the batch data
                    accumulated_inputs.append(inputs_new.cpu().numpy())
                    accumulated_targets.append(targets.cpu().numpy())
                    accumulated_outputs.append(outputs.cpu().numpy())
                
                combined_inputs = np.concatenate(accumulated_inputs, axis=0)
                combined_targets = np.concatenate(accumulated_targets, axis=0)
                combined_outputs = np.concatenate(accumulated_outputs, axis=0)
                
                if save_data_to_file:
                    save_data(combined_inputs, combined_targets, combined_outputs, sequence_length, mask_ratio, attractors_test[system_i])
                    
                loss_avg_ml, loss_std = np.mean(loss_file_ml), np.std(loss_file_ml)
                
                loss_ml[seq_length_idx, mask_ratio_idx, system_i] = loss_avg_ml
                loss_std_ml[seq_length_idx, mask_ratio_idx, system_i] = loss_std

                print(f'calculate loss of {attractors_test[system_i]} for seq length {sequence_length} and mask ratio {mask_ratio}')

# save statistics
pkl_file = open('./save_data/' + 'save_loss_seq_length_mask_ratio'+ '.pkl', 'wb')
pickle.dump(loss_ml, pkl_file)
pickle.dump(loss_std_ml, pkl_file)
pickle.dump(seq_length_set, pkl_file)
pickle.dump(mask_ratio_set, pkl_file)
pkl_file.close()

# read saved file and plot
# system_name = attractors_test[0] # 'foodchain'
system_name = 'foodchain'
plot_seq_length = sequence_length
plot_mask_ratio = 0.8

filename = f"system_{system_name}_seq_{sequence_length}_mask_{round(plot_mask_ratio,2)}.pkl"
filepath = os.path.join(save_dir, filename)

with open(filepath, 'rb') as f:
    data = pickle.load(f)

os.makedirs(save_dir + '/figures', exist_ok=True)

def plot_system_examples(iter_=0, plot_length=500):
    inputs, targets, outputs = data['inputs'], data['targets'], data['outputs']
    inputs, targets, outputs = inputs[iter_, :, :], targets[iter_, :, :], outputs[iter_,:,:]
    
    inputs_copy = copy.deepcopy(inputs)
    inputs_copy[inputs_copy == 0] = np.nan
    
    if args.dim == 1:
        fig, ax = plt.subplots(args.dim, 1, figsize=(10, 7), constrained_layout=True)
        
        ax.plot(targets[100:100+plot_length, 0], linestyle='--', linewidth=5, color='grey', label='Ground truth')
        ax.plot(outputs[100:100+plot_length, 0], linestyle='-', linewidth=5, color='b', label='Reconstructed')
        
        ax.plot(inputs_copy[100:100+plot_length, 0], marker='o', markersize=10, color='k', linestyle='', label='Observed')
    else:
        fig, axes = plt.subplots(args.dim, 1, figsize=(10, 7), constrained_layout=True)
        
        for i in range(args.dim):
            ax = axes[i]
            ax.plot(targets[100:100+plot_length, i], linestyle='--', linewidth=5, color='grey', label='Ground truth')
            ax.plot(outputs[100:100+plot_length, i], linestyle='-', linewidth=5, color='b', label='Reconstructed')
            
            ax.plot(inputs_copy[100:100+plot_length, i], marker='o', markersize=10, color='k', linestyle='', label='Observed')
        
    plt.savefig(save_dir + '/figures/testing_{}_seq_{}_mask_ratio_{}_{}_points.png'.format(system_name,sequence_length, round(plot_mask_ratio, 2), iter_ ))
    plt.show()


for i in range(5):
    plot_system_examples(iter_=i, plot_length=500)
















































