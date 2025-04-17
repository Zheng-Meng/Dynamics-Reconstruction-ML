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

import scipy.interpolate

torch.cuda.empty_cache()
gc.collect()

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# read and test length for each target system: 400000
data_read_length = 400000
directory_path = './chaos_data/'
save_file_name = 'trained' 
save_dir = './save_data/transformer_iter_interpolation_{}'.format(save_file_name)
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


test_num = len(attractors_test)
def save_data(inputs, targets, outputs, outputs_linear, outputs_spline, outputs_trig, sequence_length, mask_ratio, system_name):
    filename = f"system_{system_name}_seq_{sequence_length}_mask_{round(mask_ratio,2)}.pkl"
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump({'inputs': inputs, 'targets': targets, 'outputs': outputs, 'outputs_linear': outputs_linear, 'outputs_spline': outputs_spline, 'outputs_trig': outputs_trig}, f)

# create model
model = transformer_encoder.TimeSeriesTransformer(args.input_size, args.output_size, args.d_model, args.nhead, args.num_layers, args.hidden_size, args.dropout).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

model.load_state_dict(torch.load('./save_model/save_chaos_transformer_{}.pth'.format(save_file_name)))

# grid scan
# seq_length_set = range(50, 3010, 25)
# mask_ratio_set = np.arange(0.0, 1.01, 0.01)

seq_length_set = [2000]
mask_ratio_set = [0.86]

loss_ml = np.zeros((len(seq_length_set), len(mask_ratio_set), test_num))
loss_ml_std = np.zeros((len(seq_length_set), len(mask_ratio_set), test_num))

loss_linear = np.zeros((len(seq_length_set), len(mask_ratio_set), test_num))
loss_spline = np.zeros((len(seq_length_set), len(mask_ratio_set), test_num))
loss_trig = np.zeros((len(seq_length_set), len(mask_ratio_set), test_num))

loss_linear_std = np.zeros((len(seq_length_set), len(mask_ratio_set), test_num))
loss_spline_std = np.zeros((len(seq_length_set), len(mask_ratio_set), test_num))
loss_trig_std = np.zeros((len(seq_length_set), len(mask_ratio_set), test_num))

def interpolate_sequence(masked_seq, method='linear'):
    """
    Interpolates a masked sequence using the specified method.
    `masked_seq`: shape (T, dim)
    Returns: interpolated sequence, same shape
    """
    # remove the first dimension of masked_seq
    masked_seq = masked_seq.squeeze(0)
    T, D = masked_seq.shape
    interpolated = np.zeros_like(masked_seq)
    
    for d in range(D):
        y = masked_seq[:, d]
        x = np.arange(T)
        # Identify masked points (zeros) instead of NaN values
        mask = y != 0  # True for non-zero values (observed points)
        x_obs = x[mask]
        y_obs = y[mask]

        if len(x_obs) < 2:  # not enough points to interpolate
            interpolated[:, d] = 0.0
            continue

        try: # different interpolation methods
            if method == 'linear':
                f = scipy.interpolate.interp1d(x_obs, y_obs, kind='linear', fill_value="extrapolate")
            elif method == 'spline':
                f = scipy.interpolate.UnivariateSpline(x_obs, y_obs, s=0)
            elif method == 'trig': # it performs very bad in our task
                f = scipy.interpolate.interp1d(x_obs, y_obs, kind='cubic', fill_value="extrapolate")  # using cubic as trig approx
            else:
                raise ValueError("Unsupported interpolation method.")
            interpolated[:, d] = f(x)
        except Exception as e:
            print(f"Interpolation error for dimension {d}: {e}")
            interpolated[:, d] = 0.0  # fallback
    return interpolated



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
                accumulated_outputs_linear, accumulated_outputs_spline, accumulated_outputs_trig = [], [], []
                
                loss_file_ml, loss_file_ml_std = [], []
                loss_file_linear, loss_file_linear_std = [], []
                loss_file_spline, loss_file_spline_std = [], []
                loss_file_trig, loss_file_trig_std = [], []

                for data in test_loader:
                    inputs = data[0].to(device)
                    targets = copy.deepcopy(inputs)
        
                    inputs_new = torch.zeros((inputs.shape[0], inputs.shape[1], args.input_size)).to(device)
                    for i in range(inputs.shape[0]):
                        numpy_input, temp_mask = utils.mask_data_transformer(inputs[i].cpu().numpy(), mask_ratio)
                        
                        inputs_new[i, :, :args.dim] = torch.from_numpy(numpy_input).to(device)

                    outputs = model(inputs_new)
                    loss = criterion(outputs, targets)
                    loss_file_ml.append(loss.cpu().item())

                    interp_linear_indata = interpolate_sequence(inputs_new.cpu().numpy(), method='linear')
                    interp_spline_indata = interpolate_sequence(inputs_new.cpu().numpy(), method='spline')
                    interp_trig_indata = interpolate_sequence(inputs_new.cpu().numpy(), method='trig')

                    loss_linear_indata = criterion(outputs, torch.from_numpy(interp_linear_indata).to(device))
                    loss_spline_indata = criterion(outputs, torch.from_numpy(interp_spline_indata).to(device))
                    loss_trig_indata = criterion(outputs, torch.from_numpy(interp_trig_indata).to(device))

                    loss_file_linear.append(loss_linear_indata.cpu().item())
                    loss_file_spline.append(loss_spline_indata.cpu().item())
                    loss_file_trig.append(loss_trig_indata.cpu().item())

                    # Accumulate the batch data
                    accumulated_inputs.append(inputs_new.cpu().numpy())
                    accumulated_targets.append(targets.cpu().numpy())
                    accumulated_outputs.append(outputs.cpu().numpy())

                    accumulated_outputs_linear.append(interp_linear_indata)
                    accumulated_outputs_spline.append(interp_spline_indata)
                    accumulated_outputs_trig.append(interp_trig_indata)
                
                combined_inputs = np.concatenate(accumulated_inputs, axis=0)
                combined_targets = np.concatenate(accumulated_targets, axis=0)
                combined_outputs = np.concatenate(accumulated_outputs, axis=0)
                
                combined_outputs_linear = np.stack(accumulated_outputs_linear, axis=0)
                combined_outputs_spline = np.stack(accumulated_outputs_spline, axis=0)
                combined_outputs_trig = np.stack(accumulated_outputs_trig, axis=0)

                if save_data_to_file:
                    save_data(combined_inputs, combined_targets, combined_outputs, combined_outputs_linear, combined_outputs_spline, combined_outputs_trig, sequence_length, mask_ratio, attractors_test[system_i])
                    
                loss_ml[seq_length_idx, mask_ratio_idx, system_i] = np.mean(loss_file_ml)
                loss_ml_std[seq_length_idx, mask_ratio_idx, system_i] = np.std(loss_file_ml)

                loss_linear[seq_length_idx, mask_ratio_idx, system_i] = np.mean(loss_file_linear)
                loss_linear_std[seq_length_idx, mask_ratio_idx, system_i] = np.std(loss_file_linear)

                loss_spline[seq_length_idx, mask_ratio_idx, system_i] = np.mean(loss_file_spline)
                loss_spline_std[seq_length_idx, mask_ratio_idx, system_i] = np.std(loss_file_spline)

                loss_trig[seq_length_idx, mask_ratio_idx, system_i] = np.mean(loss_file_trig)
                loss_trig_std[seq_length_idx, mask_ratio_idx, system_i] = np.std(loss_file_trig)
                
                

                print(f'calculate loss of {attractors_test[system_i]} for seq length {sequence_length} and mask ratio {mask_ratio}')

# save statistics if needed
# pkl_file = open('./save_data/' + 'save_loss_seq_length_mask_ratio_interpolation'+ '.pkl', 'wb')
# pickle.dump(loss_ml, pkl_file)
# pickle.dump(loss_ml_std, pkl_file)
# pickle.dump(loss_linear, pkl_file)
# pickle.dump(loss_linear_std, pkl_file)
# pickle.dump(loss_spline, pkl_file)
# pickle.dump(loss_spline_std, pkl_file)
# pickle.dump(loss_trig, pkl_file)
# pickle.dump(loss_trig_std, pkl_file)
# pickle.dump(seq_length_set, pkl_file)
# pickle.dump(mask_ratio_set, pkl_file)
# pkl_file.close()

##### read saved file and plot
system_name = attractors_test[0] # 'foodchain'
sequence_length = 2000
system_name = 'foodchain'
plot_seq_length = sequence_length
plot_mask_ratio = 0.8

filename = f"system_{system_name}_seq_{sequence_length}_mask_{round(plot_mask_ratio,2)}.pkl"
filepath = os.path.join(save_dir, filename)

with open(filepath, 'rb') as f:
    data = pickle.load(f)

os.makedirs(save_dir + '/figures', exist_ok=True)

def plot_system_examples(iter_=0, plot_length=500):
    inputs, targets, outputs, outputs_linear, outputs_spline, outputs_trig = data['inputs'], data['targets'], data['outputs'], data['outputs_linear'], data['outputs_spline'], data['outputs_trig']
    inputs, targets, outputs = inputs[iter_, :, :], targets[iter_, :, :], outputs[iter_,:,:]

    outputs_linear, outputs_spline, outputs_trig = outputs_linear[iter_, :, :], outputs_spline[iter_, :, :], outputs_trig[iter_,:,:]

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
            ax.plot(outputs[100:100+plot_length, i], linestyle='-', linewidth=5, color='b', label='Reconstructed by transformer')
            ax.plot(outputs_linear[100:100+plot_length, i], linestyle='-', linewidth=2, color='r', label='Reconstructed by linear interpolation')
            ax.plot(outputs_spline[100:100+plot_length, i], linestyle='-', linewidth=2, color='g', label='Reconstructed by spline interpolation')
            ax.plot(outputs_trig[100:100+plot_length, i], linestyle='-', linewidth=2, color='y', label='Reconstructed by cubic interpolation')
            
            ax.plot(inputs_copy[100:100+plot_length, i], marker='o', markersize=10, color='k', linestyle='', label='Observed')
        
    # plt.savefig(save_dir + '/figures/testing_{}_seq_{}_mask_ratio_{}_{}_points.png'.format(system_name,sequence_length, round(plot_mask_ratio, 2), iter_ ))
    plt.show()


for i in range(3):
    plot_system_examples(iter_=i, plot_length=500)
















































