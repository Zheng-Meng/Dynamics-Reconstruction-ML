# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 16:37:43 2023

@author: zmzhai
"""

import os
import copy
import torch
import pickle
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def mask_data_transformer(x, mask_ratio, mask_as=0):
    # x is expected to be a 2D array with shape [sequence_length, features]
    sequence_length, features = x.shape
    mask_array = np.ones((sequence_length, features), dtype=bool)

    total_elements = sequence_length * features
    mask_number = int(total_elements * mask_ratio)

    flat_indices = np.random.choice(total_elements, mask_number, replace=False)
    masked_indices = np.unravel_index(flat_indices, (sequence_length, features))

    # Apply the mask to the data array and update the mask array
    x[masked_indices] = mask_as
    mask_array[masked_indices] = False

    return x, mask_array.astype(int)


def read_and_normalize_chaos(file_path, data_read_length):
    with open(file_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)

    # read a random segement of the data
    random_start = random.randint(10000, len(data) - data_read_length - 10000)
    data = data[random_start:random_start + data_read_length, :]
    
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    filename = os.path.basename(file_path)
    return normalized_data, filename


def read_and_preprocess_directory_chaos(directory_path, data_read_length):
    all_data = []
    filenames = []
    
    file_names = sorted(os.listdir(directory_path))

    for filename in file_names:
        if filename.endswith('.pkl'):
            file_path = os.path.join(directory_path, filename)
            normalized_data, file_name = read_and_normalize_chaos(file_path, data_read_length)
            all_data.append(normalized_data)
            filenames.append(file_name)

    return all_data, filenames


def rmse_calculation(A, B):
    # calculate root-mean-square-error (RMSE)
    return (np.sqrt(np.square(np.subtract(A, B)).mean()))

def dv_calculation(real, prediction, dv_dt=0.05):
    # calculate the deviation value
    real_cell = count_grid(real, dv_dt=dv_dt)
    pred_cell = count_grid(prediction, dv_dt=dv_dt)
    
    return np.sum( np.sqrt(np.square(real_cell - pred_cell)) )

def count_grid(data, dv_dt=0.05):
    # data = np.clip(data, 0., 1.)
    bins = np.arange(0., 1.01, dv_dt)
    
    cell = np.zeros((len(bins), len(bins)), dtype=float)
    data_copy = copy.deepcopy(data)
    
    for i in range(np.shape(data_copy)[0]):
        data_x, data_y = data_copy[i, 0], data_copy[i, 1]
        
        if data_x < 0 or data_y < 0:
            data_copy[i, :] = 0
        
        if data_x > 1 or data_y > 1:
            data_copy[i, :] = 1
            
        if np.isnan(data_x) or np.isnan(data_y):
            data_copy[i, :] = 1
            
    for i in range(np.shape(data_copy)[0]):
        data_x, data_y = data_copy[i, 0], data_copy[i, 1]
        
        x_idx = int(np.floor(data_x / dv_dt))
        y_idx = int(np.floor(data_y / dv_dt))
        
        cell[x_idx, y_idx] += 1

    cell /= float(np.shape(data)[0])
    
    return cell


def dv_calculation_3d(real, prediction, dv_dt=0.05):
    # calculate the deviation value for 3 dimensional
    real_cell = count_grid_3d(real, dt=dv_dt)
    pred_cell = count_grid_3d(prediction, dt=dv_dt)
    
    return np.sum( np.sqrt(np.square(real_cell - pred_cell)) )

def count_grid_3d(data, dt=0.02):
    # data = np.clip(data, 0., 1.)
    bins = np.arange(0., 1.01, dt)
    
    cell = np.zeros((len(bins), len(bins), len(bins)), dtype=float)
    data_copy = copy.deepcopy(data)
    
    for i in range(np.shape(data_copy)[0]):
        data_x, data_y, data_z = data_copy[i, 0], data_copy[i, 1], data_copy[i, 2]
        
        if data_x < 0 or data_y < 0 or data_z < 0:
            data_copy[i, :] = 0
        
        if data_x > 1 or data_y > 1 or data_z > 1:
            data_copy[i, :] = 1
            
        if np.isnan(data_x) or np.isnan(data_y) or np.isnan(data_z):
            data_copy[i, :] = 1
            
    for i in range(np.shape(data_copy)[0]):
        data_x, data_y, data_z = data_copy[i, 0], data_copy[i, 1], data_copy[i, 2]
        
        x_idx = int(np.floor(data_x / dt))
        y_idx = int(np.floor(data_y / dt))
        z_idx = int(np.floor(data_z / dt))
        
        cell[x_idx, y_idx, z_idx] += 1

    cell /= float(np.shape(cell)[0] ** 3) * len(data) * 1e-3
    
    return cell


def reshape_data(data, sequence_length, dim):
    # Assuming data is a 3D numpy array where it should be n * sequence_length * dim
    # Reshape the data to have the shape of [n, sequence_length, dim]
    n = data.shape[0] // sequence_length
    data = data[:n * sequence_length, :dim]
    data = data.reshape(n, sequence_length, dim)
    
    return data


def generate_stochastic_signals(data_length, sigma=12):
    # generate 3 dimensional stochastic signal
    scaler = MinMaxScaler()
    data_noise_1 = np.random.uniform(0, 1, size=(data_length * 2, 1))
    data_noise_2 = np.random.uniform(0, 1, size=(data_length * 2, 1))
    data_noise_3 = np.random.uniform(0, 1, size=(data_length * 2, 1))
    data_noise_1 = gaussian_filter(data_noise_1, sigma=sigma)
    data_noise_2 = gaussian_filter(data_noise_2, sigma=sigma)
    data_noise_3 = gaussian_filter(data_noise_3, sigma=sigma)

    data_noise = np.concatenate((data_noise_1, data_noise_2, data_noise_3), axis=1)
    # Clip outliers
    lower_percentile = np.percentile(data_noise, 5)
    upper_percentile = np.percentile(data_noise, 95)
    data_noise = np.clip(data_noise, lower_percentile, upper_percentile)

    # Scale to [0, 1] with adjusted range
    scaler = MinMaxScaler(feature_range=(-0.1, 1.1))
    data_noise = scaler.fit_transform(data_noise)

    # Apply non-linear transformation
    data_noise = np.tanh(2 * (data_noise - 0.5)) * 0.5 + 0.5

    scaler = MinMaxScaler()
    data_noise = scaler.fit_transform(data_noise)

    # Downsample
    data_noise = data_noise[::2, :]
    
    return data_noise






























