# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 16:47:47 2025

plot the bifurcation diagram of food chain system

@author: zmzhai
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import argrelextrema
import pickle

def func_foodchain(x, t, params):

    k = params[0]
    yc = params[1]
    yp = params[2]
        
    xc = 0.4
    xp = 0.08
    r0 = 0.16129
    c0 = 0.5
    
    dxdt = []
    dxdt.append( x[0] * (1 - x[0] / k) - xc * yc * x[1] * x[0] / (x[0] + r0) )
    dxdt.append(xc * x[1] * (yc * x[0] / (x[0] + r0) - 1) - xp * yp * x[2] * x[1] / (x[1] + c0))
    dxdt.append(xp * x[2] * (yp * x[1] / (x[1] + c0) - 1))
    
    return np.array(dxdt)

def rk4(f, x0, t, params=np.array([]), early_stop_check=True, z_threshold=0.5, check_every=2000):
    n = len(t)
    x = np.zeros((n, len(x0)))
    x[0] = x0
    h = t[1] - t[0]
    check_cutoff = 25000  # only check early stop before this step

    for i in range(n - 1):
        if len(params.shape) > 1:
            params_step = params[i, :]
        else:
            params_step = params

        k1 = f(x[i], t[i], params_step)
        k2 = f(x[i] + k1 * h / 2., t[i] + h / 2., params_step)
        k3 = f(x[i] + k2 * h / 2., t[i] + h / 2., params_step)
        k4 = f(x[i] + k3 * h, t[i] + h, params_step)
        x[i + 1] = x[i] + (h / 6.) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Check early stop condition only before cutoff
        if early_stop_check and i % check_every == 0 and i < check_cutoff:
            z_sample = x[max(0, i - check_every):i+1, 2]
            if np.max(z_sample) < z_threshold:
                return None  # signal early death of z

    return x



#################### generate foodchain
def generate_foodchain(k=0.98, data_length=5000, transient=2000, plot=False, save=False):
    print('k:', k)
    dt = 0.1
    t_end = data_length * 10
    t_all = np.arange(0, t_end, dt)
    # x0 = [0.4 * np.random.rand() + 0.6, 0.4 * np.random.rand() + 0.15, 0.5 * np.random.rand() + 0.3]

    system = 'foodchain'
    # params = np.array([0.94, 1.7, 5.0])
    params = np.array([k, 2.009, 2.876])
    # ts = rk4(func_foodchain, x0, t_all, params=params)
    z_dim_valid = False
    attempts = 0
    
    while not z_dim_valid and attempts < 500:
        print(attempts)
        x0 = [0.4 * np.random.rand() + 0.6, 0.4 * np.random.rand() + 0.15, 0.5 * np.random.rand() + 0.3]
        ts = rk4(func_foodchain, x0, t_all, params=params)
        if ts is None:
            attempts += 1
            continue
        
        if np.max(ts[:, 2]) > 0.5:
            z_dim_valid = True
        else:
            attempts += 1

        
    ts = ts[::10, :]
    ts = ts[transient:, :]
    # t_all = t_all[::scale]
    
    plot_length = 500
    if plot:
        fig, ax = plt.subplots(3, 1, figsize=(8, 13))
        ax[0].plot(range(plot_length), ts[:plot_length , 0])
        ax[1].plot(range(plot_length), ts[:plot_length , 1])
        ax[2].plot(range(plot_length), ts[:plot_length , 2])
        
        ax[2].set_xlabel('t')
        ax[0].set_ylabel('x')
        ax[1].set_ylabel('y')
        ax[2].set_ylabel('z')
        
        plt.show()

        # plot in 3d attractor
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(ts[:, 0], ts[:, 1], ts[:, 2])
        plt.show()
    
    if save:
        pkl_file = open('./data_foodchain/' + 'data_{}_k_{}'.format(system, k) + '.pkl', 'wb')
        pickle.dump(ts, pkl_file)
        pickle.dump(params, pkl_file)
        pkl_file.close()
    

def get_local_min_max(z):
    """Return local minima and maxima from a 1D array."""
    local_max = argrelextrema(z, np.greater)[0]
    local_min = argrelextrema(z, np.less)[0]
    return z[local_min], z[local_max]


def bifurcation_foodchain(k_range=np.linspace(0.957, 0.985, 500), 
                          t_max=10000, dt=0.1, transient=1000):
    yp = 2.876
    yc = 2.009
    results = []
    valid_k = []

    t = np.arange(0, t_max, dt)
    for k in k_range:
        print('k: ', k)
        x0 = [0.4 * np.random.rand() + 0.6, 
              0.4 * np.random.rand() + 0.15, 
              0.5 * np.random.rand() + 0.3]
        z_dim_valid = False
        attempts = 0

        # Try multiple times if z dies out
        while not z_dim_valid and attempts < 100:
            ts = rk4(func_foodchain, x0, t, params=np.array([k, yc, yp]))
            if ts is None:
                attempts += 1
                x0 = [0.4 * np.random.rand() + 0.6, 
                      0.4 * np.random.rand() + 0.15, 
                      0.5 * np.random.rand() + 0.3]
                continue
            
            z = ts[::10, 2]
            
            z_post_transient = z[int(transient):]
            if np.max(z_post_transient) > 0.5:
                z_dim_valid = True
            else:
                attempts += 1

        if not z_dim_valid:
            continue  # skip this k value

        z_post_transient = z[int(transient):]
        z_min, z_max = get_local_min_max(z_post_transient)
        results.append((z_min, z_max))
        valid_k.append(k)

    # Plotting the bifurcation diagram
    plt.figure(figsize=(10, 6))
    for i, k in enumerate(valid_k):
        k_array = np.full(results[i][0].shape, k)
        plt.plot(k_array, results[i][0], 'b.', markersize=1)
        k_array = np.full(results[i][1].shape, k)
        plt.plot(k_array, results[i][1], 'r.', markersize=1)

    plt.xlabel('k (bifurcation parameter)')
    plt.ylabel('z (local minima and maxima)')
    plt.title('Bifurcation Diagram of the Food Chain System')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # z_min_list = [pair[0] for pair in results]
    # z_max_list = [pair[1] for pair in results]
    
    with open('./save_data/bifurcation_foodchain.pkl', 'wb') as f:
        pickle.dump((valid_k, results), f)
        print("Saved to bifurcation_foodchain.pkl")



if __name__ == '__main__':
    print('foodchain bifurcation')
    
    bifurcation_foodchain(k_range=np.linspace(0.957, 0.985, 500), t_max=100000, dt=0.1, transient=10000)



















    