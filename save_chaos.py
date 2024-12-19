# -*- coding: utf-8 -*-
"""
Created on Tue May 16 09:26:17 2023

@author: zmzhai
"""

import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from chaotic_systems import *
import matplotlib.pyplot as plt

def plot_and_save(system, t_all, ts, params, plot_length=1000):
    # ----- to plot
    # plot_length = 500
    fig, ax = plt.subplots(3, 1, figsize=(8, 13))
    ax[0].plot(range(plot_length), ts[500:500+plot_length , 0])
    ax[1].plot(range(plot_length), ts[500:500+plot_length , 1])
    ax[2].plot(range(plot_length), ts[500:500+plot_length , 2])
    
    ax[2].set_xlabel('t')
    ax[0].set_ylabel('x')
    ax[1].set_ylabel('y')
    ax[2].set_ylabel('z')
    
    # # plt.savefig('./demonstration/{}_time_series.png'.format(system))
    plt.show()
    
    # ax = plt.figure().add_subplot(projection='3d')
    # ax.plot(ts[:, 0], ts[:, 1], ts[:, 2], color='orange')
    
    # # ax.view_init(30, 45)
    # # plt.savefig('./demonstration/{}_3d.png'.format(system))
    # plt.show()
    
    pkl_file = open('./chaos_data/' + 'data_{}'.format(system) + '.pkl', 'wb')
    pickle.dump(ts, pkl_file)
    pickle.dump(params, pkl_file)
    pkl_file.close()
    
#################### generate lorenz
def generate_lorenz(plot_length=1000, data_length=5000, scale=2):
    dt = 0.01
    t_end = data_length * scale
    t_all = np.arange(0, t_end, dt)
    x0 = [28 * np.random.rand()-14, 30 * np.random.rand()-15, 20 * np.random.rand()]
    
    system = 'lorenz'
    params = np.array([10, 28, 8 / 3])
    ts = rk4(func_lorenz, x0, t_all, params=params)
    
    ts = ts[::scale, :]
    t_all = t_all[::scale]
    
    plot_and_save(system, t_all, ts, params, plot_length=plot_length)

#################### generate rossler
def generate_rossler(scale=10, plot_length=1000, data_length=5000):
    dt = 0.01
    t_end = data_length * scale
    t_all = np.arange(0, t_end, dt)
    x0 = [28 * np.random.rand()-14, 30 * np.random.rand()-15, 20 * np.random.rand()]
    
    system = 'rossler'
    params = np.array([0.2, 0.2, 5.7])
    ts = rk4(func_rossler, x0, t_all, params=params)
    
    ts = ts[::scale, :]
    t_all = t_all[::scale]
    
    plot_and_save(system, t_all, ts, params, plot_length=plot_length)

#################### generate foodchain
def generate_foodchain(scale=5, plot_length=1000, data_length=5000):
    dt = 0.1
    t_end = data_length * scale * 10
    t_all = np.arange(0, t_end, dt)
    x0 = [0.4 * np.random.rand() + 0.6, 0.4 * np.random.rand() + 0.15, 0.5 * np.random.rand() + 0.3]

    system = 'foodchain'
    # params = np.array([0.94, 1.7, 5.0])
    params = np.array([0.98, 2.009, 2.876])
    ts = rk4(func_foodchain, x0, t_all, params=params)
    
    ts = ts[::scale, :]
    t_all = t_all[::scale]
    
    plot_and_save(system, t_all, ts, params, plot_length=plot_length)
    
#################### generate hastings
def generate_hastings(scale=6, plot_length=1000, data_length=5000):
    # we use a1 as the bifurcation parameter
    dt = 0.1
    t_end = data_length * scale * 10
    t_all = np.arange(0, t_end, dt)
    x0 = [(0.8 - 0.5) * np.random.rand() + 0.5, 0.3 * np.random.rand(), (10 - 8) * np.random.rand() + 8]

    system = 'hastings'
    params = np.array([5, 0.1, 3, 2, 0.4, 0.01])
    ts = rk4(func_hastings, x0, t_all, params=params)
    
    ts = ts[::scale, :]
    t_all = t_all[::scale]
    
    plot_and_save(system, t_all, ts, params, plot_length=plot_length)
    
#################### generate lotka volterra
def generate_lotka_volterra(scale=2, plot_length=1000, data_length=5000):
    dt = 0.1
    t_end = data_length * scale * 10
    t_all = np.arange(0, t_end, dt)
    x0 = [(0.5 - 0.2) * np.random.rand() + 0.2, (0.5 - 0.2) * np.random.rand() + 0.2, 
          (0.5 - 0.2) * np.random.rand() + 0.2, (0.5 - 0.2) * np.random.rand() + 0.2]

    system = 'lotka'
    params = np.array([0])
    ts = rk4(func_lotka_volterra, x0, t_all, params=params)
    
    ts = ts[::scale, :]
    t_all = t_all[::scale]
    
    plot_and_save(system, t_all, ts, params, plot_length=plot_length)

#################### generate rikitake
def generate_rikitake(scale=5, plot_length=1000, data_length=5000):
    dt = 0.01
    t_end = data_length * scale
    t_all = np.arange(0, t_end, dt)
    x0 = [0.1 * np.random.rand(), 0.1 * np.random.rand(), 0.1 * np.random.rand()]

    system = 'rikitake'
    params = np.array([2.0, 5.0])
    ts = rk4(func_rikitake, x0, t_all, params=params)
    
    ts = ts[::scale, :]
    t_all = t_all[::scale]
    
    plot_and_save(system, t_all, ts, params, plot_length=plot_length)

#################### generate wang
def generate_wang(scale=2, plot_length=1000, data_length=5000):
    dt = 0.01
    t_end = data_length * scale
    t_all = np.arange(0, t_end, dt)
    x0 = [0.1 * np.random.rand(), 0.1 * np.random.rand(), 0.1 * np.random.rand()]
    
    system = 'wang'
    params = 'None'
    ts = rk4(func_wang, x0, t_all)
    
    ts = ts[::scale, :]
    t_all = t_all[::scale]

    plot_and_save(system, t_all, ts, params, plot_length=plot_length)
    
#################### generate sprott
def generate_sprott(index=0, scale=2, plot_length=1000, data_length=5000):
    dt = 0.01
    t_end = data_length * scale
    t_all = np.arange(0, t_end, dt)
    x0 = [0.1 * np.random.rand(), 0.1 * np.random.rand(), 0.1 * np.random.rand()]
    
    system = 'sprott_{}'.format(index)
    params = np.array([index])
    ts = rk4(func_sprott, x0, t_all, params=params)
    
    ts = ts[::scale, :]
    t_all = t_all[::scale]
    
    plot_and_save(system, t_all, ts, params, plot_length=plot_length)
    

#################### generate chua
def generate_chua(scale=1, plot_length=1000, data_length=5000):
    dt = 0.01
    t_end = data_length * scale
    t_all = np.arange(0, t_end, dt)
    x0 = [1*np.random.rand(), 1*np.random.rand(), 1*np.random.rand()]

    system = 'chua'
    params = np.array([15.6, 1, 28])
    ts = rk4(func_chua, x0, t_all, params=params)
    
    ts = ts[::scale, :]
    t_all = t_all[::scale]
    
    plot_and_save(system, t_all, ts, params, plot_length=plot_length)
    

#################### generate dadras
def generate_dadras(scale=1, plot_length=1000, data_length=5000):
    dt = 0.01
    t_end = data_length * scale
    t_all = np.arange(0, t_end, dt)
    x0 = [0.5*np.random.rand(), 0.5*np.random.rand(), 0.5*np.random.rand()]

    system = 'dadras'
    params = np.array([3.0, 2.7, 1.7, 2.0, 9.0])
    ts = rk4(func_dadras, x0, t_all, params=params)
    
    ts = ts[::scale, :]
    t_all = t_all[::scale]
    
    plot_and_save(system, t_all, ts, params, plot_length=plot_length)
    
    
#################### generate four_wing
def generate_four_wing(scale=1, plot_length=1000, data_length=5000):
    dt = 0.01
    t_end = data_length * scale
    t_all = np.arange(0, t_end, dt)
    x0 = [0.5*np.random.rand(), 0.5*np.random.rand(), 0.5*np.random.rand()]

    system = 'four_wing'
    params = np.array([0.2, 0.01, -0.4])
    ts = rk4(func_four_wing, x0, t_all, params=params)
    
    ts = ts[::scale, :]
    t_all = t_all[::scale]
    
    plot_and_save(system, t_all, ts, params, plot_length=plot_length)
    
#################### generate aizawa
def generate_aizawa(scale=1, plot_length=1000, data_length=5000):
    dt = 0.01
    t_end = data_length * scale
    t_all = np.arange(0, t_end, dt)
    x0 = [0.1*np.random.rand(), 0.1*np.random.rand(), 0.1*np.random.rand()]

    system = 'aizawa'
    params = np.array([0.95, 0.7, 0.6, 3.5, 0.25, 0.1])
    ts = rk4(func_aizawa, x0, t_all, params=params)
    
    ts = ts[::scale, :]
    t_all = t_all[::scale]
    
    plot_and_save(system, t_all, ts, params, plot_length=plot_length)
    
#################### generate bouali
def generate_bouali(scale=2, plot_length=1000, data_length=5000):
    dt = 0.01
    t_end = data_length * scale
    t_all = np.arange(0, t_end, dt)
    x0 = [0.1*np.random.rand(), 0.1*np.random.rand(), 0.1*np.random.rand()]

    system = 'bouali'
    params = np.array([0.3, 0.05, 4, 1, 1.5, 1])
    ts = rk4(func_bouali2, x0, t_all, params=params)
        
    ts = ts[::scale, :]
    t_all = t_all[::scale]
    
    plot_and_save(system, t_all, ts, params, plot_length=plot_length)
    

if __name__ == '__main__':
    print('save_chaos')
    
    data_length = 6000
    data_length = 30000
    
    # generate_lorenz(data_length=data_length)
    generate_rossler(data_length=data_length, scale=16)
    
    # generate_hastings(scale=10, data_length=data_length) # default 30
    # generate_lotka_volterra(scale=10, data_length=data_length) # default 50
    # generate_rikitake(scale=10, data_length=data_length)
    # generate_wang(scale=5, data_length=data_length)
    # generate_sprott(index=0, scale=20, data_length=data_length)
    # generate_sprott(index=1, scale=15, data_length=data_length)
    # generate_sprott(index=2, scale=15, data_length=data_length)
    # generate_sprott(index=3, scale=13, data_length=data_length)
    # generate_sprott(index=4, scale=17, data_length=data_length)
    # generate_sprott(index=5, scale=13, data_length=data_length)
    # generate_sprott(index=6, scale=13, data_length=data_length)
    # generate_sprott(index=7, scale=18, data_length=data_length)
    # generate_sprott(index=8, scale=30, data_length=data_length)
    # generate_sprott(index=9, scale=16, data_length=data_length)
    # generate_sprott(index=10, scale=18, data_length=data_length)
    # generate_sprott(index=11, scale=10, data_length=data_length)
    # generate_sprott(index=12, scale=15, data_length=data_length)
    # generate_sprott(index=13, scale=15, data_length=data_length)
    # generate_sprott(index=14, scale=15, data_length=data_length)
    # generate_sprott(index=15, scale=15, data_length=data_length)
    # generate_sprott(index=16, scale=8, data_length=data_length)
    # generate_sprott(index=17, scale=17, data_length=data_length)
    # generate_sprott(index=18, scale=12, data_length=data_length)
    # generate_dadras(scale=2, data_length=data_length)
    # generate_four_wing(scale=24, data_length=data_length)
    
    # generate_aizawa(scale=6, data_length=data_length)
    # generate_bouali(scale=6, data_length=data_length)
    # generate_foodchain(data_length=data_length, scale=10)
    # generate_chua(scale=4, data_length=data_length)










































