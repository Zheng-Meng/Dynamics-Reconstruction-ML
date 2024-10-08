# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 17:04:42 2024

@author: yclai
"""

import numpy as np
from bayes_opt import BayesianOptimization
import networkx as nx
import time
import warnings
import utils
import pickle
import os

warnings.filterwarnings("ignore")

start = time.time()

# dim = 3
input_dim = 3
output_dim = 3
n = 500 # 500

system = 'foodchain'
mask_ratio = 0.8
sequence_length = 2000

def target_rc(d, eig_rho, gamma, alpha, beta, noise_a, iter_time=10, proportion=0.8):
    
    save_dir = './save_data/transformer_iter_trained'
    filepath = os.path.join(save_dir, 'system_{}_seq_{}_mask_{}.pkl'.format(system,sequence_length, mask_ratio))
    
    # print('system: ', system, ', mask_ratio: ', mask_ratio)

    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    inputs, targets, outputs = data['inputs'], data['targets'], data['outputs']
    # only outputs array will be used
    outputs_array = outputs.reshape(-1, 3)
    # 50000, 10000, 10000, 150
    train_length = 50000
    test_length = 10000
    washup_length = 10000
    short_prediction_length = 150
    
    total_length = train_length + 2 * test_length + 2 * washup_length + short_prediction_length
    
    beta = 10 ** (beta)
    noise_a = 10 ** (noise_a)
    
    rmse_all = []
    for i in range(iter_time):
        
        random_start = np.random.randint(1, 100001)
        ts_train = outputs_array[random_start:, :]
        
        # reservoir computer configuration
        Win = np.random.uniform(-gamma, gamma, (n, input_dim))
        
        graph = nx.erdos_renyi_graph(n, d, 42, False)
        for (u, v) in graph.edges():
            graph.edges[u, v]['weight'] = np.random.normal(0.0, 1.0)
        A = nx.adjacency_matrix(graph).todense()
        rho = max(np.linalg.eig(A)[0])
        A = (eig_rho / abs(rho)) * A
        
        # train
        r_train = np.zeros((n, train_length))
        # y_train = np.zeros((dim, train_length))
        y_train = np.zeros((output_dim, train_length))
        r_end = np.zeros((n, 1))
        
        train_x = np.zeros((train_length, input_dim))
        train_y = np.zeros((train_length, output_dim))
        
        train_y[:, :] = ts_train[1:train_length+1, :]
        
        noise = noise_a * np.random.randn(*ts_train[:train_length, :].shape)
        # Adding the noise to the ts_train data
        ts_train[:train_length, :] += noise
        
        train_x[:, :] = ts_train[:train_length, :input_dim]
        
        train_x = np.transpose(train_x)
        train_y = np.transpose(train_y)
        
        r_all = np.zeros((n, train_length + 1))
        
        for ti in range(train_length):
            r_all[:, ti+1] = (1 - alpha) * r_all[:, ti] + \
                alpha * np.tanh( np.dot(A, r_all[:, ti]) + np.dot(Win, train_x[:, ti])  )
        
        r_out = r_all[:, 1:]
        r_end[:] = r_all[:, -1].reshape(-1, 1)
        
        r_train[:, :] = r_out
        y_train[:, :] = train_y[:, :]
        
        Wout = np.dot(np.dot(y_train, np.transpose(r_train)), np.linalg.inv(np.dot(r_train, np.transpose(r_train)) + beta * np.eye(n)) )

        # test

        testing_start = train_length + 1
        
        test_pred = np.zeros((test_length, output_dim))
        test_real = np.zeros((test_length, output_dim))

        test_real[:, :] = ts_train[testing_start:testing_start+np.shape(test_real)[0], :]

        r = r_end

        # u = np.zeros((dim, 1))
        u = np.zeros((input_dim, 1))
        u[:] = ts_train[train_length, :input_dim].reshape(-1, 1)
        for ti in range(test_length-1):
            r = (1 - alpha) * r + alpha * np.tanh(np.dot(A, r) + np.dot(Win, u))
            
            pred = np.dot(Wout, r)
            test_pred[ti, :] = pred.reshape(output_dim, -1).ravel()
            
            u[:] = pred[:input_dim]

        rmse = utils.rmse_calculation(test_pred[:short_prediction_length,:], test_real[:short_prediction_length,:])
        
        rmse_all.append(np.mean(rmse))
    
    rmse_mean = np.average(sorted(rmse_all)[:int(proportion * iter_time)])
    
    print(rmse_mean)

    return 1 / rmse_mean



system_set = ['foodchain', 'lorenz', 'lotka']
# mask_ratio_set = np.arange(0.5, 1.0, 0.05)
# mask_ratio_set = [0.8, 0.75, 0.85]

# optimize the hyperparameters for each system
for system in system_set:
    # for mask_ratio_i in range(len(mask_ratio_set)):
    #     mask_ratio = round(mask_ratio_set[mask_ratio_i], 2)
    
    optimizer = BayesianOptimization(target_rc,
                                      {'d': (0.01, 1), 'eig_rho': (0.01, 5), 'gamma': (0.01, 5), 'alpha': (0.01, 1), 'beta': (-7, -1), 'noise_a': (-7, -1)},)

    optimizer.maximize(n_iter=200)
    print(optimizer.max)

    pkl_file = open('./save_model/rc_opt_3d_{}'.format(system) + '.pkl', 'wb')
    pickle.dump(optimizer.max, pkl_file)
    pkl_file.close()


    end = time.time()


end = time.time()
print(end - start)





































