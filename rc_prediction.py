# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 16:18:15 2024

@author: yclai
"""

import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
import utils
import networkx as nx

system = 'foodchain'
# system = 'lorenz'
# system = 'lotka' # mask ratio 0.75
mask_ratio = 0.8
sequence_length = 2000

save_dir = './save_data/transformer_iter_trained'
filepath = os.path.join(save_dir, 'system_{}_seq_{}_mask_{}.pkl'.format(system,sequence_length, mask_ratio))

with open(filepath, 'rb') as f:
    data = pickle.load(f)

inputs, targets, outputs = data['inputs'], data['targets'], data['outputs']

inputs_array = inputs.reshape(-1, 3)
targets_array = targets.reshape(-1, 3)
outputs_array = outputs.reshape(-1, 3)

# dimension of the system
input_dim = 3
output_dim = 3

# read save file for optimal hyperparameters
pkl_file = open('./save_model/rc_opt_3d_{}.pkl'.format(system), 'rb')

opt_results = pickle.load(pkl_file)
pkl_file.close()
opt_params = opt_results['params']

n = 500
eig_rho = opt_params['eig_rho']
gamma = opt_params['gamma']
alpha = opt_params['alpha']
beta = 10 ** opt_params['beta']
d = opt_params['d']
noise_a = 10 ** opt_params['noise_a']

train_length = 50000
test_length = 10000
short_prediction_length = 200

total_length = train_length + 2 * test_length + short_prediction_length

random_start = np.random.randint(1, 100001)
ts_train = outputs_array[random_start:, :]
ts_train_real = targets_array[random_start:, :]

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

# train_preditions = np.dot(Wout, r_out)

# fig, ax = plt.subplots(1, 1, constrained_layout=True)
# ax.plot(train_preditions[0, 20000:25000])
# ax.plot(train_y[0, 20000:25000])

# test
testing_start = train_length + 1

test_pred = np.zeros((test_length, output_dim))
test_real = np.zeros((test_length, output_dim))

# test_real[:, :] = ts_train[testing_start:testing_start+np.shape(test_real)[0], :]
test_real[:, :] = ts_train_real[testing_start:testing_start+np.shape(test_real)[0], :]

r = r_end

u = np.zeros((input_dim, 1))
u[:] = ts_train_real[train_length, :input_dim].reshape(-1, 1)
for ti in range(test_length-1):
    r = (1 - alpha) * r + alpha * np.tanh(np.dot(A, r) + np.dot(Win, u))
    
    pred = np.dot(Wout, r)
    test_pred[ti, :] = pred.reshape(output_dim, -1).ravel()
    
    u[:] = pred[:input_dim]

rmse = utils.rmse_calculation(test_pred[:short_prediction_length,:], test_real[:short_prediction_length,:])

print('rmse:', rmse )
# plot short term prediction
fig, ax = plt.subplots(3, 1, constrained_layout=True)

ax[0].plot(test_real[:short_prediction_length,0], label='real')
ax[0].plot(test_pred[:short_prediction_length,0], label='pred')
ax[0].set_ylabel('x')
ax[0].legend()

ax[1].plot(test_real[:short_prediction_length,1], label='real')
ax[1].plot(test_pred[:short_prediction_length,1], label='pred')
ax[1].set_ylabel('y')

ax[2].plot(test_real[:short_prediction_length,2], label='real')
ax[2].plot(test_pred[:short_prediction_length,2], label='pred')
ax[2].set_ylabel('z')
ax[2].set_xlabel('step')
plt.savefig('figures/rc_prediction_short.png')
plt.show()

# calculate dv 3d
real_points = test_real
pred_points = test_pred

dv = utils.dv_calculation_3d(real_points, pred_points)
print('dv:', dv)

# fig = plt.figure(figsize=(10, 8), constrained_layout=True)
# ax = fig.add_subplot(111, projection='3d')

# ax.plot(targets_array[:, 0], targets_array[:, 1], targets_array[:, 2], label='real')
# ax.plot(outputs_array[:, 0], outputs_array[:, 1], outputs_array[:, 2], label='pred')

# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.legend()
# plt.show()

fig = plt.figure(figsize=(10, 8), constrained_layout=True)
ax = fig.add_subplot(111, projection='3d')

ax.plot(real_points[:, 0], real_points[:, 1], real_points[:, 2], label='real')
ax.plot(pred_points[:-1, 0], pred_points[:-1, 1], pred_points[:-1, 2], label='pred')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
plt.savefig('figures/rc_prediction_long.png')
plt.show()

# pkl_file = open('./save_data/' + 'rc_prediction/system_{}_mask_{}'.format(system, mask_ratio)+ '.pkl', 'wb')
# pickle.dump(test_real, pkl_file)
# pickle.dump(test_pred, pkl_file)
# pickle.dump(inputs_array, pkl_file)
# pickle.dump(targets_array, pkl_file)
# pickle.dump(outputs_array, pkl_file)
# pickle.dump(A, pkl_file)
# pickle.dump(Win, pkl_file)
# pickle.dump(Wout, pkl_file)
# pkl_file.close()












