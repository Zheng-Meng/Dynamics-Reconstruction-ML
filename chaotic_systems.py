# -*- coding: utf-8 -*-
"""
Created on Mon May 15 09:23:45 2023

@author: zmzhai
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def func_lorenz(x, t, params):
    if params.size == 0:
        sigma = 10
        rho = 28
        beta = 8 / 3
    else:
        sigma = params[0]
        rho = params[1]
        beta = params[2]
    dxdt = []

    dxdt.append(sigma * (x[1] - x[0]))
    dxdt.append(x[0] * (rho - x[2]) - x[1])
    dxdt.append(x[0] * x[1] - beta * x[2])

    return np.array(dxdt)

def func_rossler(x, t, params):
    if params.size == 0:
        a = 0.2
        b = 0.2
        c = 5.7
    else:
        a = params[0]
        b = params[1]
        c = params[2]
    dxdt = []
    
    dxdt.append( - (x[1] + x[2]) )
    dxdt.append( x[0] + a * x[1] )
    dxdt.append( b + x[2] * (x[0] - c) )
    
    return np.array(dxdt)
    

def func_foodchain(x, t, params):
    if params.size == 0:
        k = 0.94
        yc = 1.7
        yp = 5.0
    else:
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

def func_hastings(x, t, params):
    if params.size == 0:
        a1 = 5
        a2 = 0.1
        b1 = 3
        b2 = 2
        d1 = 0.4
        d2 = 0.01
    else:
        a1 = params[0]
        a2 = params[1]
        b1 = params[2]
        b2 = params[3]
        d1 = params[4]
        d2 = params[5]
    
    dxdt = []
    dxdt.append( x[0] * (1 - x[0]) - a1 * x[0] / (b1 * x[0] + 1) * x[1] )
    dxdt.append( a1 * x[0] / (b1 * x[0] + 1) * x[1] - a2 * x[1] / (b2 * x[1] + 1) * x[2] - d1 * x[1] )
    dxdt.append( a2 * x[1] / (b2 * x[1] + 1) * x[2] - d2 * x[2]  )
    
    return np.array(dxdt)

def func_lotka_volterra(x, t, params):
    # sprott: Chaos in low-dimensional Lotka–Volterra models of competition
    r_i = [1, 0.72, 1.53, 1.27]
    a_ij = np.array([[1, 1.09, 1.52, 0], [0, 1, 0.44, 1.36], [2.33, 0, 1, 0.47], [1.21, 0.51, 0.35, 1]])
    
    dxdt = []
    dxdt.append(r_i[0] * x[0] * (1 - (a_ij[0, 0] * x[0] + a_ij[0, 1] * x[1] + a_ij[0, 2] * x[2] + a_ij[0, 3] * x[3])))
    dxdt.append(r_i[1] * x[1] * (1 - (a_ij[1, 0] * x[0] + a_ij[1, 1] * x[1] + a_ij[1, 2] * x[2] + a_ij[1, 3] * x[3])))
    dxdt.append(r_i[2] * x[2] * (1 - (a_ij[2, 0] * x[0] + a_ij[2, 1] * x[1] + a_ij[2, 2] * x[2] + a_ij[2, 3] * x[3])))
    dxdt.append(r_i[3] * x[3] * (1 - (a_ij[3, 0] * x[0] + a_ij[3, 1] * x[1] + a_ij[3, 2] * x[2] + a_ij[3, 3] * x[3])))
    
    return np.array(dxdt)
    
def func_rikitake(x, t, params):
    # https://www.sciencedirect.com/science/article/pii/S0167278908003849
    
    if params.size == 0:
        mu = 2
        a = 5
    else:
        mu = params[0]
        a = params[1]

    return np.array([- mu * x[0] + x[2] * x[1], - mu * x[1] + x[0] * (x[2] - a), 1 - x[0] * x[1]])

def func_aizawa(x, t, params):
    if params.size == 0:
        a = 0.95
        b = 0.7
        c = 0.6
        d = 3.5
        e = 0.25
        f = 0.1
    else:
        a = params[0]
        b = params[1]
        c = params[2]
        d = params[3]
        e = params[4]
        f = params[5]
    
    dxdt = []
    dxdt.append( (x[2] - b) * x[0] - d * x[1] )
    dxdt.append( d * x[0] + (x[2] - b) * x[1] )
    dxdt.append( c + a * x[2] - x[2] ** 3 / 3 - (x[0] ** 2 + x[1] ** 2) * (1 + e * x[2]) + f * x[2] * x[0] ** 3)
    
    return np.array(dxdt)

def func_bouali2(x, t, params):
    # https://link.springer.com/article/10.1007/s11071-012-0625-6
    if params.size == 0:
        alpha = 0.3
        beta = 0.05
        a = 4
        b = 1
        c = 1.5
        s = 1
    else:
        alpha = params[0]
        beta = params[1]
        a = params[2]
        b = params[3]
        c = params[4]
        s = params[5]
    
    dxdt = []
    dxdt.append( x[0] * (a - x[1]) + alpha * x[2] )
    dxdt.append( - x[1] * (b - x[0] ** 2) )
    dxdt.append( -x[0] * (c - s * x[2]) - beta * x[2] )
    
    return np.array(dxdt)

def func_bouali3(x, t, params):
    # https://arxiv.org/ftp/arxiv/papers/1311/1311.6128.pdf
    if params.size == 0:
        alpha = 3
        beta = 2.2
        gamma = 1
        mu = 0.001
    else:
        alpha = params[0]
        beta = params[1]
        gamma = params[2]
        mu = params[3]

    dxdt = []
    dxdt.append( alpha * x[0] * (1 - x[1]) - beta * x[2] )
    dxdt.append( - gamma * x[1] * (1 - x[0] ** 2)  )
    dxdt.append( mu * x[0] )
    
    return np.array(dxdt)


def func_wang(x, t, params):
    return np.array([x[0] - x[1] * x[2], 
                     x[0] - x[1] + x[0] * x[2], 
                     - 3 * x[2] + x[0] * x[1] ])


def func_sprott(x, t, params):
    # https://journals.aps.org/pre/pdf/10.1103/PhysRevE.50.R647
    # index (params) represents different chaotic sprott systems.
    index = params[0]
    
    if index == 0:
        # also called Nose–Hoover system
        return np.array([x[1], -x[0] + x[1] * x[2], 1 - x[1] ** 2])
    elif index == 1:
        return np.array([x[1] * x[2], x[0] - x[1], 1 - x[0] * x[1]])
    elif index == 2:
        return np.array([x[1] * x[2], x[0] - x[1], 1 - x[0] ** 2])
    elif index == 3:
        return np.array([-x[1], x[0] + x[2],  x[0] * x[2] + 3 * x[1] ** 2] )
    elif index == 4:
        return np.array([x[1] * x[2], x[0] ** 2 - x[1], 1 - 4 * x[0]])
    elif index == 5:
        return np.array([x[1] + x[2], - x[0] + x[1] / 2, x[0] ** 2 - x[2]])
    elif index == 6:
        return np.array([0.4 * x[0] + x[2], x[0] * x[2] - x[1], - x[0] + x[1]])
    elif index == 7:
        return np.array([-x[1] + x[2] ** 2, x[0] + x[1] / 2, x[0] - x[2]])
    elif index == 8:
        return np.array([-0.2 * x[1], x[0] + x[2], x[0] + x[1] ** 2 - x[2]])
    elif index == 9:
        return np.array([2 * x[2], - 2 * x[1] + x[2], - x[0] + x[1] + x[1] ** 2])
    elif index == 10:
        return np.array([x[0] * x[1] - x[2], x[0] - x[1], x[0] + 0.3 * x[2]])
    elif index == 11:
        return np.array([x[1] + 3.9 * x[2], 0.9 * x[0] ** 2 - x[1], 1 - x[0]])
    elif index == 12:
        return np.array([ - x[2], -x[0] ** 2 - x[1], 1.7 + 1.7 * x[0] + x[1] ])
    elif index == 13:
        return np.array([ - 2 * x[1], x[0] + x[2] ** 2, 1 + x[1] - 2 * x[2] ])
    elif index == 14:
        return np.array([ x[1], x[0] - x[2], x[0] + x[0] * x[2] + 2.7 * x[1] ])
    elif index == 15:
        return np.array([2.7 * x[1] + x[2], - x[0] + x[1] ** 2, x[0] + x[1]])
    # 1
    elif index == 16:
        return np.array([-x[2], x[0]-x[1], 3.1 * x[0] + x[1] ** 2 + 0.5 * x[2]])
    elif index == 17:
        return np.array([0.9-x[1], 0.4+x[2], x[0] * x[1] - x[2]])
    elif index == 18:
        return np.array([-x[0]-4 * x[1], x[0] + x[2] ** 2, 1 + x[0]])
    else:
        print('inedx exceeds the number of systems!')
    

def func_chua(x, t, params):
    if params.size == 0:
        alpha = 15.6
        gamma = 1
        beta = 28
    else:
        alpha = params[0]
        gamma = params[1]
        beta = params[2]
    
    mu0 = -1.143
    mu1 = -0.714
    
    ht = mu1 * x[0] + 0.5 * (mu0 - mu1) * (np.abs(x[0] + 1) - np.abs(x[0] - 1))

    dxdt =  []
    
    dxdt.append( alpha * (x[1] - x[0] - ht) )
    dxdt.append(gamma * (x[0] - x[1] + x[2]))
    dxdt.append(- beta * x[1])
    
    return np.array(dxdt)

def func_dadras(x, t, params):
    # https://www.dynamicmath.xyz/strange-attractors/
    if params.size == 0:
        a = 3.0
        b = 2.7
        c = 1.7
        d = 2.0
        e = 9.0
    else:
        a = params[0]
        b = params[1]
        c = params[2]
        d = params[3]
        e = params[4]
        
    return np.array([x[1] - a * x[0] + b * x[1] * x[2],
                     c * x[1] - x[0] * x[2] + x[2],
                     d * x[0] * x[1] - e * x[2]])

def func_four_wing(x, t, params):
    # https://www.dynamicmath.xyz/strange-attractors/
    if params.size == 0:
        a = 0.2
        b = 0.01
        c = -0.4
    else:
        a = params[0]
        b = params[1]
        c = params[2]
        
    return np.array([a * x[0] + x[1] * x[2], 
                     b * x[0] + c * x[1] - x[0] * x[2],
                     -x[2] - x[0] * x[1]])

def rk4(f, x0, t, params=np.array([])):
    n = len(t)
    x = np.zeros((n, len(x0)))
    x[0] = x0
    
    h = t[1] - t[0]
    
    for i in range(n-1):
        if len(params.shape) > 1:
            params_step = params[i, :]
        else:
            params_step = params
        k1 = f(x[i], t[i], params_step)
        k2 = f(x[i] + k1 * h / 2., t[i] + h / 2., params_step)
        k3 = f(x[i] + k2 * h / 2., t[i] + h / 2., params_step)
        k4 = f(x[i] + k3 * h, t[i] + h, params_step)
        x[i+1] = x[i] + (h / 6.) * (k1 + 2 * k2 + 2 * k3 + k4)
        
    return x


if __name__ == '__main__':
    print('chaotic systems')
    
    #################### generate lorenz
    # dt = 0.01
    # t_end = 100
    # t_all = np.arange(0, t_end, dt)
    # x0 = [28 * np.random.rand()-14, 30 * np.random.rand()-15, 20 * np.random.rand()]

    # ts = rk4(func_lorenz, x0, t_all, params=np.array([10, 28, 8 / 3])) 
    
    # fig, ax = plt.subplots(3, 1, figsize=(8, 13))
    # ax[0].plot(t_all, ts[:, 0])
    # ax[1].plot(t_all, ts[:, 1])
    # ax[2].plot(t_all, ts[:, 2])
    
    #################### generate rossler
    # dt = 0.01
    # t_end = 100
    # t_all = np.arange(0, t_end, dt)
    # x0 = [28 * np.random.rand()-14, 30 * np.random.rand()-15, 20 * np.random.rand()]

    # ts = rk4(func_rossler, x0, t_all, params=np.array([0.2, 0.2, 5.7]))
    
    # fig, ax = plt.subplots(3, 1, figsize=(8, 13))
    # ax[0].plot(t_all, ts[:, 0])
    # ax[1].plot(t_all, ts[:, 1])
    # ax[2].plot(t_all, ts[:, 2])
    
    #################### generate foodchain
    # dt = 0.01
    # t_end = 3000
    # t_all = np.arange(0, t_end, dt)
    # x0 = [0.4 * np.random.rand() + 0.6, 0.4 * np.random.rand() - 0.15, 0.5 * np.random.rand() + 0.3]

    # ts = rk4(func_foodchain, x0, t_all, params=np.array([0.94, 1.7, 5.0]))
    # # ts = rk4(func_foodchain, x0, t_all, params=np.array([0.94, 2.009, 2.876]))
    
    # fig, ax = plt.subplots(3, 1, figsize=(8, 13))
    
    # ax[0].plot(t_all, ts[:, 0])
    # ax[1].plot(t_all, ts[:, 1])
    # ax[2].plot(t_all, ts[:, 2])
    
    # plt.show()
    
    # fig, ax = plt.subplots(3, 1, figsize=(8, 13))
    
    # plot_length = 20000
    # ax[0].plot(t_all[:plot_length], ts[:plot_length, 0])
    # ax[1].plot(t_all[:plot_length], ts[:plot_length, 1])
    # ax[2].plot(t_all[:plot_length], ts[:plot_length, 2])
    
    # plt.show()
    
    # ax = plt.figure().add_subplot(projection='3d')
    # ax.plot(ts[:, 0], ts[:, 1], ts[:, 2], color='orange')
    
    # plt.show()
    
    #################### generate hastings
    dt = 0.01
    t_end = 3000
    t_all = np.arange(0, t_end, dt)
    x0 = [(0.8 - 0.5) * np.random.rand() + 0.5, 0.3 * np.random.rand(), (10 - 8) * np.random.rand() + 8]

    # ts = rk4(func_hastings, x0, t_all, params=np.array([5, 0.1, 3, 2, 0.4, 0.01])) # original
    ts = rk4(func_hastings, x0, t_all, params=np.array([7, 0.1, 3, 2, 0.4, 0.01]))
    
    fig, ax = plt.subplots(3, 1, figsize=(8, 13))
    
    ax[0].plot(t_all, ts[:, 0])
    ax[1].plot(t_all, ts[:, 1])
    ax[2].plot(t_all, ts[:, 2])
    
    plt.show()
    
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(ts[:, 0], ts[:, 1], ts[:, 2], color='orange')
    
    ax.view_init(0, 90)
    
    plt.show()
    
    #################### generate lotka-volterra
    # dt = 0.01
    # t_end = 1000
    # t_all = np.arange(0, t_end, dt)
    # x0 = [(0.5 - 0.2) * np.random.rand() + 0.2, (0.5 - 0.2) * np.random.rand() + 0.2, 
    #       (0.5 - 0.2) * np.random.rand() + 0.2, (0.5 - 0.2) * np.random.rand() + 0.2]

    # ts = rk4(func_lotka_volterra, x0, t_all, params=np.array([1]))
    
    # fig, ax = plt.subplots(3, 1, figsize=(8, 13))
    
    # ax[0].plot(t_all, ts[:, 0])
    # ax[1].plot(t_all, ts[:, 1])
    # ax[2].plot(t_all, ts[:, 2])
    
    # plt.show()
    
    # ax = plt.figure().add_subplot(projection='3d')
    # ax.plot(ts[:, 0], ts[:, 1], ts[:, 2], color='orange')
    
    # plt.show()

    

































































