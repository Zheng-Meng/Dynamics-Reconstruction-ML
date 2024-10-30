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

def func_epidemic(x, t, params):
    if params.size == 0:
        m = 0.02
        a = 35.84
        g = 100
        b = 1800
    else:
        m = params[0]
        a = params[1]
        g = params[2]
        b = params[3]
    
    dxdt = []
    dxdt.append( m * (1 - x[0]) - b * x[0] * x[2] )
    dxdt.append( b * x[0] * x[2] - (m + a) * x[1] )
    dxdt.append( a * x[1] - (m + g) * x[2] )
    dxdt.append( g * x[2] - m * x[3] )
    
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

def func_predator_prey(x, t, params):
    if params.size == 0:
        r = 0.9
        k = 5
        beta = 0.4
        gamma = 0.7
        c = 0.1
        r_v = 0.5
        alpha_min = 0.1
        alpha_max = 1.3
    else:
        r = params[0]
        k = params[1]
        beta = params[2]
        gamma = params[3]
        c = params[4]
        r_v = params[5]
        alpha_min = params[6]
        alpha_max = params[7]
    
    m_alpha = (x[2] ** 2 + c ) * x[1]
    
    dxdt = []
    dxdt.append( r * x[0] * (1 - x[0] / k) - x[2] * x[0] * x[1] / (1 + beta * x[0]) )
    dxdt.append( gamma * x[2] * x[0] * x[1] / (1 + beta * x[0]) - m_alpha )
    dxdt.append( r_v * (x[2] - alpha_min) * (alpha_max - x[2]) * (gamma * x[0] / (1 + beta * x[0]) - 2 * x[2]) )
    
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

def func_thomas(x, t, params):
    if params.size == 0:
        b = 0.19
    else:
        b = params[0]
        
    return np.array([np.sin(x[1]) - b * x[0], 
                     np.sin(x[2]) - b * x[1], 
                     np.sin(x[0]) - b * x[2] ])

def func_arneodo(x, t, params):
    # https://www.vorillaz.com/arneodo-attractor/
    if params.size == 0:
        a = 5.5
        b = 3.5
        c = 0.01
    else:
        a = params[0]
        b = params[1]
        c = params[2]
        
    return np.array([x[1], 
                     x[2],
                     a*x[0]-b*x[1]-x[2]-c*x[0]**3])

def func_chen_lee(x, t, params):
    # https://www.vorillaz.com/chen-lee-attractor/
    if params.size == 0:
        alpha = 5.0
        beta = -10.0
        delta = -0.38
    else:
        alpha = params[0]
        beta = params[1]
        delta = params[2]
    
    aaa = 1
        
    return np.array([alpha * x[0] - x[1] * x[2], 
                     beta * x[1] + x[0] * x[2], 
                     delta * x[2] + (x[0] * x[1]) / 3 ])

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

def func_lu_chen(x, t, params):
    # https://en.wikipedia.org/wiki/Multiscroll_attractor
    if params.size == 0:
        a = 36.0
        b = 3.0
        c = 20.0
        u = -15.15
    else:
        a = params[0]
        b = params[1]
        c = params[2]
        u = params[3]
        
    return np.array([a * (x[1] - x[0]), 
                     x[0] - x[0] * x[2] + c * x[1] + u,
                     x[0] * x[1] - b * x[2]])


def func_halvorsen(x, t, params):
    # https://www.dynamicmath.xyz/strange-attractors/
    if params.size == 0:
        a = 1.89
    else:
        a = params[0]
    
    return np.array([-a * x[0] - 4 * x[1] - 4 * x[2] - x[1] ** 2,
                     -a * x[1] - 4 * x[2] - 4 * x[0] - x[2] ** 2,
                     -a * x[2] - 4 * x[0] - 4 * x[1] - x[0] ** 2,])

def func_rabinovich_fabrikant(x, t, params):
    # https://www.dynamicmath.xyz/strange-attractors/
    if params.size == 0:
        alpha = 0.14
        gamma = 0.10
    else:
        alpha = params[0]
        gamma = params[1]
    
    return np.array([x[1] * (x[2] - 1 + x[0] ** 2) + gamma * x[0],
                     x[0] * (3 * x[2] + 1 - x[0] ** 2) + gamma * x[1],
                     - 2 * x[2] * (alpha + x[0] * x[1])])

def func_sprott_2014(x, t, params):
    # https://www.dynamicmath.xyz/strange-attractors/
    if params.size == 0:
        a = 2.07
        b = 1.79
    else:
        a = params[0]
        b = params[1]
    
    return np.array([x[1] + a * x[0] * x[1] + x[0] * x[2],
                     1 - b * x[0] ** 2 + x[1] * x[2],
                     x[0] - x[0] ** 2 - x[1] ** 2])

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

def func_lorenz84(x, t, params):
    # https://www.vorillaz.com/hadley-attractor/ # https://www.dynamicmath.xyz/strange-attractors/
    if params.size == 0:
        a = 0.95
        b = 7.91
        f = 4.83
        g = 4.66
    else:
        a = params[0]
        b = params[1]
        f = params[2]
        g = params[3]
        
    return np.array([- a * x[0] - x[1] ** 2 - x[2] ** 2 + a * f,
                     - x[1] + x[0] * x[1] - b * x[0] * x[2] + g,
                     - x[2] + b * x[0] * x[1] + x[0] * x[2]])
    

    
def func_lotka(x, t, params):
    # 2-d system
    if params.size == 0:
        a = 2 / 3
        b = 4 / 3
        c = 1
        d = 1
    else:
        a = params[0]
        b = params[1]
        c = params[2]
        d = params[3]
    
    return np.array([a * x[0] - b * x[0] * x[1], - c * x[1] + d * x[0] * x[1]])


def func_lorenz96(x, t, params):
    N = len(x)
    if params.size == 0:
        F = 8
    else:
        F = params[0]
    
    dxdt = np.zeros((N, ))

    for i in range(N):
        if i == 0:
            dxdt[i] = (x[i+1] - x[N-2]) * x[N-1] - x[i] + F
        elif i == 1:
            dxdt[i] = (x[i+1] - x[N-1]) * x[i-1] - x[i] + F
        elif i == N-1:
            dxdt[i] = (x[0] - x[i-2]) * x[i-1] - x[i] + F
        else:
            dxdt[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i] + F
    
    return dxdt

            
def func_mackeyglass(x, x_tau, t):
    # time-delay
    beta = 0.2
    gamma = 0.1
    power = 10
    
    dxdt = beta * x_tau / (1 + x_tau ** power) - gamma * x
    
    return dxdt

def func_harmonic(x, t, params):
    return np.array([x[1], - 4 * x[0]])


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

def rk4_delay(f, x0, t, params=np.array([])):
    h = t[1] - t[0]
    n = len(t)
    x = np.zeros((n, len(x0)))
    x[:round(100/h), :] = x0
    
    for i in range(round(100/h), n-1):
        if len(params.shape) > 1:
            params_step = params[i, :]
        else:
            params_step = params
        
        tau_integer = round(params_step[0] / h)
        
        k1 = f(x[i], x[i-tau_integer], t[i])
        k2 = f(x[i] + h/2 * k1 , x[i-tau_integer], t[i] + h/2)
        k3 = f(x[i] + h/2 * k2 , x[i-tau_integer], t[i] + h/2)
        k4 = f(x[i] + h * k3 , x[i-tau_integer], t[i] + h)
        x[i+1] = x[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        
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
    
    #################### generate epidemic
    # dt = 0.01
    # t_end = 1000
    # t_all = np.arange(0, t_end, dt)
    # x0 = [(0.7 - 0.5) * np.random.rand() + 0.5, 0.1 * np.random.rand(), 0.1 * np.random.rand(), 0.0]
    # x0[3] = 1 - x0[0] - x0[1] - x0[2]
    # time_divide = 1
    # m, a, g = 0.02 / time_divide, 35.84 / time_divide, 100 / time_divide
    # b0, b1 = 1800 / time_divide, 0.28 / time_divide
    
    # mt, at, gt = m * np.ones((len(t_all), 1)), a * np.ones((len(t_all), 1)), g * np.ones((len(t_all), 1))
    # bt = b0 * (1 + b1 * np.cos(2 * np.pi * t_all ))
    # bt = np.expand_dims(bt, axis=1)
    
    # params = np.concatenate((mt, at, gt, bt), axis=1)
    
    # ts = rk4(func_epidemic, x0, t_all, params=params)
    
    # fig, ax = plt.subplots(3, 1, figsize=(8, 13))
    
    # ax[0].plot(t_all, ts[:, 0])
    # ax[1].plot(t_all, ts[:, 1])
    # ax[2].plot(t_all, ts[:, 2])
    
    # plt.show()
    
    # ax = plt.figure().add_subplot(projection='3d')
    # ax.plot(ts[1000:, 0], ts[1000:, 1], ts[1000:, 2], color='orange')
    
    # plt.show()
    
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
    
    # r = 0.9
    # k = 5
    # beta = 0.4
    # gamma = 0.7
    # c = 0.1
    # r_v = 0.5
    # alpha_min = 0.1
    # alpha_max = 1.3
    # #################### generate predator_prey
    # dt = 0.01
    # t_end = 1000
    # t_all = np.arange(0, t_end, dt)
    # x0 = [(2 - 1) * np.random.rand() + 1, (6 - 2) * np.random.rand() + 2, (1.2 - 0.2) * np.random.rand() + 0.2]
    
    # period = 150
    # kt = (10 - 3) * np.abs(np.mod(t_all, period) - period / 2) + 3
    # kt = np.expand_dims(kt, axis=1)
    
    # rt = 0.9 * np.ones((len(t_all), 1))
    # beta_t = 0.4 * np.ones((len(t_all), 1))
    # gamma_t = 0.7 * np.ones((len(t_all), 1))
    # c_t = 0.1 * np.ones((len(t_all), 1))
    # r_v_t = 0.5 * np.ones((len(t_all), 1))
    # alpha_min_t = 0.1 * np.ones((len(t_all), 1))
    # alpha_max_t = 1.3 * np.ones((len(t_all), 1))
    
    # # plt.plot(t_all, kt)
    # # plt.show()
    
    # # params=np.array([0.9, 15, 0.4, 0.7, 0.1, 0.5, 0.1, 1.3])
    # params = np.concatenate((rt, kt, beta_t, gamma_t, c_t, r_v_t, alpha_min_t, alpha_max_t), axis=1)
    # ts = rk4(func_predator_prey, x0, t_all, params=params)
    
    # fig, ax = plt.subplots(3, 1, figsize=(8, 13))
    
    # ax[0].plot(t_all, ts[:, 0])
    # ax[1].plot(t_all, ts[:, 1])
    # ax[2].plot(t_all, ts[:, 2])
    
    # plt.show()
    
    # ax = plt.figure().add_subplot(projection='3d')
    # ax.plot(ts[1000:, 0], ts[1000:, 1], ts[1000:, 2], color='orange')
    
    # plt.show()
    
    #################### generate rikitake
    # dt = 0.01
    # t_end = 100
    # t_all = np.arange(0, t_end, dt)
    # x0 = [0.1 * np.random.rand(), 0.1 * np.random.rand(), 0.1 * np.random.rand()]

    # ts = rk4(func_rikitake, x0, t_all, params=np.array([2, 5]))
    
    # fig, ax = plt.subplots(3, 1, figsize=(8, 13))
    
    # ax[0].plot(t_all, ts[:, 0])
    # ax[1].plot(t_all, ts[:, 1])
    # ax[2].plot(t_all, ts[:, 2])
    
    #################### generate wang
    # dt = 0.01
    # t_end = 100
    # t_all = np.arange(0, t_end, dt)
    # x0 = [0.1 * np.random.rand(), 0.1 * np.random.rand(), 0.1 * np.random.rand()]

    # ts = rk4(func_wang, x0, t_all)
    
    # fig, ax = plt.subplots(3, 1, figsize=(8, 13))
    
    # ax[0].plot(t_all, ts[:, 0])
    # ax[1].plot(t_all, ts[:, 1])
    # ax[2].plot(t_all, ts[:, 2])
    
    ################### generate sprott
    # dt = 0.01
    # t_end = 100
    # t_all = np.arange(0, t_end, dt)
    # x0 = [0.1 * np.random.rand(), 0.1 * np.random.rand(), 0.1 * np.random.rand()]

    # ts = rk4(func_sprott, x0, t_all, params=np.array([0]))
    
    # fig, ax = plt.subplots(3, 1, figsize=(8, 13))
    
    # ax[0].plot(t_all, ts[:, 0])
    # ax[1].plot(t_all, ts[:, 1])
    # ax[2].plot(t_all, ts[:, 2])
    
    #################### generate chua
    # dt = 0.01
    # t_end = 100
    # t_all = np.arange(0, t_end, dt)
    # x0 = [1*np.random.rand(), 1*np.random.rand(), 1*np.random.rand()]

    # ts = rk4(func_chua, x0, t_all, params=np.array([15.6, 1, 28]))
    
    # fig, ax = plt.subplots(3, 1, figsize=(8, 13))
    
    # ax[0].plot(t_all, ts[:, 0])
    # ax[1].plot(t_all, ts[:, 1])
    # ax[2].plot(t_all, ts[:, 2])
    
    #################### generate thomas
    # dt = 0.01
    # t_end = 1000
    # t_all = np.arange(0, t_end, dt)
    # x0 = [0.5*np.random.rand(), 0.5*np.random.rand(), 0.5*np.random.rand()]

    # ts = rk4(func_thomas, x0, t_all, params=np.array([0.19]))
    
    # fig, ax = plt.subplots(3, 1, figsize=(8, 13))
    
    # ax[0].plot(t_all, ts[:, 0])
    # ax[1].plot(t_all, ts[:, 1])
    # ax[2].plot(t_all, ts[:, 2])
    
    
    #################### generate arneodo
    # dt = 0.01
    # t_end = 1000
    # t_all = np.arange(0, t_end, dt)
    # x0 = [1*np.random.rand(), 1*np.random.rand(), 0.1*np.random.rand()]

    # ts = rk4(func_arneodo, x0, t_all, params=np.array([5.5, 3.5, 0.01]))
    
    # fig, ax = plt.subplots(3, 1, figsize=(8, 13))
    
    # ax[0].plot(t_all, ts[:, 0])
    # ax[1].plot(t_all, ts[:, 1])
    # ax[2].plot(t_all, ts[:, 2])
    
    #################### generate chen-lee
    # dt = 0.01
    # t_end = 1000
    # t_all = np.arange(0, t_end, dt)
    # x0 = [0.5*np.random.rand(), 0.5*np.random.rand(), 0.5*np.random.rand()]

    # ts = rk4(func_chen_lee, x0, t_all, params=np.array([5.0, -10.0, -0.38]))
    
    # fig, ax = plt.subplots(3, 1, figsize=(8, 13))
    
    # ax[0].plot(t_all, ts[:, 0])
    # ax[1].plot(t_all, ts[:, 1])
    # ax[2].plot(t_all, ts[:, 2])
    
    #################### generate dadras
    # dt = 0.01
    # t_end = 1000
    # t_all = np.arange(0, t_end, dt)
    # x0 = [0.5*np.random.rand(), 0.5*np.random.rand(), 0.5*np.random.rand()]

    # ts = rk4(func_dadras, x0, t_all, params=np.array([3.0, 2.7, 1.7, 2.0, 9.0]))
    
    # fig, ax = plt.subplots(3, 1, figsize=(8, 13))
    
    # ax[0].plot(t_all, ts[:, 0])
    # ax[1].plot(t_all, ts[:, 1])
    # ax[2].plot(t_all, ts[:, 2])
    
    #################### generate lu-chen
    # dt = 0.01
    # t_end = 100
    # t_all = np.arange(0, t_end, dt)
    # x0 = [0.1*np.random.rand(), 0.3*np.random.rand(), -0.6*np.random.rand()]

    # ts = rk4(func_lu_chen, x0, t_all, params=np.array([36, 3, 20, -15.15]))
    
    # fig, ax = plt.subplots(3, 1, figsize=(8, 13))
    
    # ax[0].plot(t_all, ts[:, 0])
    # ax[1].plot(t_all, ts[:, 1])
    # ax[2].plot(t_all, ts[:, 2])
    
    #################### generate halvorsen
    # dt = 0.01
    # t_end = 100
    # t_all = np.arange(0, t_end, dt)
    # x0 = [0.1*np.random.rand(), 0.3*np.random.rand(), -0.6*np.random.rand()]

    # ts = rk4(func_halvorsen, x0, t_all, params=np.array([1.89]))
    
    # fig, ax = plt.subplots(3, 1, figsize=(8, 13))
    
    # ax[0].plot(t_all, ts[:, 0])
    # ax[1].plot(t_all, ts[:, 1])
    # ax[2].plot(t_all, ts[:, 2])
    
    #################### generate rabinovich_fabrikant
    # dt = 0.01
    # t_end = 100
    # t_all = np.arange(0, t_end, dt)
    # x0 = [0.5*np.random.rand(), 0.5*np.random.rand(), 0.5*np.random.rand()]

    # ts = rk4(func_rabinovich_fabrikant, x0, t_all, params=np.array([0.14, 0.10]))
    
    # fig, ax = plt.subplots(3, 1, figsize=(8, 13))
    
    # ax[0].plot(t_all, ts[:, 0])
    # ax[1].plot(t_all, ts[:, 1])
    # ax[2].plot(t_all, ts[:, 2])
    
    #################### generate sprott 2014
    # dt = 0.01
    # t_end = 100
    # t_all = np.arange(0, t_end, dt)
    # x0 = [0.5*np.random.rand(), 0.5*np.random.rand(), 0.5*np.random.rand()]

    # ts = rk4(func_sprott_2014, x0, t_all, params=np.array([2.07, 1.79]))
    
    # fig, ax = plt.subplots(3, 1, figsize=(8, 13))
    
    # ax[0].plot(t_all, ts[:, 0])
    # ax[1].plot(t_all, ts[:, 1])
    # ax[2].plot(t_all, ts[:, 2])
    
    #################### generate four wing
    # dt = 0.01
    # t_end = 100
    # t_all = np.arange(0, t_end, dt)
    # x0 = [0.5*np.random.rand(), 0.5*np.random.rand(), 0.5*np.random.rand()]

    # ts = rk4(func_four_wing, x0, t_all, params=np.array([0.2, 0.01, -0.4]))
    
    # fig, ax = plt.subplots(3, 1, figsize=(8, 13))
    
    # ax[0].plot(t_all, ts[:, 0])
    # ax[1].plot(t_all, ts[:, 1])
    # ax[2].plot(t_all, ts[:, 2])
    
    #################### generate lorenz84
    # dt = 0.01
    # t_end = 100
    # t_all = np.arange(0, t_end, dt)
    # x0 = [0.5*np.random.rand(), 0.5*np.random.rand(), 0.5*np.random.rand()]

    # ts = rk4(func_lorenz84, x0, t_all, params=np.array([0.95, 7.91, 4.83, 4.66]))
    
    # fig, ax = plt.subplots(3, 1, figsize=(8, 13))
    
    # ax[0].plot(t_all, ts[:, 0])
    # ax[1].plot(t_all, ts[:, 1])
    # ax[2].plot(t_all, ts[:, 2])
    
    #################### generate aizawa
    # dt = 0.01
    # t_end = 300
    # t_all = np.arange(0, t_end, dt)
    # x0 = [0.1*np.random.rand(), 0.1*np.random.rand(), 0.1*np.random.rand()]
    # # x0 = [0.1, 0, 0]

    # ts = rk4(func_aizawa, x0, t_all, params=np.array([0.95, 0.7, 0.6, 3.5, 0.25, 0.1]))
        
    # ax = plt.axes(projection='3d')
    # ax.plot3D(ts[:, 0], ts[:, 1], ts[:, 2], 'gray')
    
    # fig, ax = plt.subplots(3, 1, figsize=(8, 13))
    
    # ax[0].plot(t_all, ts[:, 0])
    # ax[1].plot(t_all, ts[:, 1])
    # ax[2].plot(t_all, ts[:, 2])
    
    #################### generate bouali
    # dt = 0.01
    # t_end = 300
    # t_all = np.arange(0, t_end, dt)
    # x0 = [0.1*np.random.rand(), 0.1*np.random.rand(), 0.1*np.random.rand()]

    # ts = rk4(func_bouali2, x0, t_all, params=np.array([0.3, 0.05, 4, 1, 1.5, 1]))
        
    # ax = plt.axes(projection='3d')
    # ax.plot3D(ts[:, 0], ts[:, 1], ts[:, 2])
    
    # fig, ax = plt.subplots(3, 1, figsize=(8, 13))
    
    # ax[0].plot(t_all, ts[:, 0])
    # ax[1].plot(t_all, ts[:, 1])
    # ax[2].plot(t_all, ts[:, 2])
    
    #################### generate bouali
    # dt = 0.01
    # t_end = 300
    # t_all = np.arange(0, t_end, dt)
    # x0 = [0.1*np.random.rand(), 0.1*np.random.rand(), 0.1*np.random.rand()]

    # ts = rk4(func_bouali3, x0, t_all, params=np.array([3, 2.2, 1, 0.001]))
        
    # ax = plt.axes(projection='3d')
    # ax.plot3D(ts[:, 0], ts[:, 1], ts[:, 2])
    
    # fig, ax = plt.subplots(3, 1, figsize=(8, 13))
    
    # ax[0].plot(t_all, ts[:, 0])
    # ax[1].plot(t_all, ts[:, 1])
    # ax[2].plot(t_all, ts[:, 2])
     
    
    ################### generate lorenz96
    # dt = 0.01
    # t_end = 500
    # t_all = np.arange(0, t_end, dt)
    # N = 40
    # x0 = np.random.rand(N)

    # ts = rk4(func_lorenz96, x0, t_all, params=np.array([8]))
        
    # ax = plt.axes(projection='3d')
    # ax.plot3D(ts[:, 0], ts[:, 1], ts[:, 2])

    # # lorenz96 system
    # fig, ax = plt.subplots(1, 1, figsize=(6,6))
    # # plot_dim_x = min([np.shape(u)[0], 10000])
    # plot_dim_x = 1000
    
    # ts = ts[2000:, :]
    
    # x_plot = ts[:plot_dim_x,:]
    # N_plot = np.shape(x_plot)[0]
    # # Plotting the contour plot
    # # t, s = np.meshgrid(np.arange(N_plot)*dt, np.array(range(N))+1)
    # n, s = np.meshgrid(np.arange(plot_dim_x), np.array(range(N))+1)
    # plt.contourf(s, n, np.transpose(x_plot), 15, cmap=plt.get_cmap("seismic"))
    # plt.colorbar()
    # plt.xlabel(r"$x$")
    # plt.ylabel(r"$n, \quad t=n \cdot {:}$".format(dt))
    
    # plt.savefig('lorenz96.png', dpi=600)
    
    #################### generate mackeyglass
    # dt = 0.01
    # t_end = 1000
    # t_all = np.arange(0, t_end, dt)
    # x0 = [1.0 + np.random.rand()]

    # ts = rk4_delay(func_mackeyglass, x0, t_all, params=np.array([30]))

    # fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # ax.plot(t_all, ts[:, 0])






























































