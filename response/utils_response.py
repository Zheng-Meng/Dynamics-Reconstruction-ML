# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 09:43:19 2025

@author: zmzhai
"""

import numpy as np
import os
import pickle
from scipy import signal
import matplotlib.pyplot as plt
from collections import Counter

# compute the PSE, dominant frequecy, and cut off frequecy (threshold 0.98)
def compute_effective_bandwidth(data, fs=1.0, power_threshold=0.98):
    """
    Compute effective bandwidth capturing specified percentage of signal power
    """
    frequencies, psd = signal.welch(data, fs=fs, nperseg=1024)
    
    # Find dominant frequency
    dominant_freq = frequencies[np.argmax(psd)]
    
    # Compute cumulative power
    cumulative_power = np.cumsum(psd) / np.sum(psd)
    
    # Find frequency that captures threshold percentage of power
    effective_freq = frequencies[np.argmax(cumulative_power >= power_threshold)]
    
    return frequencies, psd, cumulative_power, dominant_freq, effective_freq


def recurrence_times_entropy(time_series, r=0.5, dt=1, norm=True):
    """Estimate KS entropy via recurrence times (Baptista et al. 2010)."""
    if norm:
        ts = (time_series - np.mean(time_series, axis=0)) / np.std(time_series, axis=0)
    else:
        ts = time_series.copy()

    T = len(ts)
    recurrence_times = []

    for i in range(T - 1):
        left = False
        for j in range(i + 1, T):
            dist = np.linalg.norm(ts[i] - ts[j])
            if not left:
                if dist > r:
                    left = True
            elif dist < r:
                tau = (j - i) * dt
                recurrence_times.append(tau)
                
                break

    if len(recurrence_times) == 0:
        return np.nan
    
    recurrence_times = [tau for tau in recurrence_times if tau <= max_tau]
    
    denominator = np.mean(recurrence_times)

    count = Counter(recurrence_times)
    total = sum(count.values())
    probs = np.array([v / total for v in count.values()])
    # h_ks = -np.sum(probs * np.log(probs)) / min(recurrence_times)
    
    h_ks = -np.sum(probs * np.log(probs)) / denominator
    
    # denominator = np.mean(count.values())
    # h_ks = -np.sum(probs * np.log(probs)) / denominator

    return h_ks
























