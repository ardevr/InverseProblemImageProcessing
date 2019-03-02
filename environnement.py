#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 16:12:47 2019

@author: evrardgarcelon
"""

import numpy as np
from scipy.special import j0, y0
from numpy.fft import fft, ifft
from gaussian_process import GaussianProcess


class Environnement():

    def __init__(self, c0=1, L=50, N=10, nb_timesteps=100, seed = 1, eps = 10**-10):

        self.L = L
        self.eps = eps
        self.N = N
        self.c0 = c0
        self.G_hat = lambda w, x, y: 1j / 4 * \
            (np.nan_to_num(j0(w * np.linalg.norm(x - y))) + 1j * np.nan_to_num(
                    y0(w * np.linalg.norm(x - y))))
        self.G = lambda t, x, y:  np.nan_to_num(1/(2*np.pi)*(t**2 > np.dot(x-y,x-y) + self.eps)/np.sqrt(t**2 - np.dot(x-y,x-y)))
        self.nb_timesteps = nb_timesteps
        self.seed = seed

    def compute_signal(self, x1, x2, tau, T, eps=10**-10):
        
        time_discretization = np.linspace(eps, T, self.nb_timesteps)
        thetas = np.random.uniform(size=self.N)
        y = np.zeros((self.N, 2))
        y[:, 0] = self.L*np.cos(2 * np.pi * thetas)
        y[:, 1] = self.L*np.sin(2 * np.pi * thetas)
        self.y = y
        def cov(x): return np.exp(-x**2 / 4) * (2 - x**2) / (4 * np.sqrt(2))
        gp = GaussianProcess(cov)
        signal_x1 = 0
        signal_x2 = 0
        for i in range(self.N):
            temp_gp = gp.generate_n_var(
                self.nb_timesteps, time_discretization, fourier=False)
            temp_G1 = fft(self.G(time_discretization, x1, y[i]))
            temp_G2 = fft(self.G(time_discretization + tau, x2, y[i]))
            signal_x1 += np.real(ifft(temp_G1 * temp_gp))
            signal_x2 += np.real(ifft(temp_G2 * temp_gp))
        return self.L / np.sqrt(self.N) * \
            signal_x1, self.L / np.sqrt(self.N) * signal_x2, time_discretization

if __name__ == '__main__' : 
    env = Environnement(N = 50)
    x1 = x2 = np.ones(2)
    tau = 1
    u,u_lagged, time  = env.compute_signal(x1,x2,tau, 100, eps = 0)
    import pylab as plt
    plt.scatter(time, u,color = 'red',label = 'u')
    plt.scatter(time, u_lagged, color = 'blue',label = 'u_lagged')
    