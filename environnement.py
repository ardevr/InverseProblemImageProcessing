#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 16:12:47 2019

@author: evrardgarcelon
"""

import numpy as np
from scipy.special import j0, y0, hankel1
from numpy.fft import fft, ifft, fftfreq
from scipy.integrate import quad


def generate_n_gp(time_points, fourier_cov, return_fourier = True, eps=10**-5):
        
    assert isinstance(time_points, np.ndarray)
    
    n = len(time_points)
    y_fourier = fft(np.random.randn(n))
    delta_t = time_points[1] - time_points[0]
    freqs = fftfreq(n, d=delta_t)
    z_fourier = np.sqrt(fourier_cov(freqs)) * y_fourier
    if return_fourier :
        return z_fourier
    else :
        return np.real(ifft(z_fourier))


class Environnement():

    def __init__(self, c0=1, L=50, N=50, nb_timesteps=100, fourier_cov = lambda w :np.exp(-w**2)) :

        self.L = L
        self.N = N
        self.c0 = c0
        self.nb_timesteps = nb_timesteps
        thetas = np.random.uniform(size=N)
        self.ys = np.zeros((N, 2))
        self.ys[:, 0] = self.L * np.cos(2 * np.pi * thetas)
        self.ys[:, 1] = self.L * np.sin(2 * np.pi * thetas)
        self.fourier_cov = fourier_cov

    def G_hat(self,w, x, y) : 
            norme = np.linalg.norm(x - y)/self.c0
            return np.nan_to_num((1j/4*hankel1(0,w*norme)))
        
    def G(self,t, x, y) : 
            norm = np.linalg.norm(x - y)/self.c0
            if t > norm/self.c0 : 
                return 1/(2*np.pi*np.sqrt(t**2 - norm**2))
            else : 
                return 0
    def fourier_gaussian(self,w) :
    
        temp_y = np.random.randn(4)
        C = np.eye(4)/2
        C[2,0] = C[0,2]  = 1/2
        C[3,1] = C[1,3] = -1/2
        cplx_gaussian = np.dot(C,temp_y)
        return cplx_gaussian[0]
    

    def compute_signal(self, x1, x2, T):

        time_discretization = np.linspace(0, T, self.nb_timesteps)

        delta_T = (time_discretization[1] -  time_discretization[0])/T
        freqs = np.fft.fftfreq(self.nb_timesteps, d=delta_T)
        sum_gps_1 = np.zeros(len(freqs))
        sum_gps_2 = np.zeros(len(freqs))
        for j in range(self.N) : 
            y_fourier = fft(np.random.randn(len(freqs)))
            z_fourier = np.sqrt(self.fourier_cov(freqs)) * y_fourier
            G_freqs_1 = np.array([self.G_hat(w,x1,self.ys[j]) for w in freqs])
            G_freqs_2 = np.array([self.G_hat(w,x2,self.ys[j]) for w in freqs])
            z_fourier = z_fourier
            sum_gps_1 = sum_gps_1 + z_fourier*G_freqs_1
            sum_gps_2 = sum_gps_2 + z_fourier*G_freqs_2
        
        signal_1 = np.real(ifft(sum_gps_1))
        signal_2 = np.real(ifft(sum_gps_2))
        
        #signal_1 = (signal_1 - np.mean(signal_1))/np.std(signal_1)
        #signal_2 = (signal_2 - np.mean(signal_2))/np.std(signal_2)
        

        return self.c0/ np.sqrt(self.N) * \
            signal_1, self.c0/ np.sqrt(self.N) * \
            signal_2, time_discretization, self.ys


if __name__ == '__main__':
    env = Environnement(N=10, nb_timesteps= 111, c0 = 1, L = 50)
    x1 = x2 = np.ones(2)
    x2 = x2 + 1/2*x1
    tau = 1
    u1, u2, time, _ = env.compute_signal(x1, x2, 100)
    u1, u2 = u1, u2
    import pylab as plt
    plt.scatter(time, u1, color='red', label='u1')
    plt.scatter(time, u2, color='blue', label='u2')
    plt.legend()
