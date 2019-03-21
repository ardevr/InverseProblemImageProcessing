#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 16:12:47 2019

@author: evrardgarcelon
"""

import numpy as np
from scipy.special import j0, y0, hankel1
from numpy.fft import fft, ifft, fftfreq
from tqdm import trange
from numpy import convolve


def generate_n_gp(time_points,return_fourier = True):
        
    f = lambda x : np.exp(-x**2/4)*(2-x**2)/(4*np.sqrt(2))
    n = len(time_points)
    tau = time_points[-1]/(len(time_points)-1)
    burn_in = int(10/tau)
    temp_points = np.linspace(-10, time_points[-1]+10, 2*burn_in + len(time_points))
    y_fourier = fft(np.random.randn(len(temp_points)))
    f_fourier = (fft(f(temp_points)))
    z_fourier = np.sqrt(f_fourier)*y_fourier
    if return_fourier :
        return z_fourier
    else :
        inverse_z = ifft(z_fourier)[burn_in+1:-burn_in+1]
        return np.real(inverse_z)


class Environnement():

    def __init__(self, c0=1, L=50, N=50, nb_timesteps=100, fourier_cov = lambda 
                 w : w**2*np.exp(-w**2), describe = True) :

        self.L = L
        self.N = N
        self.describe = describe
        self.c0 = c0
        self.nb_timesteps = nb_timesteps
        self.ys = L*self.gen_ys(N)
        self.fourier_cov = fourier_cov
        
    def gen_ys(self,N) :
        thetas = np.random.uniform(low = 0, high = 2*np.pi, size = N)
        temp_y = np.zeros((N,2))
        for i in range(N) :
            temp_y[i] = np.array([np.cos(thetas[i]), np.sin(thetas[i])])
        return temp_y

    def G_hat(self, w, x, y) : 
            norme = np.linalg.norm(x - y)/self.c0
            if np.abs(w) > 0 : 
                return 1j/4*np.sign(w)*j0(np.abs(w)*norme) - 1/4*np.nan_to_num(y0(np.abs(w)*norme))
            else :
                return 1j/4
    def G(self, t, x, y) : 
            norm = np.linalg.norm(x - y)/self.c0
            if t > norm: 
                return 1/(2*np.pi*self.c0**2*np.sqrt(t**2 - norm**2))
            else : 
                return 0
    
    def compute_signal(self, x1, x2, T):

        time_discretization = np.linspace(0, T, self.nb_timesteps)
        self.tau = T/(self.nb_timesteps - 1)
        random_signal = np.zeros((len(self.ys), len(time_discretization)), 
                                 dtype = 'complex')
        signal_11 = np.zeros((len(self.ys), len(time_discretization)),
                             dtype = 'complex')
        signal_21 = 1*signal_11
#        freqs = np.fft.fftfreq(self.nb_timesteps, d=self.tau)
        for k in trange(len(self.ys), desc = 'Computing signals', 
                        disable = not self.describe) :
            random_signal[k, :] = fft(generate_n_gp(time_discretization,
                         return_fourier = False))
            signal_11[k, :] = fft(np.array([(self.G(t,x1,self.ys[k])) 
            for t in time_discretization]))
            signal_21[k, :] = fft(np.array([(self.G(t,x2,self.ys[k])) 
            for t in time_discretization]))
            
        
        signal_1 = signal_11*random_signal
        signal_2 = signal_21*random_signal
        signal_1 = np.real(ifft(np.sum(signal_1,axis = 0)))
        signal_2 = np.real(ifft(np.sum(signal_2,axis = 0)))
           
        return 1/ np.sqrt(self.N) * \
            signal_1, 1/ np.sqrt(self.N) * \
            signal_2, time_discretization, self.ys


if __name__ == '__main__':
    env = Environnement(N=10**3, nb_timesteps= 500, c0 = 1, L = 50)
    X = np.zeros((5,2))
    X[:,0] = -15 + 5*np.linspace(1,5,5,dtype = 'int')
    x1, x2, x3, x4, x5 = X[0], X[1], X[2], X[3], X[4]
    u1, u2, time, _ = env.compute_signal(x1, x2, 100)
    import pylab as plt
    plt.plot(time, u1, color='red', label='u1')
    plt.plot(time, u2, color='blue', label='u2')
    plt.legend()
    plt.show()