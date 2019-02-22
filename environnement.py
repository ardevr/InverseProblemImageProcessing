#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 16:12:47 2019

@author: evrardgarcelon
"""

import numpy as np
from scipy.special import j0,y0
from numpy.fft import fft,ifft
from gaussian_process import GaussianProcess

class Environnement():
    
    def __init__(self, c0 = 1, L = 50, N = 10, nb_timesteps = 100) :
        
        self.L = L
        self.N = N
        self.c0 = c0
        self.G_hat = lambda w,x,y : 1j/4*(np.nan_to_num(j0(w*np.linalg.norm(x-y))) + 1j*np.nan_to_num(y0(w*np.linalg.norm(x-y))))
        self.G = lambda t,x,y : 0 # Trouver valeur G
        self.nb_timesteps = nb_timesteps
        
        
    def compute_signal(self,x1, x2, tau, T,eps = 10**-10) :
        time_discretization = np.linspace(eps,T,self.nb_timesteps)
        thetas = np.random.uniform(size = self.N)
        y = np.zeros((self.N,2))
        y[:,0] = np.cos(2*np.pi*thetas)
        y[:,1] = np.sin(2*np.pi*thetas)
        cov = lambda x : np.exp(-x**2/4)*(2 - x**2)/(4*np.sqrt(2)) 
        gp = GaussianProcess(cov)
        signal_x1 = 0
        signal_x2 = 0
        for i in range(self.N) :
            temp_gp = gp.generate_n_var(self.nb_timesteps, time_discretization, fourier = True)
            temp_G1 = fft(self.G_hat(time_discretization,x1,y[i]))
            temp_G2 = fft(self.G_hat(time_discretization + tau,x2,y[i]))
            signal_x1 += np.real(ifft(temp_G1*temp_gp))
            signal_x2 += np.real(ifft(temp_G2*temp_gp))
        return self.L/np.sqrt(self.N)*signal_x1,self.L/np.sqrt(self.N)*signal_x2
    
            
        
        
        
        
        
        
        
        
        
        
        
        