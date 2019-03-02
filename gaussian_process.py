#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 17:27:27 2019

@author: evrardgarcelon
"""

import numpy as np 
from numpy.fft import fft,ifft

class GaussianProcess():
    
    def __init__(self,cov = None, seed = 1) :
        self.cov = cov
        self.seed = seed
        
    def generate_n_var(self,n,points, fourier = False, eps = 10**-5) :
        
        assert isinstance(points,np.ndarray)
        assert n == len(points)

        cov_vector = np.zeros(n)
        cov_vector = self.cov(points)
        np.random.seed = self.seed
        y = np.random.randn(n)
        z_fourier = np.sqrt(fft(cov_vector))*fft(y)
        if fourier :
            return z_fourier
        else : 
            temp_vec = ifft(z_fourier)
            if np.sum(np.abs(np.imag(temp_vec))) < eps :
                return np.real(ifft(z_fourier))
            else : 
                raise Exception('Inverse Fourier Transform not real')
    
