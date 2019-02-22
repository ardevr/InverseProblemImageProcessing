#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 17:27:27 2019

@author: evrardgarcelon
"""

import numpy as np 
from numpy.fft import fft,ifft

class GaussianProcess():
    
    def __init__(self,cov = None) :
        self.cov = cov
        
    def generate_n_var(self,n,points, fourier = False) :
        
        assert isinstance(points,np.ndarray)
        assert n == len(points)

        cov_vector = np.zeros(n)
        cov_vector = self.cov(points)
        y = np.random.randn(n)
        z_fourier = np.sqrt(fft(cov_vector))*fft(y)
        if fourier :
            return z_fourier
        else : 
            return np.real(ifft(z_fourier))
    
