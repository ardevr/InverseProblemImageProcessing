#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 16:12:47 2019

@author: evrardgarcelon
"""

import numpy as np
from scipy.special import j0, y0
from numpy.fft import fft, ifft, fftfreq
from scipy.integrate import quad


def generate_n_gp(time_points, fourier_cov, eps=10**-5):
        
    assert isinstance(time_points, np.ndarray)
    
    n = len(time_points)
    y_fourier = fft(np.random.randn(n))
    delta_t = time_points[1] - time_points[0]
    freqs = fftfreq(n, d=delta_t)
    z_fourier = np.sqrt(fourier_cov(freqs)) * y_fourier
    return np.real(ifft(z_fourier))

def cplx_quad(f,a,b) :
    
    real_f = lambda x : np.real(f(x))
    
    return quad(real_f,a,b)[0] 


class Environnement():

    def __init__(self, c0=1, L=50, N=50, nb_timesteps=100, fourier_cov = lambda w : w**2*np.exp(-w**2)) :

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
            return 1j/4*(j0(w*norme) + 1j*y0(w*norme))
        
    def G(self,t, x, y) : 
            norm = np.linalg.norm(x - y)/self.c0
            if t > norm : 
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
    
    def solutions_one_source(self, t, y, x1, x2) :
        TT  = 10**3
        norme_1 = np.linalg.norm(y - x1)
        norme_2 = np.linalg.norm(y - x2)
        time_points = np.linspace(1, TT)
        points_1 = time_points - norme_1*time_points
        points_2 = time_points - norme_2*time_points
        points = np.concatenate(points_1)
        gps = generate_n_gp(t - time_points, self.fourier_cov)
        
        sol_1 = 0
        sol_2 = 0
        for i in range(len(time_points)) :
            sol_1 += self.G(time_points[i],x1,y)*gps[i]
            sol_2 += self.G(time_points[i],x2,y)*gps[i]
        
        print(sol_1)
        return  sol_1, sol_2

    def compute_signal(self, x1, x2, T, tau):

        time_discretization = np.linspace(0, T, self.nb_timesteps)
        signal_x1 = 0*time_discretization
        signal_x2 = 0*time_discretization
        for i in range(len(time_discretization)) :
            t = time_discretization[i]
            for j in range(len(self.ys)):
                temp1,temp2 = self.solutions_one_source(t, self.ys[j], x1, x2)
                signal_x1[i] += temp1
                signal_x2[i] += temp2
                
        return self.c0/ np.sqrt(self.N) * \
            signal_x1, self.c0/ np.sqrt(self.N) * \
            signal_x2, time_discretization, self.ys


if __name__ == '__main__':
    env = Environnement(N=1, nb_timesteps= 111, c0 = 1, L = 5)
    x1 = x2 = np.ones(2)
    x2 = x2
    tau = 0
    u1, u2, time, _ = env.compute_signal(x1, x2, 100, tau)
    import pylab as plt
    plt.scatter(time, u1, color='red', label='u1')
    plt.scatter(time, u2, color='blue', label='u2')
    plt.legend()
