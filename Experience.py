#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 09:21:30 2019

@author: evrardgarcelon
"""

import numpy as np

from environnement import Environnement
from scipy.integrate import quad
import pylab as plt
from tqdm import tqdm
from scipy.special import j0
from scipy.stats import randint


import warnings



def complex_integration(f, a, b, eps=10**-5):
    def real_f(x): return np.real(f(x))

    def imag_f(x): return np.imag(f(x))

    real_part, err_real_part = quad(real_f, a, b)
    imag_part, err_imag_part = quad(imag_f, a, b)

    if np.abs(err_imag_part) + np.abs(err_real_part) < eps:
        return real_part + imag_part
    else:
        raise Exception('Integral not computed')


F_hat = lambda w : w**2*np.exp(-w**2)

def emp_cross_corr(tau, x1, x2, T, N, env, verbose = True) :
    if verbose : 
        print('Computing emp_cross_corr ...')
    u1, u2_lagged, dicretization, _ = env.compute_signal(x1, x2, tau, T, N)
    integral = 0
    for i in range(len(dicretization) - 1) :
        integral += u1[i]*u2_lagged[i]*(dicretization[i+1] - dicretization[i])/T
    if verbose : 
        print('Done')
    return integral

def exp_emp_cross_cor(tau, x1, x2, N, env, verbose = True) : 
    if verbose : 
        print('Computing exp_emp_cross_corr ...')
    assert N == env.N
    y = env.y
    temp_cor = 0
    for s in range(len(y)) :
        temp_g = lambda w : np.conjugate(env.G_hat(w, x1, y[s]))*env.G_hat(w, y[s], x2)*np.exp(-1j*w*tau)*F_hat(w)
        temp_cor += complex_integration(temp_g, a = -np.inf, b = + np.inf)/(2*np.pi*N)
    if verbose : 
        print('Done')
    return temp_cor

def cross_1(tau, x1, x2, L, env, verbose = True) :
        if verbose : 
            print('Computing C_(1)...')
            
        def G_hat(w): return env.G_hat(w, x1, x2)

        def f(w): return F_hat(w)*np.imag(G_hat(w))*np.exp(-1j*tau*w)/w
        
        result = complex_integration(f, a=-np.inf, b=+np.inf)/(4*np.pi**2*L)
        if verbose : 
            print('Done')
        return result

def c_asy(tau, x1, x2, L, env, c0 = 1, verbose = True) :
        if verbose : 
            print('Computing C_asy ...') 
        norm_x = np.linalg.norm(x1-x2)
        f = lambda w : w*np.cos(w*tau)*np.exp(-w**2)*j0(norm_x*w)
        temp_C_asy = - c0 / (8 * np.pi**2) * complex_integration(f, a= -np.inf, b=np.inf)
        if verbose :
            print('Done')
        return temp_C_asy



    
if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=RuntimeWarning)
        L, N, T = 50, 50, 10**3
        x_to_test = []
        X = np.zeros((5,2))
        X[:,0] = -15 + 5*np.linspace(1,5,5,dtype = 'int')
        experience ={}
        for j in tqdm(range(1)) :
            for i in range(j + 1, j+2) :
                taus = np.linspace(1,6, 10)
                env = Environnement(N = N, L = L)
                x1,x2 = X[j], X[i]
                emp_cross_corr1 = exp_emp_cross_cor1 = cross_corr_1 = cross_asy = []
                for tau in taus :
                    emp_cross_corr1.append(emp_cross_corr(tau, x1, x2, T, N, env, verbose = False))
                    exp_emp_cross_cor1.append(exp_emp_cross_cor(tau, x1, x2, N, env, verbose = False))
                    cross_corr_1.append(cross_1(tau, x1, x2, L, env, verbose = False))
                    cross_asy.append(c_asy(tau, x1, x2, L,env, verbose = False))
                emp_cross_corr1 = np.array(emp_cross_corr1)
                exp_emp_cross_cor1 = np.array(exp_emp_cross_cor1)
                cross_corr_1 = np.array(cross_corr_1)
                cross_asy = np.array(cross_asy)
                temp_dict = {'emp_cross_corr1' : emp_cross_corr1,
                             'exp_emp_cross_cor1' : exp_emp_cross_cor1,
                             'cross_corr_1' : cross_corr_1,
                             'cross_asy' : cross_asy}
                experience[(j,i)] = temp_dict
        j,i = (0,1)
        values = experience[(j,i)]
        colors = ['b', 'g', 'r','c','m','y','k']
        for key, val in values.items() :
            U = randint(low = 0, high = len(colors)).rvs()
            plt.plot(val, label= key, color = colors[U])
        plt.legend()
        plt.show()       
        
        
        Ns,Ts = np.linspace(10,100), np.linspace(100,10**4)
        c_tns = np.zeros((len(Ns), len(Ts)))
        tau = 1
        x1, x2 = X[0], X[1]
        for i,j in zip(Ns,Ts) :
            c_tns[i,j] = emp_cross_corr(tau, x1, x2, Ts[j], Ns[i])
            
        # TODO ecrire estimation de c0
            
            
                
                
        
