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
from scipy.special import j0, y0
from scipy.stats import randint
from numpy.fft import fft, ifft


import warnings


def emp_cross_corr(u_1, u_2, times, verbose=True) :
    
    if verbose:
        print('Computing emp_cross_corr ...')
    M = len (u_1)
    u1_tilde = np.zeros(2*M-1)
    u1_tilde[M-1:] = u_1
    u2_tilde = np.zeros(2*M-1)
    u2_tilde[M-1:] = u_2
    DFT_CT = (fft(u2_tilde)*np.conjugate(fft(u1_tilde)))*(-1)**np.linspace(0,2*M-2,2*M-1,dtype = 'int')
    
    
    if verbose:
        print('Done')
    return np.real(ifft(DFT_CT)),ifft(DFT_CT)
    #return np.sum(u1[:-1]*u2[:-1]*(times[1:] - times[:-1]))/T, 1

def exp_emp_cross_cor(tau, x1, x2, N, env, verbose=True):
    if verbose:
        print('Computing exp_emp_cross_corr ...')
    assert N == env.N
    ys = env.ys
    c0 = env.c0
    F_hat = lambda w : env.fourier_cov(w)
    if not isinstance(ys, np.ndarray):
        ys = np.array(ys)
    temp_cor = 0
    for y in  ys :
        norme_1 = np.linalg.norm(x1 - y)/c0
        norme_2 = np.linalg.norm(x2 - y)/c0
        a, b = 0, +np.inf
        inte_1,_ = quad(lambda w : F_hat(w)*y0(w*norme_1)*y0(w*norme_2)*np.cos(w*tau),a,b, limit = 100)
        inte_2,_ = quad(lambda w : -F_hat(w)*y0(w*norme_2)*j0(w*norme_1)*np.sin(w*tau),a,b, limit = 100)
        inte_3,_ = quad(lambda w : F_hat(w)*y0(w*norme_1)*j0(w*norme_2)*np.sin(w*tau),a,b, limit = 100)
        inte_4,_ = quad(lambda w : F_hat(w)*j0(w*norme_1)*j0(w*norme_2)*np.cos(w*tau),a,b, limit = 100)
        temp_cor += 1/8*(inte_1 + inte_2 + inte_3 + inte_4)
    if verbose:
        print('Done')
    return temp_cor/N


def C_asy(tau, x1, x2,  env, verbose=True):
    if verbose:
        print('Computing C_asy...')

    norme = np.linalg.norm(x1-x2)/env.c0
    def f(w): return env.fourier_cov(w) * j0(w*norme) * np.cos(w*tau) / w
    

    result = env.c0*quad(f, a=-np.inf, b=+np.inf)[0] / (8*np.pi)

    if verbose:
        print('Done')
    return result


def cross_1(tau, x1, x2, L, env, verbose=True):

    if verbose:
        print('Computing Cross_1 ...')
    F_hat = lambda w : env.fourier_cov(w)
    def f(theta):
        y = L*np.array([np.cos(theta), np.sin(theta)])
        norme_1 = np.linalg.norm(x1 - y)/env.c0
        norme_2 = np.linalg.norm(x2 - y)/env.c0
        a, b = 0, +np.inf
        inte_1,_ = quad(lambda w : F_hat(w)*y0(w*norme_1)*y0(w*norme_2)*np.cos(w*tau),a,b, limit = 100)
        inte_2,_ = quad(lambda w : -F_hat(w)*y0(w*norme_2)*j0(w*norme_1)*np.sin(w*tau),a,b, limit = 100)
        inte_3,_ = quad(lambda w : F_hat(w)*y0(w*norme_1)*j0(w*norme_2)*np.sin(w*tau),a,b, limit = 100)
        inte_4,_ = quad(lambda w : F_hat(w)*j0(w*norme_1)*j0(w*norme_2)*np.cos(w*tau),a,b, limit = 100)
        return 1/8*(inte_1 + inte_2 + inte_3 + inte_4)
    b = 2 * np.pi
    a = 0
    result,error = quad(f, a, b)
    if verbose:
        print('Done')
    return result / (2 * np.pi)



if __name__ == '__main__':

    L, N, T = 50, 100, 10**3
    X = np.zeros((5,2))
    X[:,0] = -15 + 5*np.linspace(1,5,5,dtype = 'int')
    x1, x2 = X[0], X[1]/100
    tau = 0.1
    eps = 10**3
    fourier_cov = lambda w : np.exp(-w**2)
    env = Environnement(nb_timesteps = T/tau, L = L, N = N, fourier_cov= fourier_cov)
    u1,u2,times,_ = env.compute_signal(x1, x2, T, tau) 
    import time
    debut = time.clock()
    C_TN, C_TN2 = emp_cross_corr(u1, u2, times, verbose=True)
    fin = time.clock()
    print('Temps emp_cross_corr : ', fin - debut)
    debut = time.clock()
    exp_emp_corr = exp_emp_cross_cor(tau, x1, x2, N, env)
    fin = time.clock()
    print('Temps exp emp corr : ', fin - debut)
    debut = time.clock()
    c_1 = cross_1(tau, x1, x2, L, env)
    fin = time.clock()
    print('Temps cross_1 : ', fin - debut)
    debut = time.clock()
    c_asy = C_asy(tau, x1, x2, env)
    fin = time.clock()
    print('Temps C_asy : ', fin - debut)
#    env = Environnement(N = N, L = L, c0 = 1)
#    #tau = 0
#    C_TN = emp_cross_corr(tau, x1, x2, T, N, env)
#    estimated_c0 = 8*np.pi**(3/2)*C_TN
    
#    with warnings.catch_warnings():
#        warnings.filterwarnings("ignore",category=RuntimeWarning)
#        # Question 1
#        L, N, T = 50, 50, 10**3
#        x_to_test = []
#        X = np.zeros((5,2))
#        X[:,0] = -15 + 5*np.linspace(1,5,5,dtype = 'int')
#        experience ={}
#        for j in tqdm(range(len(X))) :
#            for i in range(j + 1, len(X)) :
#                taus = np.linspace(1,6, 10)
#                env = Environnement(N = N, L = L)
#                x1,x2 = X[j], X[i]
#                emp_cross_corr1 = exp_emp_cross_cor1 = cross_corr_1 = cross_asy = []
#                for tau in taus :
#                    emp_cross_corr1.append(emp_cross_corr(tau, x1, x2, T, N, env, verbose = True))
#                    exp_emp_cross_cor1.append(exp_emp_cross_cor(tau, x1, x2, N, env, verbose = True))
#                    cross_corr_1.append(cross_1(tau, x1, x2, L, env, verbose = True))
#                    cross_asy.append(C_asy(tau, x1, x2,  env, verbose = True))
#                emp_cross_corr1 = np.array(emp_cross_corr1)
#                exp_emp_cross_cor1 = np.array(exp_emp_cross_cor1)
#                cross_corr_1 = np.array(cross_corr_1)
#                cross_asy = np.array(cross_asy)
#                temp_dict = {'emp_cross_corr1' : emp_cross_corr1,
#                             'exp_emp_cross_cor1' : exp_emp_cross_cor1,
#                             'cross_corr_1' : cross_corr_1,
#                             'cross_asy' : cross_asy}
#                experience[(j,i)] = temp_dict
#        j,i = (0,1)
#        values = experience[(j,i)]
#        colors = ['b', 'g', 'r','c','m','y','k']
#        for key, val in values.items() :
#            U = randint(low = 0, high = len(colors)).rvs()
#            plt.plot(val, label= key, color = colors[U])
#        plt.legend()
#        plt.show()
#
#        # Question 2
#        
#        Ns,Ts = np.linspace(10,100), np.linspace(100,10**4)
#        nb_repetitions = 20
#        c_tns = np.zeros((len(Ns), len(Ts), nb_repetitions))
#        tau = 1
#        x1, x2 = X[0], X[1]
#        env = Environnement(N = N, L = L)
#        for i in range(len(Ns)) :
#            for j in range(len(Ts)) :
#                for k in range(nb_repetitions) :
#                    c_tns[i,j,k] = emp_cross_corr(tau, x1, x2, Ts[j], Ns[i], env)

#        plt.semilogx(tsav, meanRegret, color = colors[k])
#        plt.fill_between(tsav, qRegret, QRegret, alpha=0.15, linewidth=1.5, color=colors[k])
#        plt.xlabel('Time')
#        plt.ylabel('Regret')``

        # Question 3
        
        # Question 4 
        
       
        
        
#
#     TODO ecrire estimation de c0
