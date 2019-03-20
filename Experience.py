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
# from tqdm import tqdm
from scipy.special import j0, y0, hankel1
from tqdm import trange
# from scipy.stats import randint
from numpy.fft import fft, ifft
# from scipy.signal import fftconvolve
# from numpy import correlate

# import warnings


def emp_cross_corr(u_1, u_2, times, verbose=True) :

    if verbose:
        print('Computing emp_cross_corr ...')
    M = len(u1)
    u1_tilde = np.zeros(2*M)
    u2_tilde = np.zeros(2*M)
    u1_tilde[M:] = u1
    u2_tilde[M:] = u2
    DFT_C = np.conjugate(fft(u1_tilde))*fft(u2_tilde)*((-1)**np.linspace(0,2*M-1,2*M))
    C_TN = ifft(DFT_C)/M
    C_TN = np.real(C_TN)
    if verbose:
        print('Done')
    return C_TN

def exp_emp_cross_cor(tau, x1, x2, N, env, verbose=True):
    if verbose:
        print('Computing exp_emp_cross_corr ...')
    ys = env.ys
    c0 = env.c0
    F_hat = lambda w : env.fourier_cov(w)
    temp_cor = 0
    i = 0
    while i < (len(ys)) :
        y = ys[i]
        norme_1 = np.linalg.norm(x1 - y)/c0
        norme_2 = np.linalg.norm(x2 - y)/c0
        a, b = -np.inf, +np.inf
        inte_1,_ = quad(lambda w : F_hat(w)*np.imag(hankel1(0,w*norme_1))*np.imag(hankel1(0,w*norme_2))*np.cos(w*tau),a,b, limit = 100)
        inte_2,_ = quad(lambda w : F_hat(w)*np.imag(hankel1(0,w*norme_2))*np.real(hankel1(0,w*norme_1))*np.sin(w*tau),a,b, limit = 100)
        inte_3,_ = quad(lambda w : -F_hat(w)*np.imag(hankel1(0,w*norme_1))*np.real(hankel1(0,w*norme_2))*np.sin(w*tau),a,b, limit = 100)
        inte_4,_ = quad(lambda w : F_hat(w)*np.real(hankel1(0,w*norme_1))*np.real(hankel1(0,w*norme_2))*np.cos(w*tau),a,b, limit = 100)
        temp_result =  1/16*(inte_1 + inte_2 + inte_3 + inte_4)
        temp_cor += temp_result
        i = i+1
    if verbose:
        print('Done')
    return temp_cor/(2*np.pi*N)

def cross_1(tau, x1, x2, L, env, verbose=True):

    if verbose:
        print('Computing Cross_1 ...')
    F_hat = lambda w : env.fourier_cov(w)
    c0 = env.c0
    def f(theta):
        y = L*np.array([np.cos(theta),np.sin(theta)])
        norme_1 = np.linalg.norm(x1 - y)/c0
        norme_2 = np.linalg.norm(x2 - y)/c0
        a, b = 0, +np.inf
        inte_1,_ = quad(lambda w : F_hat(w)*np.imag(hankel1(0,w*norme_1))*np.imag(hankel1(0,w*norme_2))*np.cos(w*tau),a,b, limit = 100)
        inte_2,_ = quad(lambda w : F_hat(w)*np.imag(hankel1(0,w*norme_2))*np.real(hankel1(0,w*norme_1))*np.sin(w*tau),a,b, limit = 100)
        inte_3,_ = quad(lambda w : -F_hat(w)*np.imag(hankel1(0,w*norme_1))*np.real(hankel1(0,w*norme_2))*np.sin(w*tau),a,b, limit = 100)
        inte_4,_ = quad(lambda w : F_hat(w)*np.real(hankel1(0,w*norme_1))*np.real(hankel1(0,w*norme_2))*np.cos(w*tau),a,b, limit = 100)
        return  1/8*(inte_1 + inte_2 + inte_3 + inte_4)
 #       f = lambda w : np.real(F_hat(w)*np.conjugate(1j/4*hankel1(0,w*norme_1))
 #       *1j/4*hankel1(0,w*norme_2)*np.exp(-1j*tau*w))
 #       temp_res, err = quad(f,a,b)
 #      return temp_res
    
    b = 2 * np.pi
    a = 0
    result,_ = quad(f, a, b)
    if verbose:
        print('Done')
    return result / (2 * np.pi**2)

def C_asy(tau, x1, x2,  env, verbose=True, eps = 10**-10):
    if verbose:
        print('Computing C_asy...')

    norme = np.linalg.norm(x1-x2)/env.c0
    
    if norme > eps :
        def f(w): return env.fourier_cov(w) * np.real(hankel1(0,w*norme)) * np.cos(w*tau) / w
        result = env.c0*quad(f, a=-np.inf, b=+np.inf)[0]
    else :
        F_hat = lambda w : env.fourier_cov(w)
        def f(theta):
            y = L*np.array([np.cos(theta),np.sin(theta)])
            a, b = 0, +np.inf
            inte, _ = quad(lambda w : np.cos(w*tau)*F_hat(w)*np.linalg.norm(env.G_hat(w,x1,y))**2, a, b)
            return  1/8*inte
        a, b = 0, 2*np.pi
        result, _ = quad(f, a, b)
        result = result*np.pi*L
    if verbose:
        print('Done')
    return result/(4*np.pi**2)
        
    
def plot_exp(env,X) :
    plt.figure(50)
    plt.scatter(X[:,0],X[:,1],marker = '^', color = 'red', label = 'x')
    y = env.ys
    plt.scatter(y[:,0],y[:,1],marker = 'o', color = 'green', label = 'sources')
    plt.legend()
    plt.show()

def estimated_c0(env,x1,x2,C_TN,tau):
    return((x2[0]-x1[0])/(abs(C_TN.argmax()-env.nb_timesteps)*tau))
    
def plot_c_asy(x1, x2, env, low_tau = 1, high_tau = 50, nb_steps = 100) :
    taus = np.linspace(low_tau, high_tau, nb_steps)
    result = np.zeros(len(taus))
    for i in trange(len(taus), desc = 'Computing C_asy') :
        tau = taus[i]
        result[i] = C_asy(tau, x1, x2, env, verbose = False)
    return result

def plot_c_1(x1, x2, env, low_tau = 1, high_tau = 50, nb_steps = 100) :
    taus = np.linspace(low_tau, high_tau, nb_steps)
    result = np.zeros(len(taus))
    for i in trange(len(taus), desc = 'Computing C_1') :
        tau = taus[i]
        result[i] = cross_1(tau, x1, x2, env.L, env, verbose=False)
    return result

def plot_exp_emp_cross(x1, x2, env, low_tau = 1, high_tau = 50, nb_steps = 100):
    taus = np.linspace(low_tau, high_tau, nb_steps)
    result = np.zeros(len(taus))
    for i in trange(len(taus), desc = 'Computing exp_emp_cross') :
        tau = taus[i]
        result[i] = exp_emp_cross_cor(tau, x1, x2, env.N, env, verbose=False)
    return result


if __name__ == '__main__':
        
    L, N, T = 50, 100, 100
    X = np.zeros((5,2))
    X[:,0] = -15 + 5*np.linspace(1,5,5,dtype = 'int')
    x1, x2, x3, x4, x5 = X[0], X[1], X[2], X[3], X[4]
    fourier_cov = lambda w : w**2*np.exp(-w**2)
    #%%
    env = Environnement(nb_timesteps = 7000, L = L, N = N, fourier_cov= fourier_cov)
    u1,u2,times,_ = env.compute_signal(x1, x2, T)
    C_TN = emp_cross_corr(u1, u2, times, verbose=False)  
    
    nb_steps = 100
    result_asy = plot_c_asy(x1,x3, env, low_tau = times[0], high_tau = times[-1], nb_steps= nb_steps)
    result_1 = plot_c_1(x1,x3, env, low_tau = times[0], high_tau = times[-1], nb_steps= nb_steps)
    result_2 = plot_exp_emp_cross(x1,x3, env, low_tau = times[0], high_tau = times[-1], nb_steps= nb_steps)
    taus = np.linspace(times[0], times[-1],nb_steps)
    plt.figure(1)
    plt.plot(taus, 2*L*result_1, linewidth = 1.5, color = 'blue', label = 'C_1', marker = '+')
    plt.plot(taus, 2*L*result_2, linewidth = 1.5, color = 'green', label = 'exp_emp_cross', marker = 'o')
    plt.plot(taus, result_asy, linewidth = 1.5, color = 'red', label = 'C_asy', marker = '^')
    plt.plot(times, 2*L*C_TN[len(u1):], linewidth = 1.5, color = 'magenta', label = 'Empirical Cross correlation')
    plt.plot()
    plt.legend()
    plt.show()
    
    #%%
    
    nb_repetitions = 10
    nb_timesteps = 5000
    high_N, low_N = 150, 30
    high_T, low_T = 500, 100
    Ns, Ts = np.linspace(low_N, high_N, 20, dtype = 'int'), np.linspace(low_T, high_T, 20, dtype = 'int')
    tau_max = np.linalg.norm(x1-x3)
    random_index_N = np.random.choice(
            np.linspace(0,len(Ns)-1,len(Ns),dtype = 'int'))
    random_N = Ns[random_index_N]

    C_TNs = np.zeros((nb_repetitions, len(Ts)))
    
    for i in range(nb_repetitions) :
        for j in range(len(Ts)) :
                N,T = 1*random_N, Ts[j]
                delta_t = T/(nb_timesteps-1)
                index_tau_max = int(tau_max/delta_t)
                env = Environnement(nb_timesteps = nb_timesteps, L = L, N = N, 
                                    fourier_cov= fourier_cov)
                u1,u2,times,_ = env.compute_signal(x1, x3, T)
                C_TN = emp_cross_corr(u1, u2, times, verbose=False) 
                C_TN = C_TN[len(u1)+ index_tau_max]
                C_TNs[i,j] = C_TN
    mean_C = np.mean(C_TNs, axis = 0)
    q = 0.1
    q_bound = np.quantile(C_TNs, q, axis = 0)
    Q_bound = np.quantile(C_TNs, 1-q, axis = 0)
    plt.figure(2)
    plt.fill_between(Ts, q_bound, Q_bound, alpha=0.15, linewidth=1.5, 
                     color='red')
    plt.plot(Ts, mean_C, linewidth = 1.5, color ='red', marker = '^')
    
    random_index_T = np.random.choice(
    np.linspace(0,len(Ts)-1,len(Ts),dtype = 'int'))
    random_T = Ts[random_index_T]
    C_TNs = np.zeros((nb_repetitions, len(Ts)))
    
    for i in range(nb_repetitions) :
        for j in range(len(Ns)) :
                N,T = Ns[j], 1*random_T
                delta_t = T/(nb_timesteps-1)
                index_tau_max = int(tau_max/delta_t)
                env = Environnement(nb_timesteps = nb_timesteps, L = L, N = N, 
                                    fourier_cov= fourier_cov)
                u1,u2,times,_ = env.compute_signal(x1, x3, T)
                C_TN = emp_cross_corr(u1, u2, times, verbose=False) 
                C_TN = C_TN[len(u1)+ index_tau_max]
                C_TNs[i,j] = C_TN
    mean_C = np.mean(C_TNs, axis = 0)
    q_bound = np.quantile(C_TNs, q, axis = 0)
    Q_bound = np.quantile(C_TNs, 1-q, axis = 0)
    plt.figure(3)
    plt.fill_between(Ns, q_bound, Q_bound, alpha=0.15, linewidth=1.5, color='green')
    plt.plot(Ns, mean_C, linewidth = 1.5, color ='green', marker = 'o')
    plt.show()
    
    #%%
    L, N, T = 50, 100, 250
    nb_steps = 50
    fourier_cov = lambda w : w**2*np.exp(-w**2)
    env = Environnement(nb_timesteps = 7000, L = L, N = N, 
                        fourier_cov= fourier_cov)
    u1,u2,times,_ = env.compute_signal(x1, x1, T)
    result_autocor = plot_c_asy(x1,x1,env,high_tau=T, low_tau = 10**-4, 
                                nb_steps= nb_steps)
    C_TN = emp_cross_corr(u1, u2, times, verbose=False) 
    plt.figure(4)
    taus = np.linspace(10**-4,T, nb_steps)
    plt.plot(taus, result_autocor, label = 'C_asy autocorr', 
             linewidth = 1.5, color = 'red', marker = '^')
    plt.plot(times, C_TN[len(u1):], linewidth = 1.5, color = 'blue', 
             label = 'Empirical cross correlation')
    plt.legend()
    plt.show()
    
    
    #%%
    
    # Evaluation of the error
    L, N, T = 50, 100, 250
    nb_steps = 50
    fourier_cov = lambda w : w**2*np.exp(-w**2)
    speed = np.linspace(0.1,5,10)
    erreur = []
    for i in trange(len(speed), desc = 'Computing c0'):
        c = speed[i]
        env = Environnement(nb_timesteps = 3000, L = L, N = N, 
                        fourier_cov= fourier_cov, c0 = c)
        u1,u2,times,_ = env.compute_signal(x1, x3, T)
        C_TN = emp_cross_corr(u1, u2, times, verbose = False)[len(u1):]
        argmax_index = np.argmax(C_TN)
        time_estimator = times[argmax_index]
        
        estimator = np.linalg.norm(x1-x3)/time_estimator
        print('estimator : ', estimator)
        erreur.append(np.abs(estimator - c))
    erreur = np.array(erreur)
    # pour c0 > 5 l'erreur est très grande car l'écart mesurable est inférieur à la distance
    plt.plot(speed,erreur)
    plt.title("Error evaluation on the speed estimation with the distance/time ratio")
    plt.xlabel("Speed c0")
    plt.ylabel("Error")
    plt.show()
#%%
    
                
                
    
