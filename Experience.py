#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 09:21:30 2019

@author: evrardgarcelon
"""
#%%
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


def emp_cross_corr(u_1, u_2, times, verbose=True, all_data = False) :

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
    if not all_data :
        return C_TN[M:]
    else :
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
        a, b = 0, 5
        inte_1,_ = quad(lambda w : F_hat(w)*y0(w*norme_1)*y0(w*norme_2)*np.cos(w*tau),a,b, limit = 300)
        inte_2,_ = quad(lambda w : F_hat(w)*y0(w*norme_2)*j0(w*norme_1)*np.sin(w*tau),a,b, limit = 300)
        inte_3,_ = quad(lambda w : -F_hat(w)*y0(w*norme_1)*j0(w*norme_2)*np.sin(w*tau),a,b, limit = 300)
        inte_4,_ = quad(lambda w : F_hat(w)*j0(w*norme_1)*j0(w*norme_2)*np.cos(w*tau),a,b, limit = 300)
        temp_result = 1/8*(inte_1 + inte_2 + inte_3 + inte_4)
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
        a, b = 0, 5
        inte_1,_ = quad(lambda w : F_hat(w)*y0(w*norme_1)*y0(w*norme_2)*np.cos(w*tau),a,b, limit = 300)
        inte_2,_ = quad(lambda w : F_hat(w)*y0(w*norme_2)*j0(w*norme_1)*np.sin(w*tau),a,b, limit = 300)
        inte_3,_ = quad(lambda w : -F_hat(w)*y0(w*norme_1)*j0(w*norme_2)*np.sin(w*tau),a,b, limit = 300)
        inte_4,_ = quad(lambda w : F_hat(w)*j0(w*norme_1)*j0(w*norme_2)*np.cos(w*tau),a,b, limit = 300)
        return  1/8*(inte_1 + inte_2 + inte_3 + inte_4)
#        f = lambda w : np.real(F_hat(w)*np.conjugate(1j/4*hankel1(0,w*norme_1))
#        *1j/4*hankel1(0,w*norme_2)*np.exp(-1j*tau*w))
        temp_res, err = quad(f,a,b, limit = 300)
        return temp_res
    
    b = 2 * np.pi
    a = 0
    result,_ = quad(f, a, b, limit = 300)
    if verbose:
        print('Done')
    return result / (2 * np.pi**2)

def C_asy(tau, x1, x2,  env, verbose=True, eps = 10**-10):
    if verbose:
        print('Computing C_asy...')

    norme = np.linalg.norm(x1-x2)/env.c0
    
    if norme > eps :
        def f(w): return env.fourier_cov(w) * j0(w*norme) * np.cos(w*tau) / w
        result = env.c0*quad(f, a=0, b=+np.inf)[0]
        result = 2*result
    else :
        F_hat = lambda w : env.fourier_cov(w)
        def f(theta):
            y = L*np.array([np.cos(theta),np.sin(theta)])
            norme = np.linalg.norm(x1-y)/env.c0
            a, b = 0, +np.inf
            inte, _ = quad(lambda w : np.cos(w*tau)*F_hat(w)*(j0(w*norme)**2 + np.nan_to_num(y0(w*norme))**2), a, b, limit = 300)
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

def estimated_c0(times,x1,x2,C_TN,tau):
    time_estimator = times[np.argmax(C_TN)]
    return(np.linalg.norm(x1-x2)/(time_estimator+1/2))
    
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
#%%

if __name__ == '__main__':

# Initialisation of the main environnement
    L, N, T = 50,300, 10**3
    X = np.zeros((5,2))
    X[:,0] = -15 + 5*np.linspace(1,5,5,dtype = 'int')
    x1, x2, x3, x4, x5 = X[0], X[1], X[2], X[3], X[4]
    fourier_cov = lambda w : w**2*np.exp(-w**2)
#%%

# Computation of C_TN, c_1 , c_N and C_asy
    C_TNs = np.zeros((10, 8000))
    for k in range(10) :
        env = Environnement(nb_timesteps = 8000, L = L, N = N, fourier_cov= fourier_cov)
        u1,u2,times,_ = env.compute_signal(x1, x3, T)
        C_TNs[k,:] = emp_cross_corr(u1, u2, times, verbose=True) 
    mean_c = np.mean(C_TNs, axis = 0)
    nb_steps_asy = 500
    nb_steps_c1 = 100
    nb_steps_exp = 100
    h_tau = 20
    l_tau = times[0]
    result_asy = plot_c_asy(x1,x3, env, low_tau = l_tau, high_tau = h_tau , nb_steps= nb_steps_asy)
#    result_1 = plot_c_1(x1,x3, env, low_tau = l_tau, high_tau = h_tau, nb_steps= nb_steps_c1)
#    result_2 = plot_exp_emp_cross(x1,x3, env, low_tau = l_tau, high_tau = h_tau, nb_steps= nb_steps_exp)
  #%%  
#    plt.figure(1)
    taus_asy = np.linspace(l_tau, h_tau,nb_steps_asy)
    taus_c1 = np.linspace(l_tau, h_tau,nb_steps_c1)
    taus_emp = np.linspace(l_tau, h_tau,nb_steps_exp)
#    plt.plot(taus_c1, 2*L*result_1, linewidth = 1.5, color = 'blue', label = 'C_1', marker = '+')
#    plt.plot(taus_emp, 4*L*result_2, linewidth = 1.5, color = 'green', label = 'exp_emp_cross', marker = 'o')
#    plt.plot(taus_asy, result_asy, linewidth = 1.5, color = 'red', label = 'C_asy', marker = '^')
#    plt.legend()
    
    plt.figure(2)
    tau = times[-1]/(len(times)-1)
    high_stop_limit = int(h_tau/tau)+1
    #C_TN_1 = C_TN[:high_stop_limit]
    C_TN_1 = mean_c[:high_stop_limit]
    taus = np.linspace(0,h_tau, high_stop_limit)
    plt.plot(taus, C_TN_1/1.5, linewidth = 1.5, color = 'blue', 
             label = 'Empirical cross correlation', marker = 'o')
    plt.plot(taus_asy, result_asy, linewidth = 1.5, color = 'red', label = 'C_asy', marker = '^')
    plt.plot()
    plt.legend()
    plt.show()

    nb_repetitions = 10
    nb_timesteps = 5000
    high_N, low_N = 300, 100
    high_T, low_T = 1000, 250
    Ns, Ts = np.linspace(low_N, high_N, 10, dtype = 'int'), np.linspace(low_T, high_T, 20, dtype = 'int')
    tau_max = np.linalg.norm(x1-x3) - 1
    random_index_N = len(Ns)-1
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
                C_TN = C_TN[index_tau_max]
                C_TNs[i,j] = C_TN
    mean_C = np.mean(C_TNs, axis = 0)
    q = 0.1
    q_bound = np.quantile(C_TNs, q, axis = 0)
    Q_bound = np.quantile(C_TNs, 1-q, axis = 0)
    plt.figure(2)
    plt.fill_between(Ts, q_bound, Q_bound, alpha=0.15, linewidth=1.5, 
                     color='red')
    plt.plot(Ts, mean_C, linewidth = 1.5, color ='red', marker = '^', label = 'C_TN, N = {}'.format(random_N))
    plt.xlabel('T')
#%%    
# Statistical stability of C_TN on N ans T
    random_index_T = len(Ts)-1
    random_T = Ts[random_index_T]
    C_TNs = np.zeros((nb_repetitions, len(Ns)))
    
    for i in range(nb_repetitions) :
        for j in range(len(Ns)) :
                N,T = Ns[j], 1*random_T
                delta_t = T/(nb_timesteps-1)
                index_tau_max = int(tau_max/delta_t)
                env = Environnement(nb_timesteps = nb_timesteps, L = L, N = N, 
                                    fourier_cov= fourier_cov)
                u1,u2,times,_ = env.compute_signal(x1, x3, T)
                C_TN = emp_cross_corr(u1, u2, times, verbose=False) 
                C_TN = C_TN[index_tau_max]
                C_TNs[i,j] = C_TN
    mean_C = np.mean(C_TNs, axis = 0)
    q_bound = np.quantile(C_TNs, q, axis = 0)
    Q_bound = np.quantile(C_TNs, 1-q, axis = 0)
    plt.figure(3)
    plt.fill_between(Ns, q_bound, Q_bound, alpha=0.15, linewidth=1.5, color='green')
    plt.plot(Ns, mean_C, linewidth = 1.5, color ='green', marker = 'o', label = 'C_TN, T = {}'.format(T))
    plt.xlabel('N')
    plt.show()
    
    #%%
# Autocorrelation function
    L, N, T = 50, 300, 1000
    h_tau = 20
    l_tau = -20
    nb_steps = 100
    fourier_cov = lambda w : w**2*np.exp(-w**2)
    env = Environnement(nb_timesteps = 7000, L = L, N = N, 
                        fourier_cov= fourier_cov)
    #u1,u2,times,_ = env.compute_signal(x1, x1, T)
    result_autocor = plot_c_asy(x1,x1,env,high_tau=h_tau, low_tau = l_tau, 
                                nb_steps= nb_steps)
    #C_TN = emp_cross_corr(u1, u2, times, verbose=False, all_data = True) 
    plt.figure(4)
    taus = np.linspace(l_tau,h_tau, nb_steps)
    plt.plot(taus, result_autocor, label = 'C_asy autocorr', 
             linewidth = 1.5, color = 'red', marker = '^')
    f = lambda t : (1 - 2*(t/2.1)**2)*np.exp(-(t/2.1)**2)/(6*8)
    plt.plot(taus, f(taus), linewidth = 1, color = 'blue')
    #tau = times[-1]/(len(times)-1)
    #high_stop_limit = int(h_tau/tau)+1
    #low_stop_limit = int(np.abs(l_tau)/tau)+1
    #C_TN = C_TN[len(u1) - low_stop_limit : len(u1) + high_stop_limit]
    #taus = np.linspace(l_tau,h_tau, high_stop_limit + low_stop_limit)
    #plt.plot(taus, C_TN, linewidth = 1.5, color = 'blue', 
    #         label = 'Empirical cross correlation', marker = 'o')
    plt.legend()
    plt.show()
    
    
    #%%
    
    # Evaluation of the speed estimation error
    L, N, T = 50, 300, 700
    nb_steps = 7
    fourier_cov = lambda w : w**2*np.exp(-w**2)
    speed = np.linspace(0.2,2,nb_steps)
    #speed = np.array([2])
    erreur = []
    for i in trange(len(speed), desc = 'Computing c0', disable = False):
        c = speed[i]
        env = Environnement(nb_timesteps = 3*10**4, L = L, N = N, 
                        fourier_cov= fourier_cov, c0 = c, describe = False)
        u1,u2,times,_ = env.compute_signal(x1, x2, T)
        C_TN = emp_cross_corr(u1, u2, times, verbose = False, all_data = False)
        argmax_index = np.argmax(C_TN)
        time_estimator = times[argmax_index]
        estimator = np.linalg.norm(x1-x2)/(time_estimator+1/2)
        print('estimator : ', estimator)
        print('c0 : ', c)
        print('time_estimator : ', time_estimator)
        erreur.append(np.abs(estimator - c))
    erreur = np.array(erreur)/speed
    #%%
    # pour c0 > 5 l'erreur est très grande car l'écart mesurable est inférieur à la distance
    plt.plot(speed[1::],erreur[1::], linewidth = 1.5, color = 'red', marker = '1')
    plt.title("Error evaluation on the speed estimation with the distance/time ratio")
    plt.xlabel("Speed c0")
    plt.ylabel("Relative Error")
    plt.show()
#%%
# Spatial evolution of C_asy
env = Environnement(nb_timesteps = 8000, L = L, N = N, 
                        fourier_cov= fourier_cov, c0 = 1, describe = True)
res = np.zeros((5,100))
for j in range(5) :
    res[j] = plot_c_asy(x1, X[j], env, low_tau = -20, high_tau = 20, nb_steps = 100)
time = np.linspace(-20,20,100)
plt.figure(1)
plt.subplot(5,1,1)
plt.plot(time,res[0], linewidth = 1, color = 'blue')
plt.subplot(5,1,2)
plt.plot(time,res[1], linewidth = 1, color = 'blue')
plt.subplot(5,1,3)
plt.plot(time,res[2], linewidth = 1, color = 'blue')
plt.subplot(5,1,4)
plt.plot(time,res[3], linewidth = 1, color = 'blue')
plt.subplot(5,1,5)
plt.plot(time,res[4], linewidth = 1, color = 'blue') 
plt.xlabel(r'$\tau$')  
plt.show()