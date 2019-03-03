#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 09:21:30 2019

@author: evrardgarcelon
"""
# TODO  Too Long execution time

import numpy as np
#import datetime as dt

from environnement import Environnement
from scipy.integrate import quad
import pylab as plt
from tqdm import tqdm

# import os
# import pickle

# from src.tools.utils import create_dir

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
    return quad(real_f, a, b)[0] + 1j * quad(imag_f, a, b)[0]


class Experience(object):

    def __init__(self, T=100,
                 N=50,
                 L=50,
                 tau=1,
                 F=None,
                 F_hat=None,
                 seed=1,
                 c0=1,
                 folder='experiment',
                 exp_name='green_function_estimation',
                 verbose = False):

        #        date = dt.datetime.today()
        #        string_date = "_".join([
        #            str(date.year).zfill(4),
        #            str(date.month).zfill(2),
        #            str(date.day).zfill(2),
        #            str(date.hour).zfill(2),
        #            str(date.minute).zfill(2)
        #        ])
        #        self.folder = os.path.join(folder, string_date, exp_name)
        #        create_dir(self.folder)

        self.env_parameters = {'c0': c0,
                               'N': N,
                               'L': L}
        self.tau = tau
        self.T = T
        self.seed = seed

        if F is None and F_hat is None:
            self.F_hat = lambda w: w**2 * np.exp(-w**2)
            self.F = lambda x: np.exp(-x**2 / 4) * \
                (2 - x**2) / (4 * np.sqrt(2))
        else:
            self.F_hat = F_hat
            self.F = F
        self.env = Environnement(**self.env_parameters)
        self.colors = ['red', 'green', 'blue', 'magenta', 'cyan', 'yellow', 'black', 'grey', 'orange']
        self.verbose = verbose

    def compute_signals(self, x1, x2, tau=None, T=None):

        if tau is None:
            tau = self.tau
        if T is None:
            T = self.T
        u, u_lagged, time, _ = self.env.compute_signal(x1, x2, tau, T)
        return u, u_lagged, time

    def emp_cross_correlation(self, x1, x2, T=None, tau=None):
        if self.verbose : 
            print('Computing empirical cross correlation...')
        if tau is None:
            tau = self.tau
        if T is None:
            T = self.T
        u, u_lagged, time_dicretization = self.compute_signals(x1, x2, tau, T)
        emp_cross_cor = 1 / (T - time_dicretization[0]) * np.sum(
            u[:-1] * u_lagged[:-1] * (time_dicretization[1::] - time_dicretization[:-1]))
        if self.verbose : 
            print('Done')
        return emp_cross_cor

    def exp_emp_cross_correlation(self, x1, x2, tau=None, N=None, L=None):
        if self.verbose : 
            print('Computing expectation of empirical cross correlation ...')
        if tau is None:
            tau = self.tau
        if N is None:
            N = self.env_parameters['N']
        if L is None:
            L = self.env_parameters['L']
        thetas = np.random.uniform(size=N)
        y = np.zeros((N, 2))
        y[:, 0] = L * np.cos(2 * np.pi * thetas)
        y[:, 1] = L * np.sin(2 * np.pi * thetas)
        G_hat = self.env.G_hat
        temp_cor = 0
        for s in range(N):
            def f(w): return self.F_hat(w) * np.conjugate(G_hat(w, x1,
                                                                y[s])) * G_hat(w, x2, y[s]) * np.exp(-1j * w * tau)
            temp_cor += complex_integration(f, a=-np.inf, b=+np.inf)
        if self.verbose :
            print('Done')
        return temp_cor / np.sqrt(2 * np.pi * N)

    def cross_1(self, x1, x2, tau=None, L=None):
        # Takes too long
        if self.verbose : 
            print('Computing C_(1)...')
        if L is None:
            L = self.env_parameters['L']
        if tau is None:
            tau = self.tau

        def G_hat(w, x, theta): return self.env.G_hat(
            w, x, L * np.array([np.cos(theta), np.sin(theta)]))

        def f(theta): return complex_integration(lambda w: self.F_hat(w) * np.conjugate(G_hat(w, x1, theta)) *
                                                 G_hat(w, x2, theta) * np.exp(-1j * w * tau), a=-np.inf, b=np.inf)
        result = complex_integration(f, a=0, b=2 * np.pi)
        if self.verbose : 
            print('Done')
        return result

    def C_asy(self, x1, x2, tau=None):
        if self.verbose : 
            print('Computing C_asy ...')
        if tau is None:
            tau = self.tau
        c0 = self.env_parameters['c0']
        L = self.env_parameters['L']

        def temp_f(t): return self.F(t) * self.env.G(tau - t, x1, x2)

        def temp_rev_f(t): return self.F(t) * self.env.G(t - tau, x1, x2)

        def temp(t): return temp_f(t) - temp_rev_f(t)

        def dC_asy(tau): return - c0 / (8 * np.pi**2 * L) * \
            quad(temp, a=-np.inf, b=np.inf)[0]
        temp_C_asy = complex_integration(dC_asy, a=- np.inf, b=tau)
        if self.verbose :
            print('Done')
        return temp_C_asy

    def compare_all(self, x1, x2, low_tau, high_tau, T=None, L=None, N=None):
        # Rajouter tqdm
        taus = np.linspace(low_tau, high_tau, 10)

        self.emp_cross_cors = []
        self.exp_emp_cross_cors = []
        self.cross_cors_1 = []
        self.cross_asys = []

        for tau in tqdm(taus):

            self.emp_cross_cors.append(
                self.emp_cross_correlation(
                    x1, x2, tau=tau, T=T))
            self.exp_emp_cross_cors.append(
                self.exp_emp_cross_correlation(
                    x1, x2, tau=tau, L=L, N=N))
            self.cross_cors_1.append(self.cross_1(x1, x2, tau=tau, L=L))
            self.cross_asys.append(self.C_asy(x1, x2, tau=tau))
        self.emp_cross_cors = np.array(self.emp_cross_cors)
        self.exp_emp_cross_cors = np.array(self.exp_emp_cross_cors)
        self.cross_cors_1 = np.array(self.cross_cors_1)
        self.cross_asys = np.array(self.cross_asys)
        
        plt.figure(0)       
        plt.plot(taus,self.emp_cross_cors,label = 'Empirical cross correlation',
                 marker = 'o', color = self.colors[0])
        plt.plot(taus,self.exp_emp_cross_cors,label = 'Expectation of empirical cross correlation',
                 marker = 'o', color = self.colors[1])
        plt.plot(taus,self.cross_cors_1,label = 'Cross correlation 1',
                 marker = 'o', color = self.colors[2])
        plt.plot(taus,self.cross_asys,label = 'Asymmetrical cross correlation',
                 marker = 'o', color = self.colors[3])
        plt.legend()
        plt.show()
        
    # TODO quel sorte de graphique à faire

    def save_var(self, X, name):
        try:
            with open(name, "w") as f:
                f.write(X)
        except BaseException:
            pass

    def save_all(self):
        print('Saving variables...')
        names = {'emp_cross_cor': 'empirical_cross_correlation',
                 'exp_emp_cross_cor': 'expectation_cross_correlation',
                 'cross_1': 'cross_correlataion_(1)',
                 'C_asy': 'cross_correlation_asymetrical'}
        var = [
            self.emp_cross_cors,
            self.exp_emp_cross_cors,
            self.cross_cors_1,
            self.cross_asys]
        for i, name in names.items:
            self.save_var(var[i], name)
        print('Done')


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=RuntimeWarning)
        experience = Experience()
        x1 = np.ones(2)
        x2 = 2 * np.ones(2)
#        import time
#        debut = time.clock()
#        emp_cross_correlation = experience.emp_cross_correlation(x1, x2)
#        exp_emp_cross_cor = experience.exp_emp_cross_correlation(x1, x2)
#        cross_1 = experience.cross_1(x1, x2)
#        C_asy = experience.C_asy(x1, x2)
#        fin = time.clock()
#        print("Temps d'éxécution : ", fin - debut)
        experience.compare_all(x1, x2, low_tau = 1, high_tau = 10)
