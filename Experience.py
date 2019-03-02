#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 09:21:30 2019

@author: evrardgarcelon
"""

# Attention quad et integrale complexe

import numpy as np
import datetime as dt

from environnement import Environnement
from scipy.integrate import quad

import os
import pickle

from src.tools.utils import create_dir


def complex_integration(f, a, b):
    def real_f(x): return np.real(f(x))

    def imag_f(x): return np.imag(f(x))
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
                 exp_name='green_function_estimation'):

        date = dt.datetime.today()
        string_date = "_".join([
            str(date.year).zfill(4),
            str(date.month).zfill(2),
            str(date.day).zfill(2),
            str(date.hour).zfill(2),
            str(date.minute).zfill(2)
        ])
        self.folder = os.path.join(folder, string_date, exp_name)
        create_dir(self.folder)

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
        self.env = Environnement(**self.parameters)

    def compute_signals(self, x1, x2, tau=None, T=None):

        if tau is None:
            tau = self.tau
        if T is None:
            T = self.T
        return self.env.compute_signal(x1, x2, tau, T)

    def emp_cross_correlation(self, x1, x2, T=None, tau=None):

        if tau is None:
            tau = self.tau
        if T is None:
            T = self.T
        u, u_lagged, time_dicretization = self.compute_signals(x1, x2, tau, T)
        emp_cross_cor = 1 / (T - time_dicretization[0]) * np.sum(
            u * u_lagged * (time_dicretization[1::] - time_dicretization))
        return emp_cross_cor

    def exp_emp_cross_correlation(self, x1, x2, tau=None, N=None):

        if tau is None:
            tau = self.tau
        if N is None:
            N = self.env_parameters['N']

        y = self.env.y
        G_hat = self.env.G_hat
        temp_cor = 0
        for s in range(N):
            def f(w): return self.F_hat(w) * np.conjugate(G_hat(w, x1,
                                                                y[s])) * G_hat(w, x2, y[s]) * np.exp(-1j * w * tau)
            temp_cor += quad(f, a=-np.inf, b=+np.inf)[0]
        return temp_cor / np.sqrt(2 * np.pi * N)

    def cross_1(self, x1, x2, tau=None, L=None):
        if L is None:
            L = self.env_parameters['L']
        if tau is None:
            tau = self.tau

        def G_hat(w, x, theta): return self.env.G_hat(
            w, x, L * np.array([np.cos(theta), np.sin(theta)]))

        def f(theta): return quad(lambda w: self.F_hat(w) * np.conjugate(G_hat(w, x1, theta)) *
                                  G_hat(w, x2, theta) * np.exp(-1j * w * tau), a=-np.inf, b=np.inf)[0]
        return quad(f, a=0, b=2 * np.pi)[0]

    def C_asy(self, x1, x2, tau=None):

        if tau is None:
            tau = self.tau
        c0 = self.env_parameters['c0']
        L = self.env_parameters['L']

        def G(t): return self.env.G(t, x1, x2)

        def G_rev(t): return self.env.G(-t, x1, x2)

        def dC_asy(tau): return - c0 / (8 * np.pi**2 * L) * \
            (np.convole(self.F, G)(tau) - np.convole(self.F, G_rev)(tau))
        temp_C_asy = quad(dC_asy, a=- np.inf, b=tau)[0]
        return temp_C_asy

    def plot_correlation(self, x1, x2):
        pass

    # TODO quel sorte de graphique à faire

    def save(self):
        # TODO Pickle


if __name__ == '__main__':
    pass
