# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 16:02:00 2021

@author: Marco
"""

import pandas as pd
import numpy as np
from scipy.linalg import lstsq

from simulate import Effects, Slopes, Variance, Dataset
from lib.tracktime import TrackTime, TrackReport

class PSEUDO:
    def __init__(self):
        pass


    def _cluster_regressor(self, k, q_hat):
        gamma_range = np.zeros(N-1-2*(self.min_group_size-1))
        q_hat_sorted = np.sort(q_hat)

        for i in range(len(gamma_range)):
            gamma_range[i] = (q_hat_sorted[self.min_group_size-1+i] + q_hat_sorted[self.min_group_size+i])/2

        for gamma in gamma_range:
            ssr = self._ssr(gamma, q_hat)
            TrackTime("Calc gamma_star")
            if ssr < self.gamma_star[0]:
                self.gamma_star[0] = ssr
                self.gamma_star[1] = gamma
                self.gamma_star[2] = k


    def _ssr(self, gamma, q_hat):
        TrackTime("gamma SSR")
        partition = [np.arange(self.N)[q_hat <= gamma].reshape(1,-1)[0], np.arange(self.N)[q_hat > gamma].reshape(1,-1)[0]]

        TrackTime("SSR selection")
        partition2 = [np.zeros(len(partition[0])*self.T,dtype=int), np.zeros(len(partition[1])*self.T, dtype=int)]
        for k in range(2):
            for i in range(len(partition[k])):
                partition2[k][i*T:(i+1)*T] = np.arange(partition[k][i]*T,(partition[k][i]+1)*T)

        x1 = self.X.values[partition2[0],:] # ~35x faster than  x1 = self.X.loc[partition[0],:]
        x2 = self.X.values[partition2[1],:]
        y1 = self.Y.values[partition2[0]]
        y2 = self.Y.values[partition2[1]]

        TrackTime("gamma SSR")
        beta_hat_1, _, _, _ = lstsq(x1, y1)
        beta_hat_2, _, _, _ = lstsq(x2, y2)

        residuals1 = y1 - x1 @ beta_hat_1
        residuals2 = y2 - x2 @ beta_hat_2

        return residuals1@residuals1.T + residuals2@residuals2.T


    def estimate_G(self, G):
        self.G = G

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        #demean data:
        self.X = X - X.groupby('n').mean()
        self.Y = Y - Y.groupby('n').mean()

        self.N = len(set(self.X.index.get_level_values(0)))
        self.T = len(set(self.X.index.get_level_values(1)))
        self.K = len(self.X.columns)


        q_hat = np.zeros((self.N,self.K))
        for i in range(self.N):
            q_hat[i,:], _, _, _ = lstsq(self.X.loc[i,:], self.Y.loc[i,:])

        self.min_group_size = 10

        self.gamma_star = [np.Inf, 0, 0]
        TrackTime("Calc gamma_star")
        for k in range(self.K):
            self._cluster_regressor(k, q_hat[:,k])

        TrackTime("Calc beta_hat")
        gamma = self.gamma_star[1]
        k = self.gamma_star[2]
        partition = [np.arange(self.N)[q_hat[:,k] <= gamma].reshape(1,-1)[0], np.arange(self.N)[q_hat[:,k] > gamma].reshape(1,-1)[0]]

        x1 = self.X.loc[partition[0],:]
        x2 = self.X.loc[partition[1],:]
        y1 = self.Y.loc[partition[0],:]
        y2 = self.Y.loc[partition[1],:]

        beta_hat_1, _, _, _ = lstsq(x1, y1)
        beta_hat_2, _, _, _ = lstsq(x2, y2)

        col = ['g=%d'%i for i in range(self.G)]
        row = ['k=%d'%i for i in range(self.K)]
        self.beta = pd.DataFrame(np.vstack([beta_hat_1, beta_hat_2]).T, columns=col, index=row)



np.random.seed(0)
N = 100
T = 100
K = 2

TrackTime("Simulate")
dataset = Dataset(N, T, K, G=2)
dataset.simulate(Effects.ind_fix, Slopes.heterog, Variance.homosk)

#TODO: Lin and Ng estimation for G>2 groups
TrackTime("Estimate")
G = dataset.G   #assume true value of G is known

x = dataset.data.drop(["y"], axis=1)
y = dataset.data["y"]


pseudo = PSEUDO()
pseudo.estimate_G(dataset.G)
pseudo.fit(x,y)




TrackTime("Print")

print("TRUE COEFFICIENTS:")
print(dataset.slopes_df)

print("\n\nESTIMATED COEFFICIENTS:")
print(pseudo.beta)


print("\n")
TrackReport()
