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


    def _cluster_regressor(self, k, q_hat, left_n):
        gamma_star = [np.Inf, np.Inf, 0, 0]
        #TODO: check if enough individuals w.r.t min_group_size
        gamma_range = np.zeros(len(q_hat)-1-2*(self.min_group_size-1))
        q_hat_sorted = np.sort(q_hat)

        for i in range(len(gamma_range)):
            gamma_range[i] = (q_hat_sorted[self.min_group_size-1+i] + q_hat_sorted[self.min_group_size+i])/2

        for gamma in gamma_range:
            ssr_left, ssr_right = self._ssr(gamma, q_hat, left_n)
            TrackTime("Calc gamma_star")
            if ssr_left + ssr_right < gamma_star[0] + gamma_star[1]:
                gamma_star[0] = ssr_left
                gamma_star[1] = ssr_right
                gamma_star[2] = gamma
                gamma_star[3] = k

        return gamma_star


    def _ssr(self, gamma, q_hat, left_n):
        TrackTime("Partition indices1")
        partition = [np.arange(len(q_hat))[q_hat <= gamma]+left_n, np.arange(len(q_hat))[q_hat > gamma]+left_n]

        TrackTime("Partition indices2")
        partition_indices = [np.zeros(len(partition[0])*self.T,dtype=int), np.zeros(len(partition[1])*self.T, dtype=int)]
        for k in range(2):
            for i in range(len(partition[k])):
                partition_indices[k][i*self.T:(i+1)*self.T] = np.arange(self.T) + partition[k][i]*self.T

        TrackTime("Partition selection")
        x1 = self.X.values[partition_indices[0],:] # ~35x faster than  x1 = self.X.loc[partition[0],:]
        x2 = self.X.values[partition_indices[1],:]
        y1 = self.Y.values[partition_indices[0]]
        y2 = self.Y.values[partition_indices[1]]

        TrackTime("gamma SSR")
        beta_hat_1, _, _, _ = lstsq(x1, y1)
        beta_hat_2, _, _, _ = lstsq(x2, y2)

        residuals1 = y1 - x1 @ beta_hat_1
        residuals2 = y2 - x2 @ beta_hat_2

        return residuals1@residuals1.T, residuals2@residuals2.T


    def _calc_k_star(self, gamma_stars):
        k_star = [np.Inf, 0]
        for k in range(self.K):
            tot_ssr = gamma_stars[k][0][0]
            for gamma in gamma_stars[k]:
                tot_ssr += gamma[1]

            if tot_ssr < k_star[0]:
                k_star[0] = tot_ssr
                k_star[1] = k

        return k_star[1]


    def _final_estimate(self, gamma_star, q_hat):
        TrackTime("Calc beta_hats")

        betas = np.zeros((self.K, self.G))
        for i in range(len(gamma_star)+1):
            if i==0:
                selection = np.arange(self.N)[q_hat <= gamma_star[i][2]]
            elif i==len(gamma_star):
                selection = np.arange(self.N)[q_hat > gamma_star[i-1][2]]
            else:
                selection = np.arange(self.N)[np.all([q_hat > gamma_star[i-1][2], q_hat <= gamma_star[i][2]],axis=0)]

            x_sel = self.X.loc[selection,:]
            y_sel = self.Y.loc[selection,:]
            betas[:,i], _, _, _ = lstsq(x_sel, y_sel)

        col = ['g=%d'%i for i in range(self.G)]
        row = ['k=%d'%i for i in range(self.K)]
        self.beta = pd.DataFrame(betas, columns=col, index=row)


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
            select = np.arange(i*self.T,(i+1)*self.T)
            q_hat[i,:], _, _, _ = lstsq(self.X.values[select], self.Y.values[select])

        self.min_group_size = 10
        gamma_stars = [[] for i in range(self.K)]

        for k in range(self.K):
            q_hat_k = q_hat[:,k]
            for i in range(int(np.ceil(np.log2(self.G)))):
                if i==0:
                    gamma_star = self._cluster_regressor(k, q_hat_k, 0)
                    gamma_stars[k].append(gamma_star)
                elif 2**(i+1) > self.G:
                    #TODO: select groups
                    print("HELLO")
                else:
                    left_n = 0
                    new_gammas = []
                    for j in range(len(gamma_stars[k])+1):
                        if j==0:
                            rel_q_hat = q_hat_k[q_hat_k <= gamma_stars[k][j][2]]
                        elif j==len(gamma_stars[k]):
                            rel_q_hat = q_hat_k[q_hat_k > gamma_stars[k][j-1][2]]
                        else:
                            rel_q_hat = q_hat_k[np.all([q_hat_k > gamma_stars[k][j-1][2], q_hat_k <= gamma_stars[k][j][2]],axis=0)]

                        gamma_star = self._cluster_regressor(k, rel_q_hat, left_n)
                        new_gammas.append(gamma_star)
                        left_n += len(rel_q_hat)

                    for j in range(len(gamma_stars[k])):
                        gamma_stars[k][j][0] = 0
                        gamma_stars[k][j][1] = 0
                    for j in range(len(gamma_stars[k])+1):
                        gamma_stars[k].insert(2*j,new_gammas[j])

        k_star = self._calc_k_star(gamma_stars)

        for gamma in gamma_stars[k_star]:
            print(gamma[2],end='  ')
        print("  k:", k_star)

        self._final_estimate(gamma_stars[k_star], q_hat[:,k_star])


np.random.seed(0)
N = 100
T = 250
K = 3

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
print(dataset.slopes_df.T)

print("\n\nESTIMATED COEFFICIENTS:")
print(pseudo.beta.T)


print("\n")
TrackReport()
