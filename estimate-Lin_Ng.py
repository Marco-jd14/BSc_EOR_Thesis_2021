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


    def _cluster_regressor(self, q_hat, left_n):
        gamma_star = [np.Inf, np.Inf, 0]

        nr_gamma_options = len(q_hat)-1-2*(self.min_group_size-1)
        if nr_gamma_options <= 0:
            print(nr_gamma_options)
            return gamma_star

        gamma_range = np.zeros(nr_gamma_options)
        q_hat_sorted = np.sort(q_hat)

        for i in range(nr_gamma_options):
            gamma_range[i] = (q_hat_sorted[self.min_group_size-1+i] + q_hat_sorted[self.min_group_size+i])/2

        for gamma in gamma_range:
            ssr_left, ssr_right = self._ssr(gamma, q_hat, left_n)
            if ssr_left + ssr_right < gamma_star[0] + gamma_star[1]:
                gamma_star[0] = ssr_left
                gamma_star[1] = ssr_right
                gamma_star[2] = gamma

        return gamma_star


    def _ssr(self, gamma, q_hat, left_n):
        partition = [np.arange(len(q_hat))[q_hat <= gamma]+left_n, np.arange(len(q_hat))[q_hat > gamma]+left_n]

        TrackTime("Partition indices")
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

        TrackTime("Estimate")
        residuals1 = y1 - x1 @ beta_hat_1
        residuals2 = y2 - x2 @ beta_hat_2

        return residuals1@residuals1.T, residuals2@residuals2.T


    def _split_groups_in_half(self, i, gamma_stars, q_hat_k):
        if i==0:
            new_gammas = [self._cluster_regressor(q_hat_k, 0)]
        else:
            left_n = 0
            new_gammas = []
            for j in range(len(gamma_stars)+1):
                if j==0:
                    rel_q_hat = q_hat_k[q_hat_k <= gamma_stars[j][2]]
                elif j==len(gamma_stars):
                    rel_q_hat = q_hat_k[q_hat_k > gamma_stars[j-1][2]]
                else:
                    rel_q_hat = q_hat_k[np.all([q_hat_k > gamma_stars[j-1][2], q_hat_k <= gamma_stars[j][2]],axis=0)]

                gamma_star = self._cluster_regressor(rel_q_hat, left_n)
                new_gammas.append(gamma_star)
                left_n += len(rel_q_hat)

        gamma_stars = self._update_gamma_stars(i, new_gammas, gamma_stars)

        return gamma_stars


    def _update_gamma_stars(self, i, new_gammas, gamma_stars):
        if 2**(i+1) <= self.G:
            for j in range( len(gamma_stars) ):
                gamma_stars[j][0] = 0
                gamma_stars[j][1] = 0
            for j in range( len(gamma_stars)+1 ):
                gamma_stars.insert(2*j, new_gammas[j])

        else:
            #splitting each group in half leads to too many groups --> pick best groups to split in half
            improvements = np.zeros((len(new_gammas),2))
            for j in range(len(new_gammas)):
                improvements[j,0] = gamma_stars[int(j/2)*2][j%2] / (new_gammas[j][0]+new_gammas[j][1])
                improvements[j,1] = j

            best_new_gamma_indices = np.zeros(self.G - 2**i,dtype=int)
            best_new_gamma_indices[:] = improvements[np.argsort(improvements[:,0])][-(self.G - 2**i):,1]

            for index in best_new_gamma_indices:
                gamma_stars[int(index/2)*2][index%2] = 0
            for index in best_new_gamma_indices:
                for j in range(len(gamma_stars)):
                    if gamma_stars[j][2] > new_gammas[index][2]:
                        gamma_stars.insert(j,new_gammas[index])
                        break
                    if j == len(gamma_stars)-1:
                        gamma_stars.append(new_gammas[index])

        return gamma_stars


    def _calc_k_star(self, gamma_stars_per_k):
        k_star = [np.Inf, 0]
        for k in range(self.K):
            tot_ssr = 0
            for gamma_star in gamma_stars_per_k[k]:
                tot_ssr += (gamma_star[0] + gamma_star[1])

            # print(tot_ssr/k_star[0])
            if tot_ssr < k_star[0]:
                k_star[0] = tot_ssr
                k_star[1] = k

        return k_star[1]


    def _final_estimate(self, gamma_stars, q_hat):
        TrackTime("Calc beta_hats")

        betas = np.zeros((self.K, self.G))
        for i in range(len(gamma_stars)+1):
            if i==0 and i==len(gamma_stars):
                selection = np.arange(self.N)
            elif i==0:
                selection = np.arange(self.N)[q_hat <= gamma_stars[i][2]]
            elif i==len(gamma_stars):
                selection = np.arange(self.N)[q_hat > gamma_stars[i-1][2]]
            else:
                selection = np.arange(self.N)[np.all([q_hat > gamma_stars[i-1][2], q_hat <= gamma_stars[i][2]],axis=0)]

            #TODO: make a groups_list
            #TODO: estimate individual fixed effects?

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
        self.min_group_size = 10

        q_hat = np.zeros((self.N,self.K))
        for i in range(self.N):
            select = np.arange(i*self.T,(i+1)*self.T)
            q_hat[i,:], _, _, _ = lstsq(self.X.values[select], self.Y.values[select])

        gamma_stars_per_k = [[] for i in range(self.K)]
        for k in range(self.K):
            for i in range(int(np.ceil(np.log2(self.G)))):
                gamma_stars_per_k[k] = self._split_groups_in_half(i, gamma_stars_per_k[k], q_hat[:,k])

        k_star = self._calc_k_star(gamma_stars_per_k)

        self._final_estimate(gamma_stars_per_k[k_star], q_hat[:,k_star])


np.random.seed(0)
N = 250
T = 500
K = 2


TrackTime("Simulate")
dataset = Dataset(N, T, K, G=4)
dataset.simulate(Effects.ind_fix, Slopes.heterog, Variance.homosk)


TrackTime("Estimate")
x = dataset.data.drop(["y"], axis=1)
y = dataset.data["y"]

pseudo = PSEUDO()
pseudo.estimate_G(dataset.G)    #assume true value of G is known
pseudo.fit(x,y)

#TODO: pseudo.predict()


TrackTime("Print")

print("\n\nTRUE COEFFICIENTS:")
print(dataset.slopes_df)

print("\n\nESTIMATED COEFFICIENTS:")
print(pseudo.beta)


# from linearmodels import PanelOLS
# model_fe = PanelOLS(y, x, entity_effects = True)
# fe_res = model_fe.fit()
# print("\nFIXED EFFECTS ESTIMATION:"), print(fe_res.params)


print("\n")
TrackReport()
