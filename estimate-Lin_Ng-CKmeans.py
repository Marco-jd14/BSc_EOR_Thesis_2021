# -*- coding: utf-8 -*-
"""
Created on Mon May 3 17:40:43 2021

@author: Marco
"""

import pandas as pd
import numpy as np
from copy import copy
from scipy.linalg import lstsq

from simulate import Effects, Slopes, Variance, Dataset
from lib.tracktime import TrackTime, TrackReport


class CK_means:
    def __init__(self):
        pass


    def _initial_values(self):
        groups = np.random.choice(np.arange(self.G), size=self.N, replace=True)

        self.beta_hat = np.zeros((self.K, self.G))
        self._calc_beta_hat(groups)

        return groups


    def _calc_beta_hat(self, groups):
        TrackTime("_calc_beta_hat")
        for g in range(self.G):
            selection = np.where(groups==g)[0]

            selection_indices = np.zeros(len(selection)*self.T, dtype=int)
            for i in range(len(selection)):
                selection_indices[i*self.T:(i+1)*self.T] = np.arange(self.T) + selection[i]*self.T

            x = self.X.values[selection_indices,:]
            y = self.Y.values[selection_indices]

            self.beta_hat[:,g], _, _, _ = lstsq(x, y)
        TrackTime("Estimate")


    def _ck_means(self):
        ssr_groups = np.zeros((self.N,self.G))
        groups = self._initial_values()
        prev_groups = copy(groups)

        s = 0
        while True:
            for i in range(self.N):
                x = self.X.values[i*self.T:(i+1)*self.T]
                y = self.Y.values[i*self.T:(i+1)*self.T]
                TrackTime("Calculate SSR")
                for g in range(self.G):
                    ssr_groups[i,g] = self._ssr(i,g, x, y)
                TrackTime("Select indivs")

            TrackTime("Estimate")
            best_fit = np.min(ssr_groups,axis=1)
            for g in range(self.G):
                groups[ssr_groups[:,g] == best_fit] = g

            self._calc_beta_hat(groups)

            if np.all(prev_groups == groups):
                break

            prev_groups =  copy(groups)
            s += 1

        return groups, s


    def _ssr(self, i, g, x, y):
        residuals = y - x @ self.beta_hat[:,g]
        return residuals@residuals.T


    def _sort_groups(self, best_beta_hat):
        reorder = np.argsort(best_beta_hat[0,:])
        self.beta_hat = best_beta_hat[:,reorder]

        groups = np.zeros_like(self.groups_per_indiv, dtype=int)
        for i in range(len(reorder)):
            groups[self.groups_per_indiv == reorder[i]] = i
        self.groups_per_indiv = groups


    def _estimate_fixed_effects(self):
        self.alpha_hat = np.zeros(self.N)
        for g in range(self.G):
            selection = (self.groups_per_indiv == g)
            self.alpha_hat[selection] = self.y_bar.values[selection] - self.x_bar.values[selection,:] @ self.beta_hat[:,g]


    def _make_dataframes(self):
        self.indivs_per_group = [[] for g in range(self.G)]
        for i in range(self.N):
            self.indivs_per_group[self.groups_per_indiv[i]].append(i)

        col = ['g=%d'%i for i in range(self.G)]
        row = ['k=%d'%i for i in range(self.K)]
        self.beta_hat = pd.DataFrame(self.beta_hat, columns=col, index=row)

        col = ['t=%d'%i for i in range(1)]
        row = ['n=%d'%i for i in range(len(self.alpha_hat))]
        self.alpha_hat = pd.DataFrame(self.alpha_hat, columns=col, index=row)


    def estimate_G(self, G):
        self.G = G

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        #demean data:
        self.x_bar = X.groupby('n').mean()
        self.y_bar = Y.groupby('n').mean()
        self.X = X - self.x_bar
        self.Y = Y - self.y_bar

        self.N = len(set(self.X.index.get_level_values(0)))
        self.T = len(set(self.X.index.get_level_values(1)))
        self.K = len(self.X.columns)

        best_tot_ssr = np.Inf
        for k in range(20):
            groups, s = self._ck_means()

            tot_ssr = 0
            for i in range(self.N):
                tot_ssr += self._ssr(i, groups[i], self.X.values[i*T:(i+1)*T], self.Y.values[i*T:(i+1)*T])

            if tot_ssr < best_tot_ssr:
                best_tot_ssr = tot_ssr
                best_beta_hat = self.beta_hat
                self.groups_per_indiv = copy(groups)
                self.nr_iterations = s
                print("Iteration %d:\n"%k,np.sort(best_beta_hat, axis=1))

        self._sort_groups(best_beta_hat)
        self._estimate_fixed_effects()
        self._make_dataframes()



np.random.seed(0)
N = 250
T = 50
K = 2


TrackTime("Simulate")
dataset = Dataset(N, T, K, G=7)
dataset.simulate(Effects.ind_fix, Slopes.heterog, Variance.homosk)


TrackTime("Estimate")
x = dataset.data.drop(["y"], axis=1)
y = dataset.data["y"]

ck_means = CK_means()
ck_means.estimate_G(dataset.G)    #assume true value of G is known
ck_means.fit(x,y)


#TODO: Move certain parts of code to functions

#TODO: Make comments

#TODO: pseudo.predict()

#TODO: Plot predictions

#TODO: Plot residuals

#TODO: estimate G


TrackTime("Print")

print("\n\nTOOK %s ITERATIONS\n"%ck_means.nr_iterations)

print("\n\nTRUE COEFFICIENTS:")
print(dataset.slopes_df)
print(dataset.effects_df)
# print(dataset.groups_per_indiv)
for group in dataset.indivs_per_group:
    print(group)

print("\n\nESTIMATED COEFFICIENTS:")
print(ck_means.beta_hat)
print(ck_means.alpha_hat)
# print(gfe.groups_per_indiv)
for group in ck_means.indivs_per_group:
    print(group)

# from linearmodels import PanelOLS
# model_fe = PanelOLS(y, x, entity_effects = True)
# fe_res = model_fe.fit()
# print("\nFIXED EFFECTS ESTIMATION:"), print(fe_res.params)


print("\n")
TrackReport()
