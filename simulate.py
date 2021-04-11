# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 10:58:46 2021

@author: Marco
"""
import pandas as pd
import numpy as np
import enum


class Effects(enum.Enum):
    none = 0
    ind_rand = 1
    ind_fix = 2
    gr_tvar_fix = 3
    both_fix = 4

class Slopes(enum.Enum):
    homog = 0
    heterog = 1

class Variance(enum.Enum):
    homosk = 0
    heterosk = 1


class Dataset:
    def __init__(self, N: int, T: int, K: int, G: int=1):
        self.N = N
        self.T = T
        self.K = K
        self.G = G


    def sim_groups(self):
        individuals = np.arange(self.N)
        np.random.shuffle(individuals)

        group_sizes = np.zeros(self.G, dtype=int)
        for g in range(self.G):
            if g == self.G - 1:
                group_size = self.N - np.sum(group_sizes)
            else:
                group_size = int(np.round(np.random.uniform(0.75*self.N/self.G, 1.25*self.N/self.G), 0))
            group_sizes[g] = group_size

        self.groups_list = [[] for g in range(self.G)]
        for g in range(self.G):
            for i in range(group_sizes[g]):
                self.groups_list[g].append(individuals[ np.sum(group_sizes[:g]) + i ])

        self.groups_mat = np.zeros((self.G, self.N), dtype=int)
        for g in range(self.G):
            self.groups_mat[g, self.groups_list[g]] = 1


    def sim_effects(self):
        if self.effects == Effects.none:
            effects_m = np.zeros((self.N, self.T))

        elif self.effects == Effects.ind_rand:
            effects_m = np.random.normal(0, 50/2, size=(self.N, 1)) @ np.ones((1, self.T))

        elif self.effects == Effects.ind_fix:
            effects_m = np.random.uniform(0, 50, size=(self.N, 1)) @ np.ones((1, self.T))

        elif self.effects == Effects.gr_tvar_fix:
            group_effects = np.random.uniform(0, 50, size=(self.G, self.T))
            effects_m = self.groups_mat.T @ group_effects

        elif self.effects == Effects.both_fix:
            indiv_fix_eff = np.random.uniform(0, 50, size=(self.N, 1)) @ np.ones((1, self.T))
            group_fix_eff = np.random.uniform(0, 50, size=(self.G, self.T))
            effects_m = self.groups_mat.T @ group_fix_eff + indiv_fix_eff

        col = ['t=%d'%i for i in range(self.T)]
        row = ['n=%d'%i for i in range(self.N)]
        self.effects_df = pd.DataFrame(effects_m, columns=col, index=row)


    def sim_slopes(self):
        if self.slopes == Slopes.homog:
            B = np.random.uniform(0.5,3, size=(self.K, 1))
            col = ['g=%d'%i for i in range(1)]
        elif self.slopes == Slopes.heterog:
            B = np.random.uniform(0.5,5, size=(self.K, self.G))
            col = ['g=%d'%i for i in range(self.G)]

        row = ['k=%d'%i for i in range(self.K)]
        self.slopes_df = pd.DataFrame(B, columns=col, index=row)


    def simulate(self, effects: Effects, slopes: Slopes, var: Variance):
        if hasattr(self, 'groups_list'):
            delattr(self, 'groups_list')
        if hasattr(self, 'groups_mat'):
            delattr(self, 'groups_mat')

        self.effects = effects
        self.slopes = slopes
        self.var = var
        self.has_groups = (effects == Effects.gr_tvar_fix or effects == Effects.both_fix or slopes == Slopes.heterog)

        if self.has_groups:
            self.sim_groups()
        else:
            self.G = 1

        self.sim_effects()

        X_range = [10, 40]
        X = np.random.uniform(X_range[0], X_range[1], size=(self.N, self.T, self.K))
        if self.effects != Effects.ind_rand:
            X[:,:,0] += self.effects_df.values       #create correlation between regressor and ommitted variable (fixed effects)

        # print(pd.DataFrame(np.hstack((indiv_fixed_effects,X[:,:,0]))).corr())

        self.sim_slopes()

        if self.slopes == Slopes.homog:
            Y = X @ self.slopes_df.values.reshape(self.K)
        elif self.slopes == Slopes.heterog:
            Y = np.zeros((self.N, self.T))
            temp = X @ self.slopes_df.values
            for g in range(self.G):
                Y += temp[:,:,g] * self.groups_mat.T[:,g].reshape(self.N, 1)

        Y += self.effects_df.values

        if self.var == Variance.heterosk:
            heterosk = (X[:,:,0]/np.mean(X[:,:,0])) #/np.sqrt(K)
            corr = heterosk
        elif self.var == Variance.homosk:
            homosk = np.ones((self.N, self.T))*3
            corr = homosk

        errors = np.random.normal(0, np.sqrt(np.mean(Y))*corr)
        Y += errors

        index = pd.MultiIndex.from_product([np.arange(self.N), np.arange(self.T)], names=["n", "t"])
        features = ['feature%d'%i for i in range(self.K)]
        Y = Y.reshape(self.N*self.T, 1)
        X = X.reshape(self.N*self.T, self.K)
        self.data = pd.DataFrame(np.hstack((Y, X)), columns=['y'] + features, index=index)

