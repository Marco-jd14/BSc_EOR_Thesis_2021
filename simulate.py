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


    def reset(self, N: int=0, T: int=0, K: int=0, G: int=0):
        if N==0:
            N = self.N
        if T==0:
            T = self.T
        if K==0:
            K = self.K
        if G==0:
            G = self.G

        for attr in list(vars(self).keys()):
            delattr(self, attr)

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

        self.indivs_per_group = [[] for g in range(self.G)]
        for g in range(self.G):
            for i in range(group_sizes[g]):
                self.indivs_per_group[g].append(individuals[ np.sum(group_sizes[:g]) + i ])

        self.groups_per_indiv = np.zeros(self.N, dtype=int)
        groups_mat = np.zeros((self.G, self.N), dtype=int)
        for g in range(self.G):
            self.indivs_per_group[g][:] = np.sort(self.indivs_per_group[g])
            groups_mat[g, self.indivs_per_group[g]] = 1
            self.groups_per_indiv[self.indivs_per_group[g]] = g

        return groups_mat


    def sim_effects(self, groups_mat):
        if self.effects == Effects.none:
            effects = np.zeros((self.N, 1))
            effects_m = effects @ np.ones((1, self.T))

        elif self.effects == Effects.ind_rand:
            effects = np.random.normal(0, 50/2, size=(self.N, 1))
            effects_m = effects @ np.ones((1, self.T))

        elif self.effects == Effects.ind_fix:
            effects = np.random.uniform(0, 50, size=(self.N, 1))
            effects_m = effects @ np.ones((1, self.T))

        elif self.effects == Effects.gr_tvar_fix:
            effects = np.random.uniform(0, 50, size=(self.G, self.T))
            effects_m = groups_mat.T @ effects

        elif self.effects == Effects.both_fix:
            indiv_fix_eff = np.random.uniform(0, 50, size=(self.N, 1)) @ np.ones((1, self.T))
            group_fix_eff = np.random.uniform(0, 50, size=(self.G, self.T))
            effects_m = groups_mat.T @ group_fix_eff + indiv_fix_eff
            effects = effects_m

        col = ['t=%d'%i for i in range(len(effects[0]))]
        row = ['%c=%d'%('g' if len(effects)==self.G else 'n',i) for i in range(len(effects))]
        self.effects_df = pd.DataFrame(effects, columns=col, index=row)
        return effects_m


    def sim_slopes(self):
        if self.slopes == Slopes.homog:
            B = np.random.uniform(0.5,3, size=(self.K, 1))
            col = ['g=%d'%i for i in range(1)]
        elif self.slopes == Slopes.heterog:
            B = np.random.uniform(0.5,self.G+np.sqrt(self.G), size=(self.K, self.G))
            col = ['g=%d'%i for i in range(self.G)]
            B = np.sort(B, axis=1)

        row = ['k=%d'%i for i in range(self.K)]
        self.slopes_df = pd.DataFrame(B, columns=col, index=row)


    def simulate(self, effects: Effects, slopes: Slopes, var: Variance):
        self.reset()

        self.effects = effects
        self.slopes = slopes
        self.var = var
        self.has_groups = (effects == Effects.gr_tvar_fix or effects == Effects.both_fix or slopes == Slopes.heterog)

        if self.has_groups:
            groups_mat = self.sim_groups()
        else:
            self.G = 1
            groups_mat = np.ones((self.G, self.N), dtype=int)
            self.groups_per_indiv = np.zeros(self.N, dtype=int)
            self.indivs_per_group = [np.arange(self.N)]

        effects_m = self.sim_effects(groups_mat)

        X_range = [10, 40]
        X = np.random.uniform(X_range[0], X_range[1], size=(self.N, self.T, self.K))
        #TODO: fix correlation
        # if self.effects == Effects.ind_fix:
        #     X[:,:,0] += effects_m.values       #create correlation between regressor and ommitted variable (fixed effects)

        # print(pd.DataFrame(np.hstack((indiv_fixed_effects,X[:,:,0]))).corr())

        self.sim_slopes()

        if self.slopes == Slopes.homog:
            Y = X @ self.slopes_df.values.reshape(self.K)
        elif self.slopes == Slopes.heterog:
            Y = np.zeros((self.N, self.T))
            temp = X @ self.slopes_df.values
            for g in range(self.G):
                Y += temp[:,:,g] * groups_mat.T[:,g].reshape(self.N, 1)

        Y += effects_m

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

