# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 16:03:51 2021

@author: Marco
"""

import pandas as pd
import numpy as np
from scipy.linalg import lstsq
from lib.tracktime import TrackTime, TrackReport
from sklearn import linear_model
import statsmodels.api as sm
from copy import copy
# import enum

from simulate import Effects, Slopes, Variance, Dataset


class GFE:
    def __init__(self, slopes: Slopes, effects: Effects):
        if effects != Effects.gr_tvar_fix:
            raise RuntimeError("This form of fixed effects is not (yet) supported by this GFE estimator")
        self.slopes = slopes


    def _initial_values(self):
        if self.slopes == Slopes.homog:
            self.theta = np.random.uniform(0.5, 5, size=(self.K, 1))
        else:
            self.theta = np.random.uniform(0.5, 5, size=(self.K, self.G))

        choices = np.random.choice(range(self.N), (self.G), replace=False)
        self.alpha = np.zeros((self.G,self.T))

        for g in range(self.G):
            start = choices[g]*self.T
            end = (choices[g]+1)*self.T
            if self.slopes == Slopes.homog:
                self.alpha[g,:] = (self.Y[start:end] - self.X.values[start:end] @ self.theta).reshape(self.T)
            else:
                self.alpha[g,:] = (self.Y[start:end] - self.X.values[start:end] @ self.theta[:,g].reshape(self.K,1)).reshape(self.T)

        self.initial_theta = copy(self.theta)
        self.initial_alpha = copy(self.alpha)

        self.groups = np.zeros((self.N,1), dtype=int)


    def _group_assignment(self):
        unused_g = set(range(self.G))

        for i in range(self.N):
            best_g_val = np.Inf
            if self.slopes == Slopes.homog:
                residuals = self.Y[i*self.T:(i+1)*self.T] - self.X.values[i*self.T:(i+1)*self.T] @ self.theta

            for g in range(self.G):
                if self.slopes == Slopes.heterog:
                    residuals = self.Y[i*self.T:(i+1)*self.T] - self.X.values[i*self.T:(i+1)*self.T] @ self.theta[:,g].reshape(self.K,1)

                temp = (residuals.reshape(self.T) - self.alpha[g,:]).reshape(-1,1)
                fit = temp.T@temp
                if fit < best_g_val:
                    best_g_val = fit
                    self.groups[i] = g

            unused_g.discard(self.groups[i][0])
        print(unused_g) if len(unused_g) != 0 else None


    def _prepare_dummy_dataset(self):
        effect_dummies = np.zeros(self.N * self.T, dtype=object)
        slope_dummies  = np.zeros(self.N * self.T, dtype=int)

        for i in range(self.N):
            for t in range(self.T):
                effect_dummies[i*self.T + t] = "%d_%d" %(self.groups[i],t)
                slope_dummies[i*self.T + t] = self.groups[i]

        if self.slopes == Slopes.heterog:
            # TrackTime("Slope dummies")
            new_x = np.zeros((self.N*self.T, self.K*self.G))
            slope_dummies = pd.get_dummies(slope_dummies).values

            counter = -1
            for feature in self.X.columns:
                counter += 1
                new_x[:, counter*self.G : (counter+1)*self.G] = slope_dummies * self.X[feature].values.reshape(-1,1)
            X = pd.DataFrame(new_x)
        else:
            # TrackTime("Copy new X")
            X = copy(self.X)

        X['alpha'] = effect_dummies

        # TrackTime("get_dummies()")
        X = pd.get_dummies(X)#, drop_first=True)
        return X


    def _update_values(self, p, features):
        index = self.K*self.G if self.slopes == Slopes.heterog else self.K
        if self.slopes == Slopes.heterog:
            self.theta = p[:index].reshape(self.K,self.G)
        else:
            self.theta = p[:index]

        i = 0
        for g in features[index::self.T]:
            g = int(g.split("_")[1])
            self.alpha[g,:] = p[index + i*self.T : index + (i+1)*self.T].reshape(self.T)
            i += 1


    def estimate_G(self, G):
        self.G = G

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        self.N = len(set(X.index.get_level_values(0)))
        self.T = len(set(X.index.get_level_values(1)))
        self.K = len(X.columns)

        self.X = copy(X)
        self.Y = Y.values.reshape(N*T,1)

        self._initial_values()
        # print(self.theta)
        # print(self.alpha)
        prev_theta = np.zeros_like(self.theta)

        for s in range(250):
            # TrackTime("Fit a group")
            self._group_assignment()

            # TrackTime("Make dummy labels")
            X = self._prepare_dummy_dataset()

            # TrackTime("lstsq()")
            p, _, _, _ = lstsq(X, self.Y)

            # TrackTime("Update values")
            self._update_values(p, X.columns)

            # print((self.theta-prev_theta)**2)
            if np.all((self.theta-prev_theta)**2 < 0.00001):
                break
            prev_theta = copy(self.theta)

        print("TOOK %s ITERATIONS\n"%(s+1))
        print("ESTIMATED COEFFICIENTS:")
        print(self.theta)
        print("\n")


TrackTime("Simulate")

# np.random.seed(0)
N = 250
T = 10
K = 4

# gfe = GFE(N, T, K)

dataset = Dataset(N, T, K, G=3)
dataset.simulate(Effects.gr_tvar_fix, Slopes.heterog, Variance.homosk)
#TODO: Bonhomme and Manresa estimation

TrackTime("Initialize")
G = dataset.G   #assume true value of G is known

x = dataset.data.drop(["y"], axis=1)
y = dataset.data["y"]#.values.reshape(N*T,1)

TrackTime("class")
gfe = GFE(Slopes.heterog, Effects.gr_tvar_fix)
gfe.estimate_G(dataset.G)
gfe.fit(x, y)


TrackTime("original")
y = y.values.reshape(N*T,1)
heterog = True

#initialize feasible values:
theta = np.random.uniform(0.5, 5, size=(K, G))  # slopes
alpha = np.zeros((G,T))

choices = np.random.choice(range(N), (G), replace=False)
for i in range(G):
    n = choices[i]
    if not heterog:
        alpha[i,:] = (y[n*T:(n+1)*T] - x.values[n*T:(n+1)*T] @ theta).reshape(T)
    else:
        alpha[i,:] = (y[n*T:(n+1)*T] - x.values[n*T:(n+1)*T] @ theta[:,i].reshape(K,1)).reshape(T)


groups = np.zeros((N,1), dtype=int)
prev_theta = np.zeros_like(theta)


theta = gfe.initial_theta
alpha = gfe.initial_alpha
# print(theta)
# print(alpha)

for s in range(250):
    unused_g = set(range(G))
    # TrackTime("Fit a group")
    x = dataset.data.drop(["y"], axis=1)

    for i in range(N):
        best_g_val = np.Inf
        if not heterog:
            resids = y[i*T:(i+1)*T] - x.values[i*T:(i+1)*T]@theta

        for g in range(G):
            if heterog:
                resids = y[i*T:(i+1)*T] - x.values[i*T:(i+1)*T] @ theta[:,g].reshape(K,1)

            temp = (resids.reshape(T) - alpha[g,:]).reshape(-1,1)

            fit = temp.T@temp
            if fit < best_g_val:
                best_g_val = fit
                groups[i] = g


        unused_g.discard(groups[i][0])

    # TrackTime("Make dummy labels")

    effect_dummies = np.zeros(N*T, dtype=object)
    slope_dummies = np.zeros(N*T, dtype=int)
    for i in range(N):
        for t in range(T):
            effect_dummies[i*T + t] = "%d_%d" %(groups[i],t)
            slope_dummies[i*T + t] = groups[i]

    if heterog:
        # TrackTime("Slope dummies")
        new_x = np.zeros((N*T,K*G))
        slope_dummies = pd.get_dummies(slope_dummies).values
    
        counter = -1
        for feature in x.columns:
            counter += 1
            new_x[:,counter*G:(counter+1)*G] = slope_dummies * x[feature].values.reshape(-1,1)
        x = pd.DataFrame(new_x)
    
    x['alpha'] = effect_dummies

    # TrackTime("get_dummies()")
    x = pd.get_dummies(x)#, drop_first=True)

    # TrackTime("lstsq()")
    p, _, _, _ = lstsq(x, y)

    # TrackTime("Update values")

    index = K*G if heterog else K
    if heterog:
        theta = p[:index].reshape(K,G)
    else:
        theta = p[:index]

    # print((theta-prev_theta)**2)

    i = 0
    for g in x.columns[index::T]:
        g = int(g.split("_")[1])
        alpha[g,:] = p[index + i*T : index + (i+1)*T].reshape(T)
        i += 1

    if np.all((theta-prev_theta)**2 < 0.00001):
        break
    prev_theta = theta


TrackTime("Conclusion")

# groups_list = [[] for g in range(G)]
# for i in range(N):
#     groups_list[groups[i][0]].append(i)

print("TOOK %s ITERATIONS\n"%(s+1))

print("TRUE COEFFICIENTS:")
print(dataset.slopes_df)
# print(dataset.effects_df)
# print(dataset.groups_list)

print("\n\nESTIMATED COEFFICIENTS:")
print(theta)
# print(alpha)
# print(groups_list)


exog = dataset.data.drop(['y'], axis=1)
endog = dataset.data['y']

# from linearmodels import PooledOLS
# mod = PooledOLS(endog, exog) 
# pooledOLS_res = mod.fit(cov_type='clustered', cluster_entity=True)
# print("\nPOOLED OLS ESTIMATION:"), print(pooledOLS_res.params)

# from linearmodels import RandomEffects
# model_re = RandomEffects(endog, exog)
# re_res = model_re.fit()
# print("\nRANDOM EFFECTS ESTIMATION:"), print(re_res.params)

# from linearmodels import PanelOLS
# model_fe = PanelOLS(endog, exog, entity_effects = True)
# fe_res = model_fe.fit()
# print("\nFIXED EFFECTS ESTIMATION:"), print(fe_res.params)

print("\n")
TrackReport()


