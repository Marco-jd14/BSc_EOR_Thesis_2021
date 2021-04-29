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

TrackTime("Simulate")

np.random.seed(0)
N = 250
T = 10
K = 4

# gfe = GFE(N, T, K)

dataset = Dataset(N, T, K, G=3)
dataset.simulate(Effects.gr_tvar_fix, Slopes.heterog, Variance.homosk)


#TODO: Bonhomme and Manresa estimation

TrackTime("Initialize")
G = dataset.G   #assume true value of G is known


#initialize feasible values:
theta = np.random.uniform(0.5, 5, size=(K, G))  # slopes
alpha = np.zeros((G,T))
choices = np.random.choice(range(N), (G), replace=False)

x = dataset.data.drop(["y"], axis=1)
y = dataset.data["y"]#.values.reshape(N*T,1)

heterog = True

for i in range(G):
    n = choices[i]
    if not heterog:
        alpha[i,:] = (y.values.reshape(N*T,1)[n*T:(n+1)*T] - x.values[n*T:(n+1)*T] @ theta).reshape(T)
    else:
        alpha[i,:] = (y.values.reshape(N*T,1)[n*T:(n+1)*T] - x.values[n*T:(n+1)*T] @ theta[:,i].reshape(K,1)).reshape(T)


groups = np.zeros((N,1), dtype=int)
prev_theta = np.zeros_like(theta)

class GFE:
    def __init__(self, slopes: Slopes, effects: Effects):
        if effects != Effects.gr_tvar_fix:
            raise RuntimeError("This form of fixed effects is not (yet) supported by this GFE estimator")

        self.slopes = slopes

    def _initial_values(self):
        self.theta = np.random.uniform(0.5, 5, size=(self.K, self.G))  # slopes

        choices = np.random.choice(range(self.N), (self.G), replace=False)
        self.alpha = np.zeros((self.G,self.T))

        for g in range(self.G):
            n = choices[g]
            start = n*self.T
            end = (n+1)*self.T
            if self.slopes == Slopes.homog:
                self.alpha[g,:] = (self.Y[n*self.T:(n+1)*self.T] - self.X.values[n*self.T:(n+1)*self.T] @ self.theta).reshape(self.T)
            else:
                self.alpha[g,:] = (self.Y[n*self.T:(n+1)*self.T] - self.X.values[n*self.T:(n+1)*self.T] @ self.theta[:,g].reshape(self.K,1)).reshape(self.T)


    def _group_assignment(self, X):
        unused_g = set(range(self.G))
        for i in range(self.N):
            best_g_val = np.Inf
            if self.slopes == Slopes.homog:
                residuals = self.Y[i*T:(i+1)*T] - self.X.values[i*T:(i+1)*T]@theta

            for g in range(G):
                if self.slopes == Slopes.heterog:
                    residuals = self.Y[i*T:(i+1)*T] - self.X.values[i*T:(i+1)*T] @ theta[:,g].reshape(K,1)

                temp = (residuals.reshape(T) - alpha[g,:]).reshape(-1,1)
                fit = temp.T@temp
                if fit < best_g_val:
                    best_g_val = fit
                    groups[i] = g

            unused_g.discard(groups[i][0])
        print(unused_g) if len(unused_g) != 0 else None

    def estimate_G(self, G):
        self.G = G

    def fit(self, X, Y):
        self.N = len(set(X.index.get_level_values(0)))
        self.T = len(set(X.index.get_level_values(1)))
        self.K = len(X.columns)

        self.X = copy(X)
        self.Y = Y.values

        self._initial_values()

        for s in range(250):
            
            
            
            TrackTime("Copy new X")
            X = copy(self.X)
            
        

        
gfe = GFE(Slopes.homog, Effects.gr_tvar_fix)
gfe.estimate_G(dataset.G)
# gfe.fit(x, y)
y = y.values.reshape(N*T,1)

for s in range(250):
    unused_g = set(range(G))
    TrackTime("Fit a group")
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

    TrackTime("Make dummy labels")

    effect_dummies = np.zeros(N*T, dtype=object)
    slope_dummies = np.zeros(N*T, dtype=int)
    for i in range(N):
        for t in range(T):
            effect_dummies[i*T + t] = "%d_%d" %(groups[i],t)
            slope_dummies[i*T + t] = groups[i]

    if heterog:
        TrackTime("Slope dummies")
        new_x = np.zeros((N*T,K*G))
        slope_dummies = pd.get_dummies(slope_dummies).values
    
        counter = -1
        for feature in x.columns:
            counter += 1
            new_x[:,counter*G:(counter+1)*G] = slope_dummies * x[feature].values.reshape(-1,1)
        x = pd.DataFrame(new_x)
    
    x['alpha'] = effect_dummies

    TrackTime("get_dummies()")
    x = pd.get_dummies(x)#, drop_first=True)

    TrackTime("lstsq()")
    p, _, _, _ = lstsq(x, y)

    TrackTime("Process results")

    index = K*G if heterog else K
    if heterog:
        theta = p[:index].reshape(K,G)
    else:
        theta = p[:index]

    # print((theta-prev_theta)**2)
    if np.all((theta-prev_theta)**2 < 0.00001):
        break
    prev_theta = theta

    i = 0
    for g in x.columns[index::T]:
        g = int(g.split("_")[1])
        alpha[g,:] = p[index + i*T : index + (i+1)*T].reshape(T)
        i += 1




TrackTime("Conclusion")

groups_list = [[] for g in range(G)]
for i in range(N):
    groups_list[groups[i][0]].append(i)

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

from linearmodels import RandomEffects
model_re = RandomEffects(endog, exog)
re_res = model_re.fit()
print("\nRANDOM EFFECTS ESTIMATION:"), print(re_res.params)

from linearmodels import PanelOLS
model_fe = PanelOLS(endog, exog, entity_effects = True)
fe_res = model_fe.fit()
print("\nFIXED EFFECTS ESTIMATION:"), print(fe_res.params)

print("\n")
TrackReport()


