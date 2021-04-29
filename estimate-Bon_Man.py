# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 16:03:51 2021

@author: Marco
"""

import pandas as pd
import numpy as np
from copy import copy
from scipy.linalg import lstsq

from simulate import Effects, Slopes, Variance, Dataset
from lib.tracktime import TrackTime, TrackReport


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
            new_x = np.zeros((self.N*self.T, self.K*self.G))
            slope_dummies = pd.get_dummies(slope_dummies).values

            counter = -1
            for feature in self.X.columns:
                counter += 1
                new_x[:, counter*self.G : (counter+1)*self.G] = slope_dummies * self.X[feature].values.reshape(-1,1)
            X = pd.DataFrame(new_x)
        else:
            X = copy(self.X)

        X['alpha'] = effect_dummies
        X = pd.get_dummies(X)
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
        prev_theta = np.zeros_like(self.theta)

        for s in range(100):
            TrackTime("Fit a group")
            self._group_assignment()

            TrackTime("Make dummy labels")
            X = self._prepare_dummy_dataset()

            TrackTime("Least Squares")
            p, _, _, _ = lstsq(X, self.Y)

            TrackTime("Estimate")
            self._update_values(p, X.columns)

            # print((self.theta-prev_theta)**2)
            if np.all((self.theta-prev_theta)**2 < 0.00001):
                break
            prev_theta = copy(self.theta)

        self.nr_iterations = s+1

        #TODO: order group numbers based on slope (including fixed effects)
        if self.slopes == Slopes.heterog:
            self.theta = np.sort(self.theta, axis=1)



np.random.seed(0)
N = 250
T = 10
K = 6


TrackTime("Simulate")
dataset = Dataset(N, T, K, G=4)
dataset.simulate(Effects.ind_fix, Slopes.heterog, Variance.homosk)


TrackTime("Estimate")
G = dataset.G   #assume true value of G is known

x = dataset.data.drop(["y"], axis=1)
y = dataset.data["y"]

gfe = GFE(Slopes.heterog, Effects.gr_tvar_fix)
gfe.estimate_G(dataset.G)
gfe.fit(x, y)



TrackTime("Estimate")

groups_list = [[] for g in range(G)]
for i in range(N):
    groups_list[gfe.groups[i][0]].append(i)


print("TOOK %s ITERATIONS\n"%gfe.nr_iterations)

print("TRUE COEFFICIENTS:")
print(dataset.slopes_df)
# print(dataset.effects_df)
# print(dataset.groups_list)

print("\n\nESTIMATED COEFFICIENTS:")
print(gfe.theta)
# print(alpha)
# print(groups_list)



if gfe.slopes == Slopes.homog:
    TrackTime("Standard libraries")
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


