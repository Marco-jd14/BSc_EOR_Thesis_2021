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
            self.beta_hat = np.random.uniform(0.5, 5, size=(self.K, 1))
        else:
            self.beta_hat = np.random.uniform(0.5, 5, size=(self.K, self.G))

        choices = np.random.choice(range(self.N), (self.G), replace=False)
        self.alpha_hat = np.zeros((self.G,self.T))

        for g in range(self.G):
            start = choices[g]*self.T
            end = (choices[g]+1)*self.T
            if self.slopes == Slopes.homog:
                self.alpha_hat[g,:] = (self.Y[start:end] - self.X.values[start:end] @ self.beta_hat).reshape(self.T)
            else:
                self.alpha_hat[g,:] = (self.Y[start:end] - self.X.values[start:end] @ self.beta_hat[:,g].reshape(self.K,1)).reshape(self.T)

        self.groups_per_indiv = np.zeros(self.N, dtype=int)


    def _group_assignment(self):
        unused_g = set(range(self.G))

        for i in range(self.N):
            best_g_val = np.Inf
            if self.slopes == Slopes.homog:
                residuals = self.Y[i*self.T:(i+1)*self.T] - self.X.values[i*self.T:(i+1)*self.T] @ self.beta_hat

            for g in range(self.G):
                if self.slopes == Slopes.heterog:
                    residuals = self.Y[i*self.T:(i+1)*self.T] - self.X.values[i*self.T:(i+1)*self.T] @ self.beta_hat[:,g].reshape(self.K,1)

                temp = (residuals.reshape(self.T) - self.alpha_hat[g,:]).reshape(-1,1)
                fit = temp.T@temp
                if fit < best_g_val:
                    best_g_val = fit
                    self.groups_per_indiv[i] = g

            unused_g.discard(self.groups_per_indiv[i])
        print(unused_g) if len(unused_g) != 0 else None


    def _prepare_dummy_dataset(self):
        effect_dummies = np.zeros(self.N * self.T, dtype=object)
        slope_dummies  = np.zeros(self.N * self.T, dtype=int)

        for i in range(self.N):
            for t in range(self.T):
                effect_dummies[i*self.T + t] = "%d_%d" %(self.groups_per_indiv[i],t)
                slope_dummies[i*self.T + t] = self.groups_per_indiv[i]

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

        X['alphahat'] = effect_dummies
        X = pd.get_dummies(X)
        return X


    def _update_values(self, p, features):
        index = self.K*self.G if self.slopes == Slopes.heterog else self.K
        if self.slopes == Slopes.heterog:
            self.beta_hat = p[:index].reshape(self.K,self.G)
        else:
            self.beta_hat = p[:index]

        i = 0
        for g in features[index::self.T]:
            g = int(g.split("_")[1])
            self.alpha_hat[g,:] = p[index + i*self.T : index + (i+1)*self.T].reshape(self.T)
            i += 1


    def _sort_groups(self):
        reorder = np.argsort(self.beta_hat[0,:])
        self.beta_hat = self.beta_hat[:,reorder]
        self.alpha_hat = self.alpha_hat[reorder,:]

        groups = np.zeros_like(self.groups_per_indiv, dtype=int)
        for i in range(len(reorder)):
            groups[self.groups_per_indiv == reorder[i]] = i
        self.groups_per_indiv = groups


    def _make_dataframes(self):
        self.indivs_per_group = [[] for g in range(self.G)]
        for i in range(self.N):
            self.indivs_per_group[self.groups_per_indiv[i]].append(i)

        col = ['g=%d'%i for i in range(self.G)]
        row = ['k=%d'%i for i in range(self.K)]
        self.beta_hat = pd.DataFrame(self.beta_hat, columns=col, index=row)

        col = ['t=%d'%i for i in range(len(self.alpha_hat[0]))]
        row = ['g=%d'%i for i in range(len(self.alpha_hat))]
        self.alpha_hat = pd.DataFrame(self.alpha_hat, columns=col, index=row)


    def estimate_G(self, G):
        self.G = G

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        self.N = len(set(X.index.get_level_values(0)))
        self.T = len(set(X.index.get_level_values(1)))
        self.K = len(X.columns)

        self.X = copy(X)
        self.Y = Y.values.reshape(N*T,1)

        self._initial_values()
        # prev_beta_hat = np.zeros_like(self.beta_hat)
        prev_groups_per_indiv = np.zeros_like(self.groups_per_indiv)

        for s in range(100):
            TrackTime("Fit a group")
            self._group_assignment()

            TrackTime("Make dummy labels")
            X = self._prepare_dummy_dataset()

            TrackTime("Least Squares")
            p, _, _, _ = lstsq(X, self.Y)

            TrackTime("Estimate")
            self._update_values(p, X.columns)

            if np.all(self.groups_per_indiv == prev_groups_per_indiv):
                break
            # if np.all((self.beta_hat-prev_beta_hat)**2 < 0.00001):
            #     break
            # prev_beta_hat = copy(self.beta_hat)
            prev_groups_per_indiv = copy(self.groups_per_indiv)

        self.nr_iterations = s+1

        if self.slopes == Slopes.heterog:
            self._sort_groups()

        self._make_dataframes()


    def group_similarity(self, true_groups_per_indiv, true_indivs_per_group):
        correctly_grouped_indivs = np.where(self.groups_per_indiv == true_groups_per_indiv)[0]
        print("\n%.2f%% of individuals was put in the correct group" %(len(correctly_grouped_indivs)/self.N * 100))

        for g in range(self.G):
            true_g = set(true_indivs_per_group[g])
            g_hat = set(self.indivs_per_group[g])
            print("g=%d \t%d individuals that should be in this group are in a different group" %(g, len(true_g-g_hat)))
            print("\t\t%d individuals are in this group but should be in a different group" %(len(g_hat-true_g)))


    def predict(self):
        self.fitted_values = np.zeros_like(self.Y)

        for g in range(self.G):
            selection = np.where(self.groups_per_indiv == g)[0]
            selection_indices = np.zeros(len(selection)*self.T, dtype=int)
            for i in range(len(selection)):
                selection_indices[i*self.T:(i+1)*self.T] = np.arange(self.T) + selection[i]*self.T

            fixed_effects = np.kron(np.ones(len(selection)), self.alpha_hat.values[g,:])
            self.fitted_values[selection_indices,0] = self.X.values[selection_indices,:] @ self.beta_hat.values[:,g] + fixed_effects

        self.resids = self.Y - self.fitted_values
        index = pd.MultiIndex.from_product([np.arange(self.N), np.arange(self.T)], names=["n", "t"])
        self.fitted_values = pd.DataFrame(self.fitted_values, index=index)



np.random.seed(0)
N = 500
T = 10
K = 2


TrackTime("Simulate")
dataset = Dataset(N, T, K, G=4)
dataset.simulate(Effects.gr_tvar_fix, Slopes.heterog, Variance.homosk)


TrackTime("Estimate")
x = dataset.data.drop(["y"], axis=1)
y = dataset.data["y"]

gfe = GFE(Slopes.heterog, Effects.gr_tvar_fix)
gfe.estimate_G(dataset.G)       #assume true value of G is known
gfe.fit(x, y)
gfe.predict()



TrackTime("Print")

print("TOOK %s ITERATIONS\n"%gfe.nr_iterations)

print("TRUE COEFFICIENTS:")
print(dataset.slopes_df)
# print(dataset.effects_df)
# print(dataset.groups_per_indiv)
# for group in dataset.indivs_per_group:
#     print(group)

print("\n\nESTIMATED COEFFICIENTS:")
print(gfe.beta_hat)
# print(gfe.alpha_hat)
# print(gfe.groups_per_indiv)
# for group in gfe.indivs_per_group:
#     print(group)

gfe.group_similarity(dataset.groups_per_indiv, dataset.indivs_per_group)


if gfe.slopes == Slopes.homog:
    TrackTime("Standard libraries")

    # from linearmodels import PooledOLS
    # mod = PooledOLS(y, x) 
    # pooledOLS_res = mod.fit(cov_type='clustered', cluster_entity=True)
    # print("\nPOOLED OLS ESTIMATION:"), print(pooledOLS_res.params)

    from linearmodels import RandomEffects
    model_re = RandomEffects(y, x)
    re_res = model_re.fit()
    print("\nRANDOM EFFECTS ESTIMATION:"), print(re_res.params)

    from linearmodels import PanelOLS
    model_fe = PanelOLS(y, x, entity_effects = True)
    fe_res = model_fe.fit()
    print("\nFIXED EFFECTS ESTIMATION:"), print(fe_res.params)


print("\n")
TrackReport()


