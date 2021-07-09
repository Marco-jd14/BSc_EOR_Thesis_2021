# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 16:03:51 2021

@author: Marco Deken
"""

import pandas as pd
import numpy as np
import sys
from copy import copy
from scipy.linalg import lstsq

from simulate import Effects, Slopes, Variance, Dataset, Fit


class GFE:
    def __init__(self, slopes: Slopes):
        self.slopes = slopes


    def _initial_values(self):
        if self.slopes == Slopes.homog:
            self.beta_hat = np.random.uniform(0.5, 3, size=(self.K, 1))
        else:
            self.beta_hat = np.random.uniform(0.5, 3, size=(self.K, self.G))

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


    def _group_assignment(self, verbose, s):
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

        # Make sure each group has at least 3 members
        self._ensure_min_group_size(3, verbose, s)


    def _ensure_min_group_size(self, min_group_size, verbose, s):
        for g in range(self.G):
            group_size = np.count_nonzero(self.groups_per_indiv == g)
            if group_size < min_group_size:
                if verbose:
                    print("Iteration %s, small group:"%s, g)

                # Choose 1 member from each group with > 3 members for the small group
                # Fill group to 3 members from biggest group
                biggest_group = np.argmax(np.bincount(self.groups_per_indiv))
                group_members = np.arange(self.N)[self.groups_per_indiv == biggest_group]
                indiv_to_change = np.random.choice(group_members, min_group_size-group_size, replace=False)
                self.groups_per_indiv[indiv_to_change] = g


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


    def _sort_groups(self, verbose):
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


    def set_G(self, G):
        self.G = G

    def estimate_G(self, G_max, X, Y, G_min=1):
        assert(G_max > G_min)
        BIC_G = np.zeros(G_max)

        self.set_G(G_max)
        self.fit(X, Y, verbose=False)
        self.predict()

        SSR = self.resids.values.T @ self.resids.values
        theta_hat_sq = SSR / (self.N*self.T - G_max*self.T - self.N - G_max*self.K)
        BIC_G[G_max-1] = SSR/(self.N*self.T) + theta_hat_sq * (self.G*self.T+self.N+self.G*self.K) * np.log(self.N*self.T) / (self.N*self.T)

        best_bic = np.Inf
        groups = copy(self.groups_per_indiv)

        for g in range(G_max-1):
            if g+1 < G_min:
                BIC_G[g] = np.Inf
                continue

            self.set_G(g+1)
            self.fit(X, Y, verbose=False)
            self.predict()

            SSR = self.resids.values.T @ self.resids.values
            BIC_G[g]  = SSR/(self.N*self.T) + theta_hat_sq * (self.G*self.T+self.N+self.G*self.K) * np.log(self.N*self.T) / (self.N*self.T)

            if BIC_G[g] < best_bic:
                best_bic = BIC_G[g]
                groups = copy(self.groups_per_indiv)

        self.G_hat = np.argmin(BIC_G)+1
        col = ['G=%d'%(g+1) for g in range(G_max)]
        col[self.G_hat-1] = 'G_hat=%d'%self.G_hat
        self.BIC = pd.DataFrame(BIC_G.reshape(1,-1), columns=col, index=["BIC"])

        return groups


    def fit_given_groups(self, X, Y, groups_per_indiv, first_fit=True, verbose=True):
        self.G = np.max(groups_per_indiv)+1
        if first_fit:
            self.N = len(set(X.index.get_level_values(0)))
            self.T = len(set(X.index.get_level_values(1)))
            self.K = len(X.columns)
            self.X = copy(X)
            self.Y = Y.values.reshape(self.N*self.T,1)

        self.alpha_hat = np.zeros((self.G,self.T))
        self.groups_per_indiv = copy(groups_per_indiv)

        X = self._prepare_dummy_dataset()
        p, _, _, _ = lstsq(X, self.Y)
        self._update_values(p, X.columns)

        if self.slopes == Slopes.heterog:
            self._sort_groups(verbose)

        self._make_dataframes()


    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, verbose=True):
        self.N = len(set(X.index.get_level_values(0)))
        self.T = len(set(X.index.get_level_values(1)))
        self.K = len(X.columns)

        self.X = copy(X)
        self.Y = Y.values.reshape(self.N*self.T,1)

        self._initial_values()
        prev_groups_per_indiv = np.zeros_like(self.groups_per_indiv)

        for s in range(25):
            self._group_assignment(verbose, s)

            X = self._prepare_dummy_dataset()
            p, _, _, _ = lstsq(X, self.Y)

            self._update_values(p, X.columns)

            if np.all(self.groups_per_indiv == prev_groups_per_indiv):
                break
            prev_groups_per_indiv = copy(self.groups_per_indiv)

        self.nr_iterations = s+1

        if self.slopes == Slopes.heterog:
            self._sort_groups(verbose)

        self._make_dataframes()


    def predict(self):
        self.fitted_values = np.zeros_like(self.Y)

        for g in range(self.G):
            selection = np.where(self.groups_per_indiv == g)[0]
            selection_indices = np.zeros(len(selection)*self.T, dtype=int)
            for i in range(len(selection)):
                selection_indices[i*self.T:(i+1)*self.T] = np.arange(self.T) + selection[i]*self.T

            fixed_effects = np.kron(np.ones(len(selection)), self.alpha_hat.values[g,:])
            self.fitted_values[selection_indices,0] = self.X.values[selection_indices,:] @ self.beta_hat.values[:,g] + fixed_effects

        self.resids = pd.DataFrame(self.Y - self.fitted_values, index=self.X.index)
        index = pd.MultiIndex.from_product([np.arange(self.N), np.arange(self.T)], names=["n", "t"])
        self.fitted_values = pd.DataFrame(self.fitted_values, index=index)





def main():
    from new_estimate import estimate_model, load_results, continue_estimation
    np.random.seed(0)

    N = 100
    T = 10
    G = 2
    K = 1
    M = 10

    fit = Fit.G_known
    var = Variance.homosk
    G_max = G if fit == Fit.G_known else G+4

    train = False
    overwrite = False

    filename = "gfe/gfe_N=%d_T=%d_G=%d_K=%d_M=%d_fit=%d_e=%d" %(N,T,G,K,M,fit.value,var.value)
    if not continue_estimation(filename, fit, train, overwrite, load_results):
        sys.exit(0)

    B = np.array([[0.3, 0.9]])
    col = ['g=%d'%i for i in range(G)]
    row = ['k=%d'%i for i in range(K)]
    slopes_df = pd.DataFrame(B, columns=col, index=row)

    dataset = Dataset(N, T, K, G)
    dataset.simulate(Effects.gr_tvar_fix, Slopes.heterog, var, slopes_df=slopes_df)

    model = GFE(Slopes.heterog)
    model_name = "gfe"

    estimate_model(model, dataset, N, T, K, M, G_max, fit, model_name, filename)

    load_results(filename, fit)


if __name__ == "__main__":
    main()


