# -*- coding: utf-8 -*-
"""
Created on Mon May 3 17:40:43 2021

@author: Marco
"""

import pandas as pd
import numpy as np
import os.path
import sys
import pickle
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
        # TrackTime("_calc_beta_hat")
        for g in range(self.G):
            selection = np.where(groups==g)[0]

            selection_indices = np.zeros(len(selection)*self.T, dtype=int)
            for i in range(len(selection)):
                selection_indices[i*self.T:(i+1)*self.T] = np.arange(self.T) + selection[i]*self.T

            x = self.X.values[selection_indices,:]
            y = self.Y.values[selection_indices]

            self.beta_hat[:,g], _, _, _ = lstsq(x, y)
        # TrackTime("Estimate")


    def _ck_means(self):
        ssr_groups = np.zeros((self.N,self.G))
        groups = self._initial_values()
        prev_groups = copy(groups)

        s = 0
        while True:
            for i in range(self.N):
                x = self.X.values[i*self.T:(i+1)*self.T]
                y = self.Y.values[i*self.T:(i+1)*self.T]
                # TrackTime("Calculate SSR")
                for g in range(self.G):
                    ssr_groups[i,g] = self._ssr(g, x, y)
                # TrackTime("Select indivs")

            # TrackTime("Estimate")
            best_fit = np.min(ssr_groups,axis=1)
            for g in range(self.G):
                groups[ssr_groups[:,g] == best_fit] = g

            self._calc_beta_hat(groups)

            if np.all(prev_groups == groups):
                break

            prev_groups =  copy(groups)
            s += 1

        return groups, s


    def _ssr(self, g, x, y):
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


    def set_G(self, G):
        self.G = G

    def estimate_G(self, G_max, X, Y):
        BIC_G = np.zeros(G_max)
        best_bic = np.Inf
        for g in range(G_max):
            self.set_G(g+1)
            try:
                self.fit(X, Y, verbose=False)
                self.predict(verbose=False)
                SSR = self.resids.values @ self.resids.values.T
                NT = self.N*self.T
                BIC_G[g]  = np.log(SSR/NT) + g*self.K * np.sqrt(min(self.N, self.T)) * np.log(NT) / NT + (g-1) * np.log(self.N**2)/self.N**2
                if BIC_G[g] < best_bic:
                    best_bic = BIC_G[g]
                    groups = copy(self.groups_per_indiv)
            except:
                BIC_G[g] = np.Inf

        self.G_hat = np.argmin(BIC_G)+1
        col = ['G=%d'%(g+1) for g in range(G_max)]
        col[self.G_hat-1] = 'G_hat=%d'%self.G_hat
        self.BIC = pd.DataFrame(BIC_G.reshape(1,-1), columns=col, index=["BIC"])

        return groups


    def fit_given_groups(self, X, Y, groups_per_indiv, first_fit=True, verbose=True):
        self.G = np.max(groups_per_indiv)+1
        if first_fit:
            if isinstance(Y, pd.DataFrame):
                Y = Y.iloc[:,0]

            self.x_bar = X.groupby('n').mean()
            self.y_bar = Y.groupby('n').mean()
            self.X = X - self.x_bar
            self.Y = Y - self.y_bar

            self.N = len(set(self.X.index.get_level_values(0)))
            self.T = len(set(self.X.index.get_level_values(1)))
            self.K = len(self.X.columns)

        self.groups_per_indiv = copy(groups_per_indiv)
        self.beta_hat = np.zeros((self.K, self.G))
        self._calc_beta_hat(self.groups_per_indiv)
        self._sort_groups(self.beta_hat)
        self._estimate_fixed_effects()
        self._make_dataframes()


    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, verbose=True):
        if isinstance(Y, pd.DataFrame):
            Y = Y.iloc[:,0]

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
                tot_ssr += self._ssr(groups[i], self.X.values[i*self.T:(i+1)*self.T], self.Y.values[i*self.T:(i+1)*self.T])

            if tot_ssr < best_tot_ssr:
                best_tot_ssr = tot_ssr
                best_beta_hat = self.beta_hat
                self.groups_per_indiv = copy(groups)
                self.nr_iterations = s
                if verbose:
                    print("Iteration %d:\n"%k,np.sort(best_beta_hat, axis=1))

        self._sort_groups(best_beta_hat)
        self._estimate_fixed_effects()
        self._make_dataframes()


    def group_similarity(self, true_groups_per_indiv, true_indivs_per_group, verbose=True):
        # This method only makes sense if G_true == G_hat
        correctly_grouped_indivs = np.where(self.groups_per_indiv == true_groups_per_indiv)[0]
        print("%.2f%% of individuals was put in the correct group" %(len(correctly_grouped_indivs)/self.N * 100))

        if verbose:
            for g in range(self.G):
                true_g = set(true_indivs_per_group[g])
                g_hat = set(self.indivs_per_group[g])
                print("g=%d  \t%d individuals that should be in this group are in a different group" %(g, len(true_g-g_hat)))
                print("\t\t%d individuals are in this group but should be in a different group" %(len(g_hat-true_g)))


    def predict(self, verbose=True):
        self.fitted_values = np.zeros_like(self.Y)

        X = self.X + self.x_bar

        for g in range(self.G):
            selection = np.where(self.groups_per_indiv == g)[0]
            if len(selection) == 0:
                if verbose:
                    print("Group %d out of %d is empty"%(g,self.G))
                continue
            selection_indices = np.zeros(len(selection)*self.T, dtype=int)
            for i in range(len(selection)):
                selection_indices[i*self.T:(i+1)*self.T] = np.arange(self.T) + selection[i]*self.T

            fixed_effects = np.kron(self.alpha_hat.values[selection].reshape(len(selection)),np.ones(self.T))
            self.fitted_values[selection_indices] = X.values[selection_indices,:] @ self.beta_hat.values[:,g] + fixed_effects

        self.resids = self.Y + self.y_bar - self.fitted_values
        index = pd.MultiIndex.from_product([np.arange(self.N), np.arange(self.T)], names=["n", "t"])
        self.fitted_values = pd.DataFrame(self.fitted_values, index=index)



def main():
    from estimate import plot_residuals, plot_fitted_values, plot_clusters
    from Bon_Man import Result

    np.random.seed(0)
    N = 100
    T = 50
    G = 2
    K = 1
    M = 10
    filename = "ckmeans/ckmeans_N=%d_T=%d_G=%d_K=%d_M=%d" %(N,T,G,K,M)

    train = 1
    if not train:
        load_results(filename)
        sys.exit(0)
    else:
        if os.path.isfile(filename):
            print(r"THIS FILE ALREADY EXISTS, ARE YOU SURE YOU WANT TO OVERWRITE? Y\N")
            if not input().upper() == "Y":
                sys.exit(0)


    B = np.array([[0.3, 0.9]])
    # B = np.array([[0.3, 0.5, 0.8]])
    # B = np.array([[0.1, 2/3], [0.3, 0.6]])
    # B = np.array([[0.3, 0.5, 0.7], [-0.3, 0.0, 0.3]])

    # B = np.array([[0.55, 0.65]])
    # B = np.array([[0.4, 0.5, 0.8]])
    # B = np.array([[0.3, 0.4], [0.4, 0.5]])
    # B = np.array([[0.4, 0.5, 0.6], [0.2, 0.3, 0.4]])
    col = ['g=%d'%i for i in range(G)]
    row = ['k=%d'%i for i in range(K)]
    slopes_df = pd.DataFrame(B, columns=col, index=row)


    TrackTime("Simulate")
    dataset = Dataset(N, T, K, G=G)
    dataset.simulate(Effects.ind_fix, Slopes.heterog, Variance.homosk, slopes_df)
    model = CK_means()

    G_max = 5#int(N/12)
    slopes_ests = np.zeros((M, K, G_max))
    groups_ests = np.zeros((M,N), dtype=int)
    bar_length = 30
    for m in range(M):
        if (m) % 1 == 0:
            percent = 100.0*m/(max(1,M-1))
            sys.stdout.write("\rExperiment progress: [{:{}}] {:>3}%".format('='*int(percent/(100.0/bar_length)),bar_length, int(percent)))
            sys.stdout.flush()

        TrackTime("Re-simulate")
        dataset.re_simulate()

        x = dataset.data.drop(["y"], axis=1)
        y = dataset.data["y"]

        TrackTime("Estimate")
        # ASSUME TRUE GROUP MEMBERSHIP IS KNOWN
        # model.fit_given_groups(x, y, dataset.groups_per_indiv, first_fit=True, verbose=False)

        # ASSUME TRUE VALUE OF G IS KNOWN
        # model.set_G(dataset.G)
        # model.fit(x,y,verbose=False)

        # ESTIMATE EVERYTHING
        best_groups = model.estimate_G(G_max, x, y)
        model.fit_given_groups(x, y, best_groups, first_fit=False, verbose=False)
        print("G_hat =",model.G_hat) if model.G_hat != dataset.G else None

        TrackTime("Save results")
        slopes_ests[m,:,:] = np.hstack((model.beta_hat.values,np.zeros((K, G_max-model.G))))
        groups_ests[m,:] = model.groups_per_indiv


    TrackTime("Results")
    result = Result(dataset.slopes_df, slopes_ests, dataset.groups_per_indiv, groups_ests)

    with open(filename, 'wb') as output:
        pickle.dump(result, output, pickle.HIGHEST_PROTOCOL)

    load_results(filename)

    print("\n")
    TrackReport()



def load_results(filename):
    if os.path.isfile(filename):
        with open(filename, 'rb') as output:
            result = pickle.load(output)

    print("\n\nTRUE COEFFICIENTS:")
    print(result.slopes_true)

    result.RMSE()
    print("\nRMSE: %.4f\n" %(result.RMSE*100))

    result.confusion_mat_groups()
    print(result.conf_mat, "\n")

    result.conf_interval(0.05)
    print(result.summary)



if __name__ == "__main__":
    main()
