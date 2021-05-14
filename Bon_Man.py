# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 16:03:51 2021

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


class GFE:
    def __init__(self, slopes: Slopes):
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

        # self.indivs_per_group = [[] for g in range(self.G)]
        # for i in range(self.N):
        #     self.indivs_per_group[self.groups_per_indiv[i]].append(i)

        # Make sure each group has at least 5 members
        for g in range(self.G):
            group_size = np.count_nonzero(self.groups_per_indiv == g) #len(self.indivs_per_group[g])
            if group_size < 5:
                if verbose:
                    print("%s: Small group:"%s, g)

                # Choose 1 member from each group with > 5 members for the small group
                # Fill group to 5 members from biggest group
                biggest_group = np.argmax(np.bincount(self.groups_per_indiv))
                group_members = np.arange(self.N)[self.groups_per_indiv == biggest_group]
                indiv_to_change = np.random.choice(group_members, 5-group_size, replace=False)
                self.groups_per_indiv[indiv_to_change] = g
                # for i in indiv_to_change:
                #     self.indivs_per_group[biggest_group].remove(i)
                # self.indivs_per_group[g].append(indiv_to_change)


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

    def estimate_G(self, G_max, X, Y):
        BIC_G = np.zeros(G_max)

        self.set_G(G_max)
        self.fit(X, Y, verbose=False)
        self.predict()
        SSR = self.resids.values.T @ self.resids.values
        theta_hat_sq = SSR / (self.N*self.T - G_max*self.T - self.N - self.K)
        BIC_G[G_max-1] = SSR/(self.N*self.T) + theta_hat_sq * (self.G*self.T+self.N+self.K) * np.log(self.N*self.T) / (self.N*self.T)

        for g in range(G_max-1):
            self.set_G(g+1)
            # try:
            self.fit(X, Y, verbose=False)
            self.predict()
            SSR = self.resids.values.T @ self.resids.values
            BIC_G[g]  = SSR/(self.N*self.T) + theta_hat_sq * (self.G*self.T+self.N+self.K) * np.log(self.N*self.T) / (self.N*self.T)
            # except:
            #     BIC_G[g] = np.Inf

        self.G_hat = np.argmin(BIC_G)+1
        col = ['G=%d'%(g+1) for g in range(G_max)]
        col[self.G_hat-1] = 'G_hat=%d'%self.G_hat
        self.BIC = pd.DataFrame(BIC_G.reshape(1,-1), columns=col, index=["BIC"])


    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, verbose=True):
        self.N = len(set(X.index.get_level_values(0)))
        self.T = len(set(X.index.get_level_values(1)))
        self.K = len(X.columns)

        self.X = copy(X)
        self.Y = Y.values.reshape(self.N*self.T,1)

        self._initial_values()
        # prev_beta_hat = np.zeros_like(self.beta_hat)
        prev_groups_per_indiv = np.zeros_like(self.groups_per_indiv)

        for s in range(25):
            # TrackTime("Fit a group")
            self._group_assignment(verbose, s)

            # TrackTime("Make dummy labels")
            X = self._prepare_dummy_dataset()

            # TrackTime("Least Squares")
            p, _, _, _ = lstsq(X, self.Y)

            # TrackTime("Estimate")
            self._update_values(p, X.columns)

            if np.all(self.groups_per_indiv == prev_groups_per_indiv):
                break
            # if np.all((self.beta_hat-prev_beta_hat)**2 < 0.00001):
            #     break
            # prev_beta_hat = copy(self.beta_hat)
            prev_groups_per_indiv = copy(self.groups_per_indiv)

        self.nr_iterations = s+1

        if self.slopes == Slopes.heterog:
            self._sort_groups(verbose)

        self._make_dataframes()


    def group_similarity(self, true_groups_per_indiv, true_indivs_per_group, verbose=True):
        # This method only makes sense if G_true == G_hat
        correctly_grouped_indivs = np.where(self.groups_per_indiv == true_groups_per_indiv)[0]
        print("%.2f%% of individuals was put in the correct group" %(len(correctly_grouped_indivs)/self.N * 100))

        if verbose and self.G == len(true_indivs_per_group):
            for g in range(self.G):
                true_g = set(true_indivs_per_group[g])
                g_hat = set(self.indivs_per_group[g])
                print("g=%d  \t%d individuals that should be in this group are in a different group" %(g, len(true_g-g_hat)))
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

        self.resids = pd.DataFrame(self.Y - self.fitted_values, index=self.X.index)
        index = pd.MultiIndex.from_product([np.arange(self.N), np.arange(self.T)], names=["n", "t"])
        self.fitted_values = pd.DataFrame(self.fitted_values, index=index)


class Result:
    def __init__(self, slopes_true, slopes_ests, groups_true, groups_ests):
        self.M = len(groups_ests)
        self.N = len(groups_ests[0])
        self.K = len(slopes_true)
        self.G = len(slopes_true.columns)

        self.G_MAX = np.max(groups_ests)+1
        self.slopes_ests = slopes_ests[:,:,:self.G_MAX]
        self.slopes_true = slopes_true

        self.groups_ests = groups_ests
        self.groups_true = groups_true

    def RMSE(self):
        self.RMSE = 0
        for m in range(self.M):
            for i in range(self.N):
                  difference = self.slopes_ests[m,:,self.groups_ests[m,i]] - self.slopes_true.values[:,self.groups_true[i]]
                  self.RMSE += difference @ difference.T
        self.RMSE = np.sqrt(self.RMSE / (self.N*self.K*self.M))
        return self.RMSE

    def confusion_mat_groups(self):
        conf_mat = np.zeros((self.G, self.G_MAX),dtype=int)

        for m in range(self.M):
            for i in range(self.N):
                true_group = self.groups_true[i]
                est_group = self.groups_ests[m,i]
                conf_mat[true_group, est_group] += 1

        col = ['g_hat=%d'%i for i in range(self.G_MAX)]
        row = ['g=%d'%i for i in range(self.G)]
        self.conf_mat = pd.DataFrame(conf_mat, columns=col, index=row)

    def conf_interval(self, alpha):
        estimates = np.sort(self.slopes_ests,axis=0).reshape(self.M, self.K*self.G_MAX)

        if self.G_MAX > self.G:
            mean = np.zeros(self.K*self.G_MAX)
            stdev = np.zeros(self.K*self.G_MAX)
            lower_ci = np.zeros(self.K*self.G_MAX)
            upper_ci = np.zeros(self.K*self.G_MAX)

            for k in range(self.K):
                for g in range(self.G_MAX):
                    rel_ests = estimates[np.nonzero(estimates[:,k*self.G_MAX+g]),k*self.G_MAX+g]
                    mean[k*self.G_MAX+g] = np.mean(rel_ests)
                    stdev[k*self.G_MAX+g] = np.std(rel_ests)
                    lower_ci[k*self.G_MAX+g] = rel_ests[0,int(alpha/2 * len(rel_ests[0]))]
                    upper_ci[k*self.G_MAX+g] = rel_ests[0,int((1-alpha/2) * len(rel_ests[0]))]

        else:
            mean = np.mean(estimates, axis=0)
            stdev = np.std(estimates, axis=0)
            lower_ci = estimates[int(alpha/2 * self.M),:]
            upper_ci = estimates[int((1-alpha/2) * self.M),:]

        row = ["k=%d g=%d"%(k,g) for k in range(self.K) for g in range(self.G_MAX)]
        col = ["Lower CI", "Upper CI", "Mean", "St. dev."]
        self.summary = pd.DataFrame(np.vstack((lower_ci,upper_ci,mean,stdev)).T, columns=col, index=row)


def main():
    from estimate import plot_residuals, plot_fitted_values, plot_clusters

    # CHANGED s=100 TO s=25 !!!!!

    np.random.seed(0)
    N = 100
    T = 10
    G = 2
    K = 1
    M = 20
    filename = "gfe/gfe_N=%d_T=%d_G=%d_K=%d_M=%d" %(N,T,G,K,M)

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
    dataset.simulate(Effects.gr_tvar_fix, Slopes.heterog, Variance.homosk, slopes_df)
    # dataset.simulate(Effects.gr_tvar_fix, Slopes.heterog, Variance.homosk)
    model = GFE(Slopes.heterog)

    G_max = 7#int(N/12)
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
        # model.estimate_G(dataset.G)    #assume true value of G is known
        model.estimate_G(G_max, x, y)
        print("G_hat =",model.G_hat)
        print(model.BIC)
        model.set_G(model.G_hat)
        if M == 1:
            model.fit(x,y,verbose=True)
            print("\nTook %d iterations"%model.nr_iterations)
            model.group_similarity(dataset.groups_per_indiv, dataset.indivs_per_group, verbose=True)
            print("\nESTIMATED COEFFICIENTS:\n",model.beta_hat)
        else:
            model.fit(x,y,verbose=False)

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


