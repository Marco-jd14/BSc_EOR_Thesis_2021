# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 11:02:00 2021

@author: Marco
"""

import pandas as pd
import numpy as np
import os.path
import sys
import pickle
from scipy.linalg import lstsq

from simulate import Effects, Slopes, Variance, Dataset
from lib.tracktime import TrackTime, TrackReport


class PSEUDO:
    def __init__(self):
        pass


    def _cluster_regressor(self, q_hat, left_n, verbose):
        gamma_star = [np.Inf, np.Inf, 0]

        nr_gamma_options = len(q_hat)-1-2*(self.min_group_size-1)
        if nr_gamma_options <= 0:
            if verbose:
                print("Too few group members to split:", len(q_hat))
            return gamma_star

        gamma_range = np.zeros(nr_gamma_options)
        q_hat_sorted = np.sort(q_hat)

        for i in range(nr_gamma_options):
            gamma_range[i] = (q_hat_sorted[self.min_group_size-1+i] + q_hat_sorted[self.min_group_size+i])/2

        for gamma in gamma_range:
            ssr_left, ssr_right = self._ssr(gamma, q_hat, left_n)
            if ssr_left + ssr_right < gamma_star[0] + gamma_star[1]:
                gamma_star[0] = ssr_left
                gamma_star[1] = ssr_right
                gamma_star[2] = gamma

        return gamma_star


    def _ssr(self, gamma, q_hat, left_n):
        partition = [np.arange(len(q_hat))[q_hat <= gamma]+left_n, np.arange(len(q_hat))[q_hat > gamma]+left_n]

        # TrackTime("Partition indices")
        partition_indices = [np.zeros(len(partition[0])*self.T,dtype=int), np.zeros(len(partition[1])*self.T, dtype=int)]
        for k in range(2):
            for i in range(len(partition[k])):
                partition_indices[k][i*self.T:(i+1)*self.T] = np.arange(self.T) + partition[k][i]*self.T

        # TrackTime("Partition selection")
        x1 = self.X.values[partition_indices[0],:] # ~35x faster than  x1 = self.X.loc[partition[0],:]
        x2 = self.X.values[partition_indices[1],:]
        y1 = self.Y.values[partition_indices[0]]
        y2 = self.Y.values[partition_indices[1]]

        # TrackTime("gamma SSR")
        beta_hat_1, _, _, _ = lstsq(x1, y1)
        beta_hat_2, _, _, _ = lstsq(x2, y2)

        # TrackTime("Estimate")
        residuals1 = y1 - x1 @ beta_hat_1
        residuals2 = y2 - x2 @ beta_hat_2

        return residuals1@residuals1.T, residuals2@residuals2.T


    def _split_groups_in_half(self, i, gamma_stars, q_hat_k, verbose):
        if i==0:
            new_gammas = [self._cluster_regressor(q_hat_k, 0, verbose)]
        else:
            left_n = 0
            new_gammas = []
            for j in range(len(gamma_stars)+1):
                if j==0:
                    rel_q_hat = q_hat_k[q_hat_k <= gamma_stars[j][2]]
                elif j==len(gamma_stars):
                    rel_q_hat = q_hat_k[q_hat_k > gamma_stars[j-1][2]]
                else:
                    rel_q_hat = q_hat_k[np.all([q_hat_k > gamma_stars[j-1][2], q_hat_k <= gamma_stars[j][2]],axis=0)]

                gamma_star = self._cluster_regressor(rel_q_hat, left_n, verbose)
                new_gammas.append(gamma_star)
                left_n += len(rel_q_hat)

        gamma_stars = self._update_gamma_stars(i, new_gammas, gamma_stars)

        return gamma_stars


    def _update_gamma_stars(self, i, new_gammas, gamma_stars):
        if 2**(i+1) <= self.G:
            for j in range( len(gamma_stars) ):
                gamma_stars[j][0] = 0
                gamma_stars[j][1] = 0
            for j in range( len(gamma_stars)+1 ):
                gamma_stars.insert(2*j, new_gammas[j])

        else:
            #splitting each group in half leads to too many groups --> pick best groups to split in half
            improvements = np.zeros((len(new_gammas),2))
            for j in range(len(new_gammas)):
                improvements[j,0] = gamma_stars[int(j/2)*2][j%2] / (new_gammas[j][0]+new_gammas[j][1])
                improvements[j,1] = j

            best_new_gamma_indices = np.zeros(self.G - 2**i,dtype=int)
            best_new_gamma_indices[:] = improvements[np.argsort(improvements[:,0])][-(self.G - 2**i):,1]

            for index in best_new_gamma_indices:
                gamma_stars[int(index/2)*2][index%2] = 0
            for index in best_new_gamma_indices:
                #find right place to insert the new_gammas in the gamma_stars list to keep it sorted
                for j in range(len(gamma_stars)):
                    if gamma_stars[j][2] > new_gammas[index][2]:
                        gamma_stars.insert(j,new_gammas[index])
                        break
                    if j == len(gamma_stars)-1:
                        gamma_stars.append(new_gammas[index])

        return gamma_stars


    def _calc_k_star(self, gamma_stars_per_k):
        k_star = [np.Inf, 0]
        for k in range(self.K):
            tot_ssr = 0
            for gamma_star in gamma_stars_per_k[k]:
                tot_ssr += (gamma_star[0] + gamma_star[1])

            if tot_ssr < k_star[0]:
                k_star[0] = tot_ssr
                k_star[1] = k

        return k_star[1]


    def _final_estimate(self, gamma_stars, q_hat):
        # TrackTime("Calc beta_hats")

        self.groups_per_indiv = np.zeros(self.N, dtype=int)
        self.alpha_hat = np.zeros(self.N)
        betas = np.zeros((self.K, self.G))
        for g in range(len(gamma_stars)+1):
            if g==0 and g==len(gamma_stars):
                selection = np.arange(self.N)
            elif g==0:
                selection = np.arange(self.N)[q_hat <= gamma_stars[g][2]]
            elif g==len(gamma_stars):
                selection = np.arange(self.N)[q_hat > gamma_stars[g-1][2]]
            else:
                selection = np.arange(self.N)[np.all([q_hat > gamma_stars[g-1][2], q_hat <= gamma_stars[g][2]],axis=0)]

            self.groups_per_indiv[selection] = g

            selection_indices = np.zeros(len(selection)*self.T, dtype=int)
            for i in range(len(selection)):
                selection_indices[i*self.T:(i+1)*self.T] = np.arange(self.T) + selection[i]*self.T

            x_sel = self.X.values[selection_indices,:]
            y_sel = self.Y.values[selection_indices]

            betas[:,g], _, _, _ = lstsq(x_sel, y_sel)

            # Estimate individual fixed effects
            self.alpha_hat[selection] = self.y_bar.values[selection] - self.x_bar.values[selection,:] @ betas[:,g]

        self._make_dataframes(betas)


    def _make_dataframes(self, betas):
        self.indivs_per_group = [[] for g in range(self.G)]
        for i in range(self.N):
            self.indivs_per_group[self.groups_per_indiv[i]].append(i)

        col = ['g=%d'%i for i in range(self.G)]
        row = ['k=%d'%i for i in range(self.K)]
        self.beta_hat = pd.DataFrame(betas, columns=col, index=row)

        col = ['t=%d'%i for i in range(1)]
        row = ['n=%d'%i for i in range(len(self.alpha_hat))]
        self.alpha_hat = pd.DataFrame(self.alpha_hat, columns=col, index=row)


    def set_G(self, G):
        self.G = G

    def estimate_G(self, G_max, X, Y):
        BIC_G = np.zeros(G_max)
        for g in range(G_max):
            self.set_G(g+1)
            try:
                self.fit(X, Y, verbose=False)
                self.predict(verbose=False)
                SSR = self.resids.values @ self.resids.values.T
                NT = self.N*self.T
                BIC_G[g]  = np.log(SSR/NT) + g*self.K * np.sqrt(min(self.N, self.T)) * np.log(NT) / NT + (g-1) * np.log(self.N**2)/self.N**2
            except:
                BIC_G[g] = np.Inf

        self.G_hat = np.argmin(BIC_G)+1
        col = ['G=%d'%(g+1) for g in range(G_max)]
        col[self.G_hat-1] = 'G_hat=%d'%self.G_hat
        self.BIC = pd.DataFrame(BIC_G.reshape(1,-1), columns=col, index=["BIC"])


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
        self.min_group_size = 10
        self.nr_iterations = self.K

        q_hat = np.zeros((self.N,self.K))
        for i in range(self.N):
            select = np.arange(i*self.T,(i+1)*self.T)
            q_hat[i,:], _, _, _ = lstsq(self.X.values[select], self.Y.values[select])

        gamma_stars_per_k = [[] for i in range(self.K)]
        for k in range(self.K):
            for i in range(int(np.ceil(np.log2(self.G)))):
                gamma_stars_per_k[k] = self._split_groups_in_half(i, gamma_stars_per_k[k], q_hat[:,k], verbose)

        k_star = self._calc_k_star(gamma_stars_per_k)

        self._final_estimate(gamma_stars_per_k[k_star], q_hat[:,k_star])


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
        fitted_values2 = np.zeros_like(self.Y)

        X = self.X + self.x_bar

        for g in range(self.G):
            selection = np.where(self.groups_per_indiv == g)[0]
            if len(selection) == 0:
                if verbose:
                    print("GROUP %d OUT OF %d IS EMPTY"%(g,self.G))
                continue

            selection_indices = np.zeros(len(selection)*self.T, dtype=int)
            for i in range(len(selection)):
                selection_indices[i*self.T:(i+1)*self.T] = np.arange(self.T) + selection[i]*self.T

            fixed_effects = np.kron(self.alpha_hat.values[selection].reshape(len(selection)),np.ones(self.T))
            self.fitted_values[selection_indices] = X.values[selection_indices,:] @ self.beta_hat.values[:,g] + fixed_effects
            fitted_values2[selection_indices] = self.X.values[selection_indices,:] @ self.beta_hat.values[:,g]

        self.resids = self.Y + self.y_bar - self.fitted_values
        self.resids2 = self.Y - fitted_values2
        index = pd.MultiIndex.from_product([np.arange(self.N), np.arange(self.T)], names=["n", "t"])
        self.fitted_values = pd.DataFrame(self.fitted_values, index=index)



def main():
    from estimate import plot_residuals, plot_fitted_values, plot_clusters
    from Bon_Man import Result

    # np.random.seed(0)
    N = 100
    T = 50
    G = 5
    K = 2
    M = 100
    filename = "pseudo/pseudo_N=%d_T=%d_G=%d_K=%d_M=%d" %(N,T,G,K,M)

    train = 0
    if not train:
        load_results(filename)
        sys.exit(0)
    # else:
    #     if os.path.isfile(filename):
    #         print(r"THIS FILE ALREADY EXISTS, ARE YOU SURE YOU WANT TO OVERWRITE? Y\N")
    #         if not input().upper() == "Y":
    #             sys.exit(0)


    # B = np.array([[0.3, 0.9]])
    # B = np.array([[0.3, 0.5, 0.8]])
    # B = np.array([[0.1, 2/3], [0.3, 0.6]])
    # B = np.array([[0.3, 0.5, 0.7], [-0.3, 0.0, 0.3]])
    B = np.array([[0.1, 0.3, 0.5, 0.7, 0.9], [0.1, 0.2, 0.3, 0.4, 0.5]])

    # B = np.array([[0.55, 0.65]])
    # B = np.array([[0.4, 0.5, 0.8]])
    # B = np.array([[0.3, 0.4], [0.4, 0.5]])
    # B = np.array([[0.4, 0.5, 0.6], [0.2, 0.3, 0.4]])
    # B = np.array([[0.3, 0.4, 0.5, 0.6, 0.7], [0.1, 0.2, 0.3, 0.4, 0.5]])
    col = ['g=%d'%i for i in range(G)]
    row = ['k=%d'%i for i in range(K)]
    slopes_df = pd.DataFrame(B, columns=col, index=row)


    TrackTime("Simulate")
    dataset = Dataset(N, T, K, G=G)
    dataset.simulate(Effects.ind_fix, Slopes.heterog, Variance.homosk, slopes_df)
    model = PSEUDO()

    G_max = int(N/12)
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
        # model.set_G(dataset.G)    #assume true value of G is known
        model.estimate_G(G_max, x, y)
        print("G_hat =",model.G_hat)
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

    if M > 0:
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

