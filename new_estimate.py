# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 20:26:36 2021

@author: Marco Deken
"""
import pandas as pd
import numpy as np
import os.path
import sys
import pickle
import matplotlib.pyplot as plt

from simulate import Effects, Slopes, Variance, Dataset, Fit


class Result:
    def __init__(self, slopes_true, slopes_ests, groups_true, groups_ests, alphas_ests_m, alphas_true_m, G_hats):
        self.M = len(groups_ests)
        self.N = len(groups_ests[0])
        self.K = len(slopes_true)
        self.G = len(slopes_true.columns)

        self.G_MAX = np.max(groups_ests)+1
        self.slopes_ests = slopes_ests[:,:,:self.G_MAX]
        self.slopes_true = slopes_true

        self.groups_ests = groups_ests
        self.groups_true = groups_true

        self.RMSE_alpha = self.calc_RMSE_alpha(alphas_ests_m, alphas_true_m)

        self.G_hats = G_hats


    def calc_RMSE_alpha(self, alphas_ests_m, alphas_true_m):
        RMSE_alpha = 0.0
        T = alphas_true_m.shape[1]
        for m in range(self.M):
            for i in range(self.N):
                  difference = alphas_ests_m[m,i,:] - alphas_true_m[i,:]
                  RMSE_alpha += difference @ difference.T
        RMSE_alpha = np.sqrt(RMSE_alpha / (self.N*T*self.M))
        return RMSE_alpha


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

        row = ["k=%d g=%d"%(k,g) for k in range(self.K) for g in range(self.G_MAX)]
        col = ["Lower CI", "Upper CI", "Mean", "St. dev."]
        self.summary = pd.DataFrame(np.vstack((lower_ci,upper_ci,mean,stdev)).T, columns=col, index=row)



def estimate_model(model, dataset, N, T, K, M, G_max, fit, model_name, filename, G_min=1):

    slopes_ests = np.zeros((M, K, G_max))
    groups_ests = np.zeros((M,N), dtype=int)
    if model_name == "gfe":
        alphas_ests = np.zeros((M, T, G_max))
    else:
        alphas_ests = np.zeros((M, N))
    G_hats = np.zeros(M, dtype=int)

    bar_length = 30
    for m in range(M):
        if (m) % 1 == 0:
            percent = 100.0*m/(max(1,M-1))
            sys.stdout.write("\rExperiment progress: [{:{}}] {:>3}%".format('='*int(percent/(100.0/bar_length)),bar_length, int(percent)))
            sys.stdout.flush()

        dataset.re_simulate()

        x = dataset.data.drop(["y"], axis=1)
        y = dataset.data["y"]

        if fit == Fit.groups_known:
            # ASSUME TRUE GROUP MEMBERSHIP IS KNOWN
            model.fit_given_groups(x, y, dataset.groups_per_indiv, first_fit=True, verbose=False)

        elif fit == Fit.G_known:
            # ASSUME TRUE VALUE OF G IS KNOWN
            model.set_G(dataset.G)
            model.fit(x,y,verbose=False)

        elif fit == Fit.complete:
            # ESTIMATE EVERYTHING
            best_groups = model.estimate_G(G_max, x, y, G_min)
            model.fit_given_groups(x, y, best_groups, first_fit=False, verbose=False)
            G_hats[m] = model.G_hat

        slopes_ests[m,:,:] = np.hstack((model.beta_hat.values,np.zeros((K, G_max-model.G))))
        groups_ests[m,:]   = model.groups_per_indiv
        if model_name == "gfe":
            alphas_ests[m,:,:] = np.hstack((model.alpha_hat.values.T,np.zeros((T, G_max-model.G))))
        else:
            alphas_ests[m,:] = model.alpha_hat.values.reshape(model.N)

    # Preparing fixed effects for calculation of RMSE in Result()
    alphas_ests_m = np.zeros((M, N, T))
    alphas_true_m = np.zeros((N, T))
    for i in range(N):
        alphas_true_m[i,:] = (dataset.effects_df.values[dataset.groups_per_indiv[i],:] if dataset.effects == Effects.gr_tvar_fix else dataset.effects_df.values[i])
        for m in range(M):
            alphas_ests_m[m,i,:] = (alphas_ests[m,:,groups_ests[m,i]] if model_name == "gfe" else alphas_ests[m,i])

    # Initialize Result object to save the results
    result = Result(dataset.slopes_df, slopes_ests, dataset.groups_per_indiv, groups_ests, alphas_ests_m, alphas_true_m, G_hats)

    with open(filename, 'wb') as output:
        pickle.dump(result, output, pickle.HIGHEST_PROTOCOL)




def main():
    from lib.tracktime import TrackTime, TrackReport
    from Bon_Man_GFE import GFE
    from Lin_Ng_PSEUDO import PSEUDO
    from Lin_Ng_CKmeans import CK_means
    np.random.seed(0)

    model_names = ["gfe", "ckmeans", "pseudo"]
    G = 4
    K = 1
    M = 100

    fit = Fit.G_known
    var = Variance.homosk
    DoF = 0
    G_max = G if fit == Fit.G_known else G+4

    # B = np.array([[0.2, 0.5, 0.8], [-0.3, 0.0, 0.3]])
    B = np.array([[0.35, 0.5, 0.65, 0.8]])

    col = ['g=%d'%i for i in range(G)]
    row = ['k=%d'%i for i in range(K)]
    slopes_df = pd.DataFrame(B, columns=col, index=row)


    select = [0,1,2]  #gfe ; ckmeans ; pseudo
    train = True
    overwrite = False

    N_range = [50, 100, 200]
    T_range = [5, 10, 20, 50, 100]
    for N in N_range:
        for T in T_range:
            dataset = Dataset(N, T, K, G=G)
            for i in select:
                model_name = model_names[i]
                filename = "%s/%s_N=%d_T=%d_G=%d_K=%d_M=%d_fit=%d_e=%d" %(model_name,model_name,N,T,G,K,M,fit.value,var.value)
                TrackTime(filename[len(model_name)+1:]+"\t\t")
                print("\n"+model_name.upper())

                if not continue_estimation(filename, fit, train, overwrite, load_results):
                    sys.exit(0)

                if model_name == "gfe":
                    model = GFE(Slopes.heterog)
                    dataset.simulate(Effects.gr_tvar_fix, Slopes.heterog, var, slopes_df, DoF)
                elif model_name == "ckmeans":
                    model = CK_means()
                    dataset.simulate(Effects.ind_fix, Slopes.heterog, var, slopes_df, DoF)
                elif model_name == "pseudo":
                    model = PSEUDO()
                    dataset.simulate(Effects.ind_fix, Slopes.heterog, var, slopes_df, DoF)
                else:
                    sys.exit(0)

                estimate_model(model, dataset, N, T, K, M, G_max, fit, model_name, filename)

            print("\n")
            TrackReport()



def load_results(filename, fit):
    if os.path.isfile(filename):
        with open(filename, 'rb') as output:
            result = pickle.load(output)

    print("\n\nTRUE COEFFICIENTS:")
    print(result.slopes_true)

    result.RMSE()
    print("\nRMSE: %.4f" %(result.RMSE*100))

    print("RMSE_alpha: %.4f\n" %(result.RMSE_alpha*100))

    if fit == Fit.complete:
        plt.hist(result.G_hats)
        plt.title(filename)
        plt.show()
    else:
        result.confusion_mat_groups()
        print(result.conf_mat, "\n")

        result.conf_interval(0.05)
        print(result.summary)



def continue_estimation(filename, fit, train, overwrite, load_results):
    if not train and os.path.isfile(filename):
        load_results(filename, fit)
        return False
    else:
        if os.path.isfile(filename) and not overwrite:
            print(r"THIS FILE ALREADY EXISTS, ARE YOU SURE YOU WANT TO OVERWRITE? Y\N")
            if not input().upper() == "Y":
                return False

    return True



if __name__ == "__main__":
    main()