# -*- coding: utf-8 -*-
"""
Created on Tue May  4 21:58:41 2021

@author: Marco
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from simulate import Effects, Slopes, Variance, Dataset
from lib.tracktime import TrackTime, TrackReport
from Bon_Man import GFE
from Lin_Ng_PSEUDO import PSEUDO
from Lin_Ng_CKmeans import CK_means


def plot_residuals(fitted_values, residuals, title):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(fitted_values, residuals, color = 'r', s=2, alpha=0.5)
    ax.axhline(0, color = 'k', ls = '--')
    ax.set_xlabel('Predicted Values', fontsize = 12)
    ax.set_ylabel('Residuals', fontsize = 12)
    ax.set_title(title, fontsize=15)
    plt.show()

def plot_fitted_values(feat_dim, true_values, fitted_values, title):
    fitted_values = fitted_values.values if isinstance(fitted_values, pd.DataFrame) else fitted_values
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(feat_dim, true_values, 'ok', markersize=3, alpha=0.7)
    ax.plot(feat_dim, fitted_values, 'or', markersize=1, alpha=0.7)
    ax.set_xlabel('First feature', fontsize = 12)
    ax.set_title(title, fontsize=15)
    ax.legend(['True values', 'Fitted values'])
    plt.show()

def plot_clusters(feat_dim, y, title):
    # color_map = ["#F4F6CC", "#B9D5B2", "#84B29E", "#568F8B", "#326B77", "#1A445B", "#122740", "#000812"]
    # color_map = ["#000812", "#122740", "#1A445B", "#326B77", "#568F8B", "#84B29E", "#B9D5B2", "#F4F6CC"]
    fig, axs = plt.subplots(2, figsize=(10,8), sharex=True)
    fig.suptitle(title, fontsize=15)

    # axs[0].set_prop_cycle(color=color_map)#,marker=['o','+','x','o','+','x','o','+'])
    axs[0].set_title('True groups', fontsize=12)
    axs[1].set_title('Estimated groups', fontsize=12)
    axs[1].set_xlabel('First feature', fontsize = 12)

    for g in set(y['true_label'].astype(dtype=int)):
        selection = (y['true_label']==g).values
        axs[0].plot(feat_dim[selection], y.iloc[selection,0], 'o', markersize=2, alpha=0.7)

    for g in set(y['est_label'].astype(dtype=int)):
        selection = (y['est_label']==g).values
        axs[1].plot(feat_dim[selection], y.iloc[selection,0], 'o', markersize=2, alpha=0.7)

    lgnd1 = axs[0].legend(set(y['true_label'].astype(dtype=int)), prop={'size': 10})
    lgnd2 = axs[1].legend(set(y['est_label'].astype(dtype=int)), prop={'size': 10})
    for handle in lgnd1.legendHandles:
        handle._legmarker.set_markersize(6)
    for handle in lgnd2.legendHandles:
        handle._legmarker.set_markersize(6)
    plt.show()


def main():
    # Worst case: N=250, T=50, K=1, G=6, seed(0), gr_tvar_fix:   talk about different ways to enforce nonempty groups
    np.random.seed(0)
    N = 200
    T = 50
    K = 1

    #TODO: Make comments
    #TODO: estimate G
    #TODO: Negative slope coefficients?

    TrackTime("Simulate")
    dataset = Dataset(N, T, K, G=6)
    dataset.simulate(Effects.ind_fix, Slopes.heterog, Variance.heterosk)


    x = dataset.data.drop(["y"], axis=1)
    y = pd.DataFrame(dataset.data["y"],columns=['y'])

    models = [CK_means(), PSEUDO(), GFE(Slopes.heterog)]
    model_names = ["CK_means", "PSEUDO", "GFE"]

    select = [2,0,1]
    for i in range(len(select)):
        for j in select:
            TrackTime(model_names[j])

        print("\n%s:" %model_names[select[i]])
        TrackTime(model_names[select[i]])
        model = models[select[i]]

        model.estimate_G(dataset.G)    #assume true value of G is known
        model.fit(x,y)
        model.predict()

        TrackTime("Plot")
        plot_residuals(model.fitted_values, model.resids, model_names[select[i]])
        plot_fitted_values(x['feature0'], y, model.fitted_values, model_names[select[i]])
        y['true_label'] = np.kron(dataset.groups_per_indiv,np.ones(dataset.T))
        y['est_label'] = np.kron(model.groups_per_indiv,np.ones(dataset.T))
        plot_clusters(x['feature0'], y, model_names[select[i]])

        TrackTime("Print")
        print("%s TOOK %s ITERATIONS"%(model_names[select[i]], model.nr_iterations))
        model.group_similarity(dataset.groups_per_indiv, dataset.indivs_per_group, verbose=False)


    print("\n")
    col = dataset.slopes_df.columns
    index = pd.MultiIndex.from_product([["TRUE VALUES"] + [model_names[i] for i in select], ['k=%d'%i for i in range(dataset.K)]])#, names=["model", "k"])
    final_df = dataset.slopes_df.values

    for i in range(len(select)):
        model = models[select[i]]
        final_df = np.vstack([final_df, model.beta_hat.values])
    final_df = pd.DataFrame(final_df, columns=col, index=index)

    s = final_df.to_string()
    for name in list(model_names[i] for i in select):
        s = s.replace('\n%s'%name, '\n'+'-'*(15+10*dataset.G)+'\n%s'%name)
    print(s)



    # from linearmodels import PanelOLS
    # model_fe = PanelOLS(y, x, entity_effects = True)
    # fe_res = model_fe.fit()
    # print("\nFIXED EFFECTS ESTIMATION:"), print(fe_res.params)


    print("\n")
    TrackReport()


if __name__ == "__main__":
    main()