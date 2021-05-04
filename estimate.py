# -*- coding: utf-8 -*-
"""
Created on Tue May  4 21:58:41 2021

@author: Marco
"""

import pandas as pd
import numpy as np
from copy import copy
from scipy.linalg import lstsq
import matplotlib.pyplot as plt

from simulate import Effects, Slopes, Variance, Dataset
from lib.tracktime import TrackTime, TrackReport
from Bon_Man import GFE
from Lin_Ng_PSEUDO import PSEUDO
from Lin_Ng_CKmeans import CK_means


def plot_residuals(fitted_values, residuals, title):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(fitted_values, residuals, color = 'blue', s=2)
    ax.axhline(0, color = 'r', ls = '--')
    ax.set_xlabel('Predicted Values', fontsize = 12)
    ax.set_ylabel('Residuals', fontsize = 12)
    ax.set_title(title, fontsize=15)
    plt.show()

def plot_fitted_values(feat_dim, true_values, fitted_values, title):
    fitted_values = fitted_values.values if isinstance(fitted_values, pd.DataFrame) else fitted_values
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(feat_dim, true_values, 'ok')
    ax.plot(feat_dim, fitted_values, 'o', markersize=1)
    ax.set_xlabel('First feature', fontsize = 12)
    ax.set_title(title, fontsize=15)
    ax.legend(['True values', 'Fitted values'])
    plt.show()


def main():
    np.random.seed(0)
    N = 250
    T = 100
    K = 3

    #TODO: Make comments
    #TODO: estimate G

    TrackTime("Simulate")
    dataset = Dataset(N, T, K, G=3)
    dataset.simulate(Effects.ind_fix, Slopes.heterog, Variance.homosk)


    x = dataset.data.drop(["y"], axis=1)
    y = dataset.data["y"]

    models = [CK_means(), PSEUDO(), GFE(Slopes.heterog)]
    model_names = ["CK_means", "PSEUDO", "GFE"]

    select = [2, 0]
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