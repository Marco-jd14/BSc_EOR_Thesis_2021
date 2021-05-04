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


def plot_residuals(fitted_values, residuals):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(fitted_values, residuals, color = 'blue', s=2)
    ax.axhline(0, color = 'r', ls = '--')
    ax.set_xlabel('Predicted Values', fontsize = 12)
    ax.set_ylabel('Residuals', fontsize = 12)
    plt.show()

def plot_fitted_values(feat_dim, true_values, fitted_values):
    fitted_values = fitted_values.values if isinstance(fitted_values, pd.DataFrame) else fitted_values
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(feat_dim, true_values, 'ok')
    ax.plot(feat_dim, fitted_values, 'o', markersize=1)
    ax.set_xlabel('First feature', fontsize = 12)
    ax.legend(['True values', 'Fitted values'])
    plt.show()


def main():
    np.random.seed(0)
    N = 250
    T = 50
    K = 2


    TrackTime("Simulate")
    dataset = Dataset(N, T, K, G=7)
    dataset.simulate(Effects.ind_fix, Slopes.heterog, Variance.homosk)


    TrackTime("Estimate")
    x = dataset.data.drop(["y"], axis=1)
    y = dataset.data["y"]

    # model = CK_means()
    # model = PSEUDO()
    model = GFE(Slopes.heterog)
    model.estimate_G(dataset.G)    #assume true value of G is known
    model.fit(x,y)
    model.predict()


    TrackTime("Plot")
    plot_residuals(model.fitted_values, model.resids)
    plot_fitted_values(x['feature0'], y, model.fitted_values)

    #TODO: Make comments

    #TODO: estimate G


    TrackTime("Print")

    #TODO: if model.hasattr
    # print("\n\nTOOK %s ITERATIONS\n"%model.nr_iterations)

    print("\n\nTRUE COEFFICIENTS:")
    print(dataset.slopes_df)
    # print(dataset.effects_df)
    # print(dataset.groups_per_indiv)
    # for group in dataset.indivs_per_group:
    #     print(group)

    print("\n\nESTIMATED COEFFICIENTS:")
    print(model.beta_hat)
    # print(model.alpha_hat)
    # print(model.groups_per_indiv)
    # for group in model.indivs_per_group:
    #     print(group)

    model.group_similarity(dataset.groups_per_indiv, dataset.indivs_per_group)


    # from linearmodels import PanelOLS
    # model_fe = PanelOLS(y, x, entity_effects = True)
    # fe_res = model_fe.fit()
    # print("\nFIXED EFFECTS ESTIMATION:"), print(fe_res.params)


    print("\n")
    TrackReport()


if __name__ == "__main__":
    main()