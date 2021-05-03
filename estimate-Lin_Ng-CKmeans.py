# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 16:02:00 2021

@author: Marco
"""

import pandas as pd
import numpy as np
from scipy.linalg import lstsq

from simulate import Effects, Slopes, Variance, Dataset
from lib.tracktime import TrackTime, TrackReport


class CK_means:
    def __init__(self):
        pass


    def estimate_G(self, G):
        self.G = G

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        #demean data:
        self.X = X - X.groupby('n').mean()
        self.Y = Y - Y.groupby('n').mean()

        self.N = len(set(self.X.index.get_level_values(0)))
        self.T = len(set(self.X.index.get_level_values(1)))
        self.K = len(self.X.columns)



np.random.seed(0)
N = 250
T = 100
K = 1


TrackTime("Simulate")
dataset = Dataset(N, T, K, G=3)
dataset.simulate(Effects.ind_fix, Slopes.heterog, Variance.homosk)


TrackTime("Estimate")
x = dataset.data.drop(["y"], axis=1)
y = dataset.data["y"]

ck_means = CK_means()
ck_means.estimate_G(dataset.G)    #assume true value of G is known
ck_means.fit(x,y)

#TODO: pseudo.predict()


TrackTime("Print")

print("\n\nTRUE COEFFICIENTS:")
print(dataset.slopes_df)

print("\n\nESTIMATED COEFFICIENTS:")
# print(ck_means.beta)


# from linearmodels import PanelOLS
# model_fe = PanelOLS(y, x, entity_effects = True)
# fe_res = model_fe.fit()
# print("\nFIXED EFFECTS ESTIMATION:"), print(fe_res.params)


print("\n")
TrackReport()
