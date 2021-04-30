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

np.random.seed(0)
N = 100
T = 100
K = 4

TrackTime("Simulate")
dataset = Dataset(N, T, K, G=2)
dataset.simulate(Effects.ind_fix, Slopes.heterog, Variance.homosk)

#TODO: Lin and Ng estimation
TrackTime("Estimate")
G = dataset.G   #assume true value of G is known

x = dataset.data.drop(["y"], axis=1)
y = dataset.data["y"]

x = x - x.groupby('n').mean()
y = y - y.groupby('n').mean()

# print(np.all(np.abs(x.groupby('n').mean()) < 10**-10))
# print(np.all(np.abs(y.groupby('n').mean()) < 10**-10))


TrackTime("Calc q_hat")
q_hat = np.zeros((N,K))
for i in range(N):
    q_hat[i,:], _, _, _ = lstsq(x.loc[i,:], y.loc[i,:])


def squared_residuals(gamma, q_hat, k):
    partition = [np.arange(N)[q_hat[:,k] <= gamma].reshape(1,-1)[0], np.arange(N)[q_hat[:,k] > gamma].reshape(1,-1)[0]]

    x1 = x.loc[partition[0],:]
    x2 = x.loc[partition[1],:]
    y1 = y.loc[partition[0],:]
    y2 = y.loc[partition[1],:]

    beta_hat_1, _, _, _ = lstsq(x1, y1)
    beta_hat_2, _, _, _ = lstsq(x2, y2)

    residuals1 = (y1 - x1 @ beta_hat_1).values
    residuals2 = (y2 - x2 @ beta_hat_2).values

    return residuals1@residuals1.T + residuals2@residuals2.T

TrackTime("Prepare")
min_group_size = 10

gamma_range = np.zeros(N-1-2*(min_group_size-1))

gamma_star = [np.Inf, 0, 0]
for k in range(K):

    q_hat_sorted = np.sort(q_hat[:,k])
    
    for i in range(len(gamma_range)):
        gamma_range[i] = (q_hat_sorted[min_group_size-1+i] + q_hat_sorted[min_group_size+i])/2
    
    TrackTime("Calc gamma_star")
    # gamma_star = [np.Inf, 0]
    for gamma in gamma_range:
        ssr = squared_residuals(gamma, q_hat, k)
        if ssr < gamma_star[0]:
            gamma_star[0] = ssr
            gamma_star[1] = gamma
            gamma_star[2] = k

print(gamma_star)
TrackTime("Calc beta_hat")
gamma = gamma_star[1]
partition = [np.arange(N)[q_hat[:,k] <= gamma].reshape(1,-1)[0], np.arange(N)[q_hat[:,k] > gamma].reshape(1,-1)[0]]

x1 = x.loc[partition[0],:]
x2 = x.loc[partition[1],:]
y1 = y.loc[partition[0],:]
y2 = y.loc[partition[1],:]

beta_hat_1, _, _, _ = lstsq(x1, y1)
beta_hat_2, _, _, _ = lstsq(x2, y2)



TrackTime("Print")

print("TRUE COEFFICIENTS:")
print(dataset.slopes_df)

print("\n\nESTIMATED COEFFICIENTS:")
print(beta_hat_1)
print(beta_hat_2)


print("\n")
TrackReport()