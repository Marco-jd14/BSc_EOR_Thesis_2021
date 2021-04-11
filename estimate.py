# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11  14:34:23 2021

@author: Marco
"""
import pandas as pd
import numpy as np
import enum

from simulate import Effects, Slopes, Variance, Dataset

#TODO: constant / intercept

np.random.seed(0)
dataset = Dataset(N=200, T=9, K=1, G=10)
dataset.simulate(Effects.ind_fix, Slopes.homog, Variance.homosk)


# dataset = pd.read_csv('guns.csv', usecols = ['n', 't', 'feature0', 'y'], index_col = ['n', 't'])



# Perform PooledOLS
from linearmodels import PooledOLS
import statsmodels.api as sm

exog = sm.tools.tools.add_constant(dataset.data).drop(['y'], axis=1)
exog = dataset.data.drop(['y'], axis=1)
endog = dataset.data['y']
mod = PooledOLS(endog, exog)
pooledOLS_res = mod.fit(cov_type='clustered', cluster_entity=True)
# Store values for checking homoskedasticity graphically
fittedvals_pooled_OLS = pooledOLS_res.predict().fitted_values
residuals_pooled_OLS = pooledOLS_res.resids


# 3A. Homoskedasticity
import matplotlib.pyplot as plt

 # 3A.1 Residuals-Plot for growing Variance Detection
fig, ax = plt.subplots()
ax.scatter(fittedvals_pooled_OLS, residuals_pooled_OLS, color = 'blue')
ax.axhline(0, color = 'r', ls = '--')
ax.set_xlabel('Predicted Values', fontsize = 15)
ax.set_ylabel('Residuals', fontsize = 15)
ax.set_title('Homoskedasticity Test', fontsize = 30)
plt.show()

# 3A.2 White-Test
from statsmodels.stats.diagnostic import het_white, het_breuschpagan
exog = sm.tools.tools.add_constant(dataset.data).drop(['y'], axis=1).fillna(0)
white_test_results = het_white(residuals_pooled_OLS, exog)
labels = ['LM-Stat', 'LM p-val', 'F-Stat', 'F p-val'] 
print(dict(zip(labels, white_test_results)), '\n')

# 3A.3 Breusch-Pagan-Test
breusch_pagan_test_results = het_breuschpagan(residuals_pooled_OLS, exog)
labels = ['LM-Stat', 'LM p-val', 'F-Stat', 'F p-val']  
print(dict(zip(labels, breusch_pagan_test_results)), '\n')


# 3.B Non-Autocorrelation
# Durbin-Watson-Test
from statsmodels.stats.stattools import durbin_watson

durbin_watson_test_results = durbin_watson(residuals_pooled_OLS) 
print("Durbin watson test:", durbin_watson_test_results, '\n')


# FE und RE model
from linearmodels import PanelOLS
from linearmodels import RandomEffects
exog = sm.tools.tools.add_constant(dataset.data).drop(['y'], axis=1)
exog = dataset.data.drop(['y'], axis=1)
endog = dataset.data['y']

# random effects model
model_re = RandomEffects(endog, exog)
re_res = model_re.fit()

# fixed effects model
model_fe = PanelOLS(endog, exog, entity_effects = True)
fe_res = model_fe.fit()

#print results
# print(re_res)
# print('\n')
# print(fe_res)

# print(fe_res.params)

#plot results
feat_dim = exog['feature0']
plt.plot(feat_dim, endog, 'ok')
plt.plot(feat_dim, fe_res.predict().fitted_values, 'o', markersize=1)
plt.show()


#compare RE with FE
import numpy.linalg as la
from scipy import stats

def hausman(fe, re):
    b = fe.params
    B = re.params
    v_b = fe.cov
    v_B = re.cov

    df = b[np.abs(b) < 1e8].size
    chi2 = np.dot((b - B).T, la.inv(v_b - v_B).dot(b - B)) 

    pval = stats.chi2.sf(chi2, df)
    return chi2, df, pval

hausman_results = hausman(fe_res, re_res) 
print('Hausman test results:')
print('\tchi-Squared: ' + str(hausman_results[0]))
print('\tdegrees of freedom: ' + str(hausman_results[1]))
print('\tp-Value: ' + str(hausman_results[2]))


print("\n\n\nPOOLED OLS:"), print(pooledOLS_res.params)
print("\nRANDOM EFFECTS:"), print(re_res.params)
print("\nFIXED EFFECTS:"), print(fe_res.params)
print("\nTRUE COEFFICIENTS:"), print(dataset.slopes_df)