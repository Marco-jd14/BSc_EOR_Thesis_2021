# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 10:58:46 2021

@author: Marco
"""
import pandas as pd
import numpy as np

K = 0

def simulate():
    T = 23  #number of timepoints
    N = 50  #number of individuals
    global K
    K = 2   #number of explanatory variables

    G = np.ones((1,T))
    indiv_fixed_effects = np.random.uniform(0, 50, size=(N,1))
    # indiv_fixed_effects = np.random.normal(0, 50/2, size=(N,1))

    X_range = [10, 40]
    X = np.random.uniform(X_range[0], X_range[1], size=(N, T, K))
    X[:,:,0] += (indiv_fixed_effects @ G)

    # print(pd.DataFrame(np.hstack((indiv_fixed_effects,X[:,:,0]))).corr())


    B = np.random.uniform(0.5,3, size=(K))

    Y = X @ B
    Y += indiv_fixed_effects.reshape(N,1) @ G

    """
    heterosk = (X[:,:,0]/np.mean(X[:,:,0]))#/np.sqrt(K)
    corr = heterosk
    """
    homosk = np.ones((N,T))*3
    corr = homosk
    #"""

    errors = np.random.normal(0, np.sqrt(np.mean(Y))*corr)
    Y += errors

    index = pd.MultiIndex.from_product([np.arange(N), np.arange(T)], names=["n", "t"])
    features = ['feature%d'%i for i in range(K)]
    dataset = pd.DataFrame(np.hstack((Y.reshape(N*T,1), X.reshape(N*T,K))), columns=['y'] + features, index=index)

    fixed_effects = pd.DataFrame(indiv_fixed_effects)

    return dataset, fixed_effects, B


np.random.seed(0)
dataset, fixed_effects, true_coef = simulate()
print("True coefficients:", true_coef, '\n')
# print('\n')

# dataset = pd.read_csv('guns.csv', usecols = ['n', 't', 'feature0', 'y'], index_col = ['n', 't'])



# Perform PooledOLS
from linearmodels import PooledOLS
import statsmodels.api as sm

exog = sm.tools.tools.add_constant(dataset).drop(['y'], axis=1)
endog = dataset['y']
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
exog = sm.tools.tools.add_constant(dataset).drop(['y'], axis=1).fillna(0)
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
print(durbin_watson_test_results, '\n')


# FE und RE model
from linearmodels import PanelOLS
from linearmodels import RandomEffects
exog = sm.tools.tools.add_constant(dataset).drop(['y'], axis=1)
exog = dataset.drop(['y'], axis=1)
endog = dataset['y']

# random effects model
model_re = RandomEffects(endog, exog)
re_res = model_re.fit()

# fixed effects model
model_fe = PanelOLS(endog, exog, entity_effects = True)
fe_res = model_fe.fit()

#print results
print(re_res)
print('\n')
print(fe_res)

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
print('\n')
print('chi-Squared: ' + str(hausman_results[0]))
print('degrees of freedom: ' + str(hausman_results[1]))
print('p-Value: ' + str(hausman_results[2]))