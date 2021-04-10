# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 10:58:46 2021

@author: Marco
"""
import pandas as pd
import numpy as np
import enum


class Effects(enum.Enum):
    none = 0
    ind_rand = 1
    ind_fix = 2
    gr_tvar_fix = 3
    both_fix = 4

class Slopes(enum.Enum):
    homog = 0
    heterog = 1

class Variance(enum.Enum):
    homosk = 0
    heterosk = 1


class Dataset:
    def __init__(self, T: int, N: int, K: int, G=1):
        self.T = T
        self.N = N
        self.K = K
        self.G = G


    def sim_groups(self):
        individuals = np.arange(self.N)
        np.random.shuffle(individuals)

        group_sizes = np.zeros(self.G, dtype=int)
        for g in range(self.G):
            if g == self.G - 1:
                group_size = self.N - np.sum(group_sizes)
            else:
                group_size = int(np.round(np.random.uniform(0.75*self.N/self.G, 1.25*self.N/self.G), 0))
            group_sizes[g] = group_size

        self.groups_list = [[] for g in range(self.G)]
        for g in range(self.G):
            for i in range(group_sizes[g]):
                self.groups_list[g].append(individuals[ np.sum(group_sizes[:g]) + i ])

        self.groups_mat = np.zeros((self.G, self.N), dtype=int)
        for g in range(self.G):
            self.groups_mat[g, self.groups_list[g]] = 1


    def sim_effects(self):
        if self.effects == Effects.none:
            effects_m = np.zeros((self.N, self.T))

        elif self.effects == Effects.ind_rand:
            effects_m = np.random.normal(0, 50/2, size=(self.N, 1)) @ np.ones((1, self.T))

        elif self.effects == Effects.ind_fix:
            effects_m = np.random.uniform(0, 50, size=(self.N, 1)) @ np.ones((1, self.T))

        elif self.effects == Effects.gr_tvar_fix:
            group_effects = np.random.uniform(0, 50, size=(self.G, self.T))
            effects_m = self.groups_mat.T @ group_effects

        elif self.effects == Effects.both_fix:
            indiv_fix_eff = np.random.uniform(0, 50, size=(self.N, 1)) @ np.ones((1, self.T))
            group_fix_eff = np.random.uniform(0, 50, size=(self.G, self.T))
            effects_m = self.groups_mat.T @ group_fix_eff + indiv_fix_eff

        col = ['t=%d'%i for i in range(self.T)]
        row = ['n=%d'%i for i in range(self.N)]
        self.effects_df = pd.DataFrame(effects_m, columns=col, index=row)


    def sim_slopes(self):
        # if self.slopes == Slopes.homog:
        #     B = np.random.uniform(0.5,3, size=(self.K, self.G))
        # elif self.slopes == Slopes.heterog:
        B = np.random.uniform(0.5,5, size=(self.K, self.G))

        col = ['g=%d'%i for i in range(self.G)]
        row = ['k=%d'%i for i in range(self.K)]
        self.slopes_df = pd.DataFrame(B, columns=col, index=row)


    def simulate(self, effects: Effects, slopes: Slopes, var: Variance):
        self.effects = effects
        self.slopes = slopes
        self.var = var
        self.has_groups = (effects == Effects.gr_tvar_fix or effects == Effects.both_fix or slopes == Slopes.heterog)

        if self.has_groups:
            self.sim_groups()
            print("GROUPS:\n", self.groups_list)
        else:
            self.G = 1

        self.sim_effects()

        X_range = [10, 40]
        X = np.random.uniform(X_range[0], X_range[1], size=(self.N, self.T, self.K))
        if self.effects != Effects.ind_rand:
            X[:,:,0] += self.effects_df.values       #create correlation between regressor and ommitted variable (fixed effects)

        # print(pd.DataFrame(np.hstack((indiv_fixed_effects,X[:,:,0]))).corr())

        self.sim_slopes()
        if not self.has_groups:
            Y = X @ self.slopes_df.values.reshape(self.K)
        else:
            temp = X @ self.slopes_df.values
            Y = np.zeros((self.N, self.T))
            for g in range(self.G):
                Y += temp[:,:,g] * self.groups_mat.T[:,g].reshape(self.N, 1)

        Y += self.effects_df.values

        if self.var == Variance.heterosk:
            heterosk = (X[:,:,0]/np.mean(X[:,:,0])) #/np.sqrt(K)
            corr = heterosk
        elif self.var == Variance.homosk:
            homosk = np.ones((self.N, self.T))*3
            corr = homosk

        errors = np.random.normal(0, np.sqrt(np.mean(Y))*corr)
        Y += errors

        index = pd.MultiIndex.from_product([np.arange(self.N), np.arange(self.T)], names=["n", "t"])
        features = ['feature%d'%i for i in range(self.K)]
        Y = Y.reshape(self.N*self.T, 1)
        X = X.reshape(self.N*self.T, self.K)
        self.data = pd.DataFrame(np.hstack((Y, X)), columns=['y'] + features, index=index)


np.random.seed(0)
dataset = Dataset(23, 50, 1, 1)
dataset.simulate(Effects.ind_fix, Slopes.homog, Variance.homosk)

print("TRUE COEFFICIENTS:\n", dataset.slopes_df, '\n')

# dataset = pd.read_csv('guns.csv', usecols = ['n', 't', 'feature0', 'y'], index_col = ['n', 't'])



# Perform PooledOLS
from linearmodels import PooledOLS
import statsmodels.api as sm

exog = sm.tools.tools.add_constant(dataset.data).drop(['y'], axis=1)
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
print(durbin_watson_test_results, '\n')


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


print("\n\nTRUE COEFFICIENTS:\n", dataset.slopes_df)