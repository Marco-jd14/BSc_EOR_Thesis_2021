# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 16:03:51 2021

@author: Marco
"""

import pandas as pd
import numpy as np
import enum

from simulate import Effects, Slopes, Variance, Dataset

np.random.seed(0)
dataset = Dataset(N=50, T=5, K=1, G=5)
dataset.simulate(Effects.gr_tvar_fix, Slopes.homoh, Variance.homosk)

#TODO: Bonhomme and Manresa estimation