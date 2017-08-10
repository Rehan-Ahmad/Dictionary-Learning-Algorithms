# -*- coding: utf-8 -*-
"""
Created on Wed Aug 09 14:00:03 2017

@author: Rehan

Gradient Descent Dictionary Learning (GDDL) with Momentum term  

"""
import numpy as np
from copy import deepcopy
from omperr import omperr
from sklearn import preprocessing

def GDDL(Y,param):
    iterations = param.itN
    D = deepcopy(param.initialDictionary)
    errglobal = param.errorGoal
    gamma = param.MomentumGamma
    alpha = param.alpha
    
    v = np.zeros(D.shape)
    for j in range(iterations):
        X = omperr(D,Y,errglobal)
        v = gamma*v - alpha*np.dot(Y-np.dot(D,X),X.T)
        D = D - v
        D = preprocessing.normalize(D,norm='l2',axis=0)
    return D