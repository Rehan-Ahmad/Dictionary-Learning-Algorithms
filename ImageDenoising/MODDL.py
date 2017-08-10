# -*- coding: utf-8 -*-
"""
Created on Wed Aug 09 16:10:33 2017

MOD algorithm for Dictionary Learning

@author: Rehan
"""
import numpy as np
from copy import deepcopy
from omperr import omperr
from sklearn import preprocessing

def MODDL(Y,param):
    D = deepcopy(param.initialDictionary)
    iterations = param.itN
    errglobal = param.errorGoal

    for j in range(iterations):
        X = omperr(D,Y,errglobal)
        D = np.dot(Y,np.linalg.pinv(X))
        D = preprocessing.normalize(D,norm='l2',axis=0)
    return D
