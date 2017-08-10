# -*- coding: utf-8 -*-
"""
Created on Wed Aug 09 16:13:55 2017

KSVD based Dictionary Learning algorithm.

@author: Rehan
"""
import numpy as np
from copy import deepcopy
from omperr import omperr
from KSVD import KSVD

def KSVDDL(Y,param):
    D = deepcopy(param.initialDictionary)
    iterations = param.itN
    errglobal = param.errorGoal

    for j in range(iterations):
        X = omperr(D,Y,errglobal)
        D,X = KSVD(Y,D,X)

    return D