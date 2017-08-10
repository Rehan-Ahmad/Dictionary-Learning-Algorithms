# -*- coding: utf-8 -*-
"""
Created on Sun Aug 06 11:34:26 2017

@author: Rehan
"""
from copy import deepcopy
from DictUpdate03 import DictUpdate03
from omperr import omperr

def RSimCo(Y,param):
    class IPar():
        pass
    IPara = IPar()
    itN = param.itN
    D = deepcopy(param.initialDictionary)
    errglobal = param.errorGoal
    IPara.mu = 0.05
    IPara.I = param.I
    IPara.dispN = 20
    IPara.DebugFlag = 0
    IPara.itN = 1
    IPara.gmin = 1e-5 # the minimum value of gradient
    IPara.Lmin = 1e-6 # t4-t1 should be larger than Lmin
    IPara.t4 = 1e-2   # the initial value of t4
    IPara.rNmax = 3   # the number of iterative refinement in Part B in DictLineSearch03.m
    
    for itn in range(itN):
#        X = omp(D,Y,param.sparsity)
        X = omperr(D,Y,errglobal)
        D,X,_ = DictUpdate03(Y,D,X,IPara)
    
    return D