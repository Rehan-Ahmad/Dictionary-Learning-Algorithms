# -*- coding: utf-8 -*-
"""
Created on Wed Aug 09 14:24:07 2017
Gradient Descent Back Tracking Line Search (GDBTLS) Algorithm
@author: Rehan
"""
import numpy as np
from omperr import omperr
from copy import deepcopy
from sklearn import preprocessing

def GDBTLS(Y,param):
    D = deepcopy(param.initialDictionary)
    errglobal = param.errorGoal
    iterations = param.itN
    np.random.seed(3)
    beta = np.random.rand()
    np.random.seed(3)
    eta = np.random.rand()*0.5    
    Grad = np.zeros(D.shape)
    
    for j in range(iterations):
        alpha = 1
        X = omperr(D,Y,errglobal)
        Dhat_GDtemp = deepcopy(D)
        
        #################################################################
        # Back Tracking line search Algorithm (BTLS) to find optimal    #
        # value of alpha                                                #
        #################################################################
        Grad = -np.dot(Y-np.dot(D,X),X.T)
        oldfunc = np.linalg.norm(Y-np.dot(D,X),'fro')**2 
        newfunc = np.linalg.norm(Y-np.dot(Dhat_GDtemp,X),'fro')**2 
        while(~(newfunc <= oldfunc-eta*alpha*np.sum(Grad**2))):
            alpha = beta*alpha
            Dhat_GDtemp = deepcopy(D)            
            Dhat_GDtemp = Dhat_GDtemp + alpha*np.dot(Y-np.dot(Dhat_GDtemp,X),X.T)
            Dhat_GDtemp = preprocessing.normalize(Dhat_GDtemp,norm='l2',axis=0)
            newfunc = np.linalg.norm(Y-np.dot(Dhat_GDtemp,X),'fro')**2
            if(alpha < 1e-9):
                break
        #################################################################
        #################################################################
        D = D + alpha*np.dot(Y-np.dot(D,X),X.T)
        D = preprocessing.normalize(D,norm='l2',axis=0)
        
    return D