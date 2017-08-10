# -*- coding: utf-8 -*-
"""
Created on Wed Aug 09 14:35:43 2017

Regularized Gradient Descent based Back Tracking Line Search (RGDBTLS) 
Algorithm. Regularization on Sparse matrix X.

@author: Rehan
"""
import numpy as np
from sklearn import preprocessing
from omperr import omperr
from copy import deepcopy

def RGDBTLS(Y,param):
    mu = param.mu
    D = deepcopy(param.initialDictionary)
    iterations = param.itN
    errglobal = param.errorGoal
    np.random.seed(3)
    beta = np.random.rand()
    np.random.seed(3)
    eta = np.random.rand()*0.5
    Grad = np.zeros(D.shape)

    for j in range(iterations):                        
        alpha = 1
        X = omperr(D,Y,errglobal)
        Dhat_RGDtemp = deepcopy(D)
        
        #################################################################
        # Back Tracking line search Algorithm (BTLS) to find optimal    #
        # value of alpha                                                #
        #################################################################
        Grad = -np.dot(Y-np.dot(D,X),X.T)
        oldfunc = np.linalg.norm(Y-np.dot(D,X),'fro')**2 + mu*np.linalg.norm(X,'fro')**2
        newfunc = np.linalg.norm(Y-np.dot(Dhat_RGDtemp,X),'fro')**2 + mu*np.linalg.norm(X,'fro')**2
        while(~(newfunc <= oldfunc-eta*alpha*np.sum(Grad**2))):
            alpha = beta*alpha
            Dhat_RGDtemp = deepcopy(D)            
            Dhat_RGDtemp = Dhat_RGDtemp + alpha*np.dot(Y-np.dot(Dhat_RGDtemp,X),X.T)
            Dhat_RGDtemp = preprocessing.normalize(Dhat_RGDtemp,norm='l2',axis=0)
            newfunc = np.linalg.norm(Y-np.dot(Dhat_RGDtemp,X),'fro')**2 + mu*np.linalg.norm(X,'fro')**2
            if(alpha < 1e-9):
                break
        #################################################################
        #################################################################
        D = D + alpha*np.dot(Y-np.dot(D,X),X.T)
        D = preprocessing.normalize(D,norm='l2',axis=0)
        ########## Update X Considering same sparsity pattern ###########
        Omega = X!=0
        ColUpdate = np.sum(Omega,axis=0)!=0
        YI = deepcopy(Y[:,ColUpdate])
        DI = deepcopy(D)
        OmegaI = deepcopy(Omega[:,ColUpdate])
        OmegaL = np.sum(OmegaI,axis=0)
        mu_sqrt = np.sqrt(mu)
        
        for cn in range(YI.shape[1]):
            L = deepcopy(OmegaL[cn])
            X[OmegaI[:,cn],cn] = np.linalg.lstsq(np.append(DI[:,OmegaI[:,cn]],\
              np.diag(mu_sqrt*np.ones((L,))),axis=0),\
                np.append(YI[:,cn],np.zeros((L,)),axis=0))[0]
    
    return D