# -*- coding: utf-8 -*-
"""
Created on Fri Jul 07 17:13:53 2017

@author: Rehan Ahmad

 fg_tilde_eval01 computes the gradient descent direction for LineSearch 
 in regularized SimCO version

 References:
 W. Dai, T. Xu, and W. Wang, 
 "Simultaneous Codeword Optimization (SimCO) for Dictionary Update and Learning,"
 submitted to IEEE Transactions on Signal Processing, October 2011.
 Full text is available at http://arxiv.org/abs/1109.5302

"""
import numpy as np
from copy import deepcopy
import pdb

def fg_tilde_eval01(Y,D,Omega,IPara):
    m,n = Y.shape
    d = D.shape[1]
    X = np.zeros((d,n))
    OmegaL = np.sum(Omega,axis = 0)
    mu = IPara.mu #the parameter of regularized item
    mu_sqrt = np.sqrt(mu)
    for cn in range(n):
        L = deepcopy(OmegaL[cn])
        X[Omega[:,cn],cn] = np.linalg.lstsq(np.append(D[:,Omega[:,cn]],np.diag(mu_sqrt*np.ones((L,))),axis=0),\
         np.append(Y[:,cn],np.zeros((L,)),axis=0))[0]

    Yr = Y - np.dot(D,X)
    # the cost function with regularized term
    f = np.sum(Yr*Yr) + mu*np.sum(X*X)
    freal = np.sum(Yr*Yr)
    
    g = -2*np.dot(Yr,X.T)
    # additional steps to make sure the orthoganilty
    DGcorr = np.sum(D*g, axis = 0)
    g = g - D*np.tile(DGcorr,(m,1))
    
    return f,X,g,freal
