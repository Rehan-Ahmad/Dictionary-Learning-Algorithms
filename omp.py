# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 12:08:22 2017

@author: Rehan Ahmad

Orthogonal Matching Persuit (OMP) algorithm for sparse representation

 D: Dictionary, columns must be normalized (by l2 norm)
 X: input signal to represent
 L: max. no. of coefficients for each signal
 A: Sparse coefficient Matrix
 
"""
import numpy as np
from copy import deepcopy

def omp(D,X,L):
    n,P = X.shape
    n,K = D.shape
    A = np.ndarray((D.shape[1],X.shape[1]))
    for k in range(P):
        a = 0
        x = deepcopy(X[:,k])
        residual = deepcopy(x)
        indx = np.zeros((L,),dtype = int)
        for j in range(L):
            proj = np.dot(D.T,residual)
            pos = np.argmax(np.abs(proj))
            indx[j] = int(pos)
            a = np.dot(np.linalg.pinv(D[:,indx[0:j+1]]),x)
            residual = x-np.dot(D[:,indx[0:j+1]],a)
            if np.sum(residual**2) < 1e-6:
                break
        temp = np.zeros((K,))
        temp[indx[0:j+1]] = deepcopy(a)
        A[:,k] = temp
    return A
