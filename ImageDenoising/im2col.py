# -*- coding: utf-8 -*-
"""
Created on Mon Aug 07 23:56:42 2017
 
 Implementation taken from Matlab im2col.py
 This version contains sliding order only.

@author: Rehan
"""
import numpy as np
from copy import deepcopy

def im2col(a,block):
    ma,na = a.shape
    m = block[0] 
    n = block[1]
    
    if (ma<m or na<n): # if neighborhood is larger than image
        b = np.zeros((m*n,))
        return b
    
    # Create Hankel-like indexing sub matrix.
    mc = block[0]
    nc = ma-m+1
    nn = na-n+1
    cidx = np.reshape(range(mc),(-1,1))
    ridx = np.reshape(range(1,nc+1),(1,-1))
    t = cidx[:,np.zeros((nc,),dtype=int)] + ridx[np.zeros((mc,),dtype=int),:] # Hankel Subscripts
    tt = np.zeros((mc*n,nc))
    rows = np.array(range(mc))
    for i in range(n):
        tt[i*mc+rows,:] = t+ma*i

    ttt = np.zeros((mc*n,nc*nn),dtype=int)
    cols = np.array(range(nc))
    for j in range(nn):
        ttt[:,j*nc+cols] = tt+ma*j
    
    # If a is a row vector, change it to a column vector. This change is
    # necessary when A is a row vector and [M N] = size(A).

    b = np.zeros((ttt.shape[0],ttt.shape[1]))
    for i in range(ttt.shape[0]):
        b[i,:] = deepcopy(a[np.unravel_index(ttt[i,:]-1,(int(a.shape[0]),int(a.shape[1])),order='F')])
    #b = deepcopy(a[ttt-1])
    return b

