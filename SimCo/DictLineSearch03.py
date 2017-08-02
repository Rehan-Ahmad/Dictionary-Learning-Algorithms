# -*- coding: utf-8 -*-
"""
Created on Fri Jul 07 15:10:42 2017

@author: Rehan Ahmad

 DictUpdate03 is the dictionary update function in SimCO.
 Use line search mechanism to update the dictionary D. This is one
 iteration of the line search algorithm for dictionary update
 
 References:
 W. Dai, T. Xu, and W. Wang, 
 "Simultaneous Codeword Optimization (SimCO) for Dictionary Update and Learning,"
 submitted to IEEE Transactions on Signal Processing, October 2011.
 Full text is available at http://arxiv.org/abs/1109.5302

"""
import numpy as np
from copy import deepcopy
from fg_tilde_eval01 import fg_tilde_eval01

def DictLineSearch03(Y,Dhat,Omega,IPara):
    D = deepcopy(Dhat)
    class OPara():
        pass
    OPara = OPara()
    c = (np.sqrt(5)-1)/2.0
    fv = np.zeros((100,))
    tv = np.zeros((100,))
    f4v = np.zeros((4,))
    t4v = np.zeros((4,))
    m,n = Y.shape
    d = D.shape[1]

    gmin = IPara.gmin
    Lmin = IPara.Lmin
    rNmax = IPara.rNmax
    t4v[3] = IPara.t4
    
    # compute the direction and corresponding gradient
    f,X,g,_ = fg_tilde_eval01(Y,D,Omega,IPara)
    evaln = 0
    fv[evaln] = deepcopy(f)
    tv[evaln] = 0
    
    # look at the magnitude of the gradient
    OPara.gn2 = np.linalg.norm(g,'fro')/np.linalg.norm(Y,'fro')**2
    gColn2 = np.sqrt(np.sum(g*g,axis = 0))
    gZero = gColn2 < gmin*np.linalg.norm(Y,'fro')**2/n
    # if the the magnitude of the gradient is less than the minimum threshold 
    # value, then quit and return D and X
    if np.sum(gZero) == D.shape[1]: 
        OPara.Flag = 1
        OPara.fv = fv[0:evaln]
        OPara.tv = tv[0:evaln]
        OPara.topt = 0
        OPara.f0 = deepcopy(f)
        OPara.f1 = deepcopy(f)
        return D,X,OPara

    gColn2[gZero] = 0
    H = np.zeros((m,d))
    H[0,gZero] = 1
    H[1:m,gZero] = 0
    H[:,~gZero] = g[:,~gZero]*np.tile(-1/gColn2[~gZero],(m,1))
    Step = gColn2/np.mean(gColn2)
    
    # Part A : find a good t4
    # set t4v and f4v; 
    t4v[2] = t4v[3]*c
    t4v[1] = t4v[3]*(1-c)
    f4v[0] = fv[0]
    for evaln in range(1,4):
        t = t4v[evaln]
        Dt = D*np.tile(np.cos(Step*t),(m,1)) + H*np.tile(np.sin(Step*t),(m,1))
        f4v[evaln],_,_,_ = fg_tilde_eval01(Y,Dt,Omega,IPara)

    fv[1:4] = f4v[1:4]
    tv[1:4] = t4v[1:4]
    # loop to find a good t4
    while t4v[3]-t4v[0] >= Lmin:
        # if f(D(t1)) is not greater than f(D(t2)), then t4=t2, t3=c*t4,
        #   t2=(1-c)*t4
        if f4v[0] <= f4v[1]:
            t4v[3] = t4v[1]
            t4v[2] = t4v[3]*c
            t4v[1] = t4v[3]*(1-c)
            f4v[3] = f4v[1]
            evaln = evaln + 1
            t = t4v[1]
            tv[evaln] = t
            Dt = D*np.tile(np.cos(Step*t),(m,1)) + H*np.tile(np.sin(Step*t),(m,1))
            ft,_,_,_ = fg_tilde_eval01(Y,Dt,Omega,IPara) 
            f4v[1] = ft
            fv[evaln] = ft
            evaln = evaln + 1
            t = t4v[2]
            tv[evaln] = t
            Dt = D*np.tile(np.cos(Step*t),(m,1)) + H*np.tile(np.sin(Step*t),(m,1))
            ft,_,_,_ = fg_tilde_eval01(Y,Dt,Omega,IPara)
            f4v[2] = ft
            fv[evaln] = ft
              
            # if f(D(t2)) is not greater than f(D(t3)), then t4=t3, t3=t2,
            #   t2=(1-c)*t4 
        elif f4v[1] <= f4v[2]:
            t4v[3] = t4v[2]; t4v[2] = t4v[1]; t4v[1] = t4v[3]*(1-c);
            f4v[3] = f4v[2]; f4v[2] = f4v[1]
            evaln = evaln + 1
            t = t4v[1]; tv[evaln] = t
            Dt = D*np.tile(np.cos(Step*t),(m,1)) + H*np.tile(np.sin(Step*t),(m,1))
            ft,_,_,_ = fg_tilde_eval01(Y,Dt,Omega,IPara)
            f4v[1] = ft; fv[evaln] = ft;
            # if f(D(t3)) is greater than f(D(t4)), then t2=t3, t3=t4
            #   t4=t3/c 
        elif f4v[2]>f4v[3]:
            t4v[1] = t4v[2]; t4v[2] = t4v[3]; t4v[3] = t4v[2]/c;
            f4v[1] = f4v[2]; f4v[2] = f4v[3]; 
            evaln = evaln + 1
            t = t4v[3]; tv[evaln] = t;
            Dt = D*np.tile(np.cos(Step*t),(m,1)) + H*np.tile(np.sin(Step*t),(m,1))
            ft,_,_,_ = fg_tilde_eval01(Y,Dt,Omega,IPara);
            f4v[3] = ft; fv[evaln] = ft;
        else:
            #quit
            break

    # if t is too small, fet Flag to minus 1
#    if t4v[3]-t4v[0] < Lmin:
#        Flag = -1
    
    # Part B: refine the segment
    evalN = evaln
    # iterate until t4-t1 is small enough
    while (t4v[3]-t4v[0]) >= Lmin and evaln-evalN <= rNmax:
        # if f(D(t1))>f(D(t2))>f(D(t3)), then t1=t2, t2=t3, t3=t1+c*(t4-t1)
        if f4v[0]>f4v[1] and f4v[1]>f4v[2]:
            t4v[0] = t4v[1]; t4v[1] = t4v[2]; t4v[2] = t4v[0]+c*(t4v[3]-t4v[0]);
            f4v[0] = f4v[1]; f4v[1] = f4v[2]
            evaln = evaln + 1
            t = t4v[2]; tv[evaln] = t;
            Dt = D*np.tile(np.cos(Step*t),(m,1)) + H*np.tile(np.sin(Step*t),(m,1))
            ft,_,_,_ = fg_tilde_eval01(Y,Dt,Omega,IPara);
            f4v[2] = ft; fv[evaln] = ft;
        # otherwise, t4=43, t3=t2, t2=t1+(1-c)(t4-t1)    
        else:
            t4v[3] = t4v[2]; t4v[2] = t4v[1]; t4v[1] = t4v[0]+(1-c)*(t4v[3]-t4v[0]);
            f4v[3] = f4v[2]; f4v[2] = f4v[1];
            evaln = evaln + 1;
            t = t4v[1]; tv[evaln] = t;
            Dt = D*np.tile(np.cos(Step*t),(m,1)) + H*np.tile(np.sin(Step*t),(m,1))
            ft,_,_,_ = fg_tilde_eval01(Y,Dt,Omega,IPara)
            f4v[1] = ft; fv[evaln] = ft;
    
    # finalize
    fv = fv[0:evaln]
    tv = tv[0:evaln]
    findex = np.argmin(fv)
    t = tv[findex]
    D = D*np.tile(np.cos(Step*t),(m,1)) + H*np.tile(np.sin(Step*t),(m,1))
    # compute X
    f,X,_,_ = fg_tilde_eval01(Y,D,Omega,IPara)
    
    OPara.f0 = fv[0]
    OPara.f1 = deepcopy(f)
    OPara.fv = deepcopy(fv)
    OPara.tv = deepcopy(tv)
    OPara.topt = deepcopy(t)
    OPara.Flag = 0
    
    return D,X,OPara
