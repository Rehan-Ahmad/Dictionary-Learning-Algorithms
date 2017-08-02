# -*- coding: utf-8 -*-
"""
Created on Fri Jul 07 12:23:12 2017

@author: Rehan Ahmad

 DictUpdate03 is the dictionary update function in SimCO.
 Given the initial dictionary D, initial sparse coefficient matrix X and
 the traning data matrix Y, this function produces the updated D and X
 through itN iterations of line search algorithm in DictLineSearch03

 References:
 W. Dai, T. Xu, and W. Wang, 
 "Simultaneous Codeword Optimization (SimCO) for Dictionary Update and Learning,"
 submitted to IEEE Transactions on Signal Processing, October 2011.
 Full text is available at http://arxiv.org/abs/1109.5302

"""
import numpy as np
from copy import deepcopy
from DictLineSearch03 import DictLineSearch03
import pdb

def DictUpdate03(Y, Dhat, Xhat, IPara):
    D = deepcopy(Dhat)
    X = deepcopy(Xhat)
    
    class OPara():
        pass
    OPara = OPara()
    I = deepcopy(IPara.I)
    itN = deepcopy(IPara.itN)
    OPara.Flag = np.zeros((itN,))
    OPara.f0 = np.zeros((itN,))
    OPara.f1 = np.zeros((itN,))
    OPara.f0real = np.zeros((itN,))
    OPara.f1real = np.zeros((itN,))
    OPara.gn2 = np.zeros((itN,))
    OPara.topt = np.zeros((itN,))
    d = X.shape[0]
    m = Y.shape[0]

    #Ic is the complementary set of I
    I = np.intersect1d(range(d),I)
    Ic = np.setdiff1d(range(d),I)
    Yp = Y - np.dot(D[:,Ic],X[Ic,:])
    Omega = deepcopy(X!=0)
    ColUpdate = np.sum(Omega[I,:],axis=0)!=0
    YI = deepcopy(Yp[:,ColUpdate])
    DI = deepcopy(D[:,I])
    XI = deepcopy(X[:,ColUpdate][I,:])   # XI = X[I,ColUpdate]
    OmegaI = deepcopy(Omega[:,ColUpdate][I,:])
    f_YIComp = np.linalg.norm(Yp[:,~ColUpdate],'fro')**2
    
    # gradient descent line search

    for itn in range(itN):
        if itn == 0:
            OPara.f0real[itn] = np.linalg.norm(Y-np.dot(D,X),'fro')**2
        else:
            OPara.f0real[itn] = OPara.f1real[itn-1]
        
        # use the line search mechanism for dictionary update
        DI,XI,OParaLS = DictLineSearch03(YI,DI,OmegaI,IPara)
        D[:,I] = deepcopy(DI)
        X[np.ix_(I,ColUpdate)] = deepcopy(XI)
        OPara.Flag[itn] = OParaLS.Flag
        OPara.f0[itn] = OParaLS.f0 + f_YIComp
        OPara.f1[itn] = OParaLS.f1 + f_YIComp
        OPara.f1real[itn] = np.linalg.norm(Y-np.dot(D,X),'fro')**2
        OPara.gn2[itn] = deepcopy(OParaLS.gn2)
        OPara.topt[itn] = deepcopy(OParaLS.topt)
        
        if OParaLS.Flag != 0:
            OPara.Flag = OPara.Flag[0:itn]
            OPara.f0 = OPara.f0[0:itn]
            OPara.f1 = OPara.f1[0:itn]            
            OPara.f0real = OPara.f0real[0:itn]
            OPara.f1real = OPara.f1real[0:itn]
            OPara.gn2 = OPara.gn2[0:itn]
            OPara.topt = OPara.topt[0:itn]
            break;

        IPara.t4 = deepcopy(OParaLS.topt)
            
    # finalize
    D[:,I] = deepcopy(DI)
    X[np.ix_(I,ColUpdate)] = deepcopy(XI)

    return D,X,OPara
    
