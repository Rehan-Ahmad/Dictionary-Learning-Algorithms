# -*- coding: utf-8 -*-
"""
Created on Mon Jun 05 23:31:51 2017

@author: Rehan Ahmad

Back Tracking Line Search taken from:
http://users.ece.utexas.edu/~cmcaram/EE381V_2012F/Lecture_4_Scribe_Notes.final.pdf

"""
import numpy as np
from sklearn import preprocessing
import matplotlib.pylab as plt
from copy import deepcopy
import time
from omp import omp
from KSVD import KSVD
from FindDistanceBetweenDictionaries import FindDistanceBetweenDictionaries
from DictUpdate03 import DictUpdate03
import pdb

def awgn(x,snr_db):
    L = len(x)
    Es = np.sum(np.abs(x)**2)/L
    snr_lin = 10**(snr_db/10.0)
    noise = np.sqrt(Es/snr_lin)*np.random.randn(L)
    y = x + noise
    return y

if __name__ == "__main__":
    tic = time.time()
    
    FlagPGD = True; FlagPGDMom = True; FlagMOD = False; FlagKSVD = True;
    FlagRSimCo = True; FlagPSimCo = True; FlagGDBTLS = True; FlagRGDBTLS = True
    
    drows = 16 #20 #16
    dcols = 32 #50 #32
    ycols = 78 #1500 #78
    alpha = 0.005

    iterations = 1000
    SNR = 20
    epochs = 1
    sparsity = 4
    
    count_success = np.ndarray((iterations,epochs))
    count_success_momen = np.ndarray((iterations,epochs))
    count_success_MOD = np.ndarray((iterations,epochs))
    count_success_KSVD = np.ndarray((iterations,epochs))
    count_success_RSimCo = np.ndarray((iterations,epochs))
    count_success_PSimCo = np.ndarray((iterations,epochs))
    count_success_GDBTLS = np.ndarray((iterations,epochs))
    count_success_RGDBTLS = np.ndarray((iterations,epochs))

    e = np.ndarray((iterations,epochs))
    e_momen = np.ndarray((iterations,epochs))
    e_GDBTLS = np.ndarray((iterations,epochs))
    e_MOD = np.ndarray((iterations,epochs))
    e_KSVD = np.ndarray((iterations,epochs))
    e_RSimCo = np.ndarray((iterations,epochs))
    e_PSimCo = np.ndarray((iterations,epochs))
    e_RGDBTLS = np.ndarray((iterations,epochs))
    
    for epoch in range(epochs):
        alpha = 0.005
#        np.random.seed(epoch)

         ################# make initial dictionary #############################
#        Pn=ceil(sqrt(K));
#        DCT=zeros(bb,Pn);
#        for k=0:1:Pn-1,
#            V=cos([0:1:bb-1]'*k*pi/Pn);
#            if k>0, V=V-mean(V); end;
#            DCT(:,k+1)=V/norm(V);
#        end;
#        DCT=kron(DCT,DCT);
        ######################################################################

        # Creating dictionary from uniform iid random distribution
        # and normalizing atoms by l2-norm                
        D = np.random.rand(drows,dcols)
        D = preprocessing.normalize(D,norm='l2',axis=0)            
        # Creating data Y by linear combinations of randomly selected
        # atoms and iid uniform coefficients         
        Y = np.ndarray((drows,ycols))
        for i in range(ycols):
            PermIndx = np.random.permutation(dcols)
            Y[:,i] = np.random.rand()*D[:,PermIndx[0]] + \
             np.random.rand()*D[:,PermIndx[1]] + \
             np.random.rand()*D[:,PermIndx[2]] + \
             np.random.rand()*D[:,PermIndx[3]]
             
        # Add awgn noise in data Y 
#        for i in range(ycols):
#            Y[:,i] = awgn(Y[:,i],SNR)
        
        Dhat = np.ndarray((drows,dcols))
        Dhat = deepcopy(Y[:,np.random.permutation(ycols)[0:dcols]])
        Dhat = preprocessing.normalize(Dhat,norm='l2',axis=0)
        Dhat_momen = deepcopy(Dhat)
        Dhat_MOD = deepcopy(Dhat)
        Dhat_KSVD = deepcopy(Dhat)
        Dhat_RSimCo = deepcopy(Dhat)
        Dhat_PSimCo = deepcopy(Dhat)
        Dhat_GDBTLS = deepcopy(Dhat)
        Dhat_RGDBTLS = deepcopy(Dhat)
    
        ########################################################
        # Applying Projected Gradient Descent without momentum #
        ########################################################
        if(FlagPGD==True):
            X = omp(D,Y,sparsity)
            for j in range(iterations):
#                X = omp(Dhat,Y,sparsity)
    #            for i in range(dcols):
    #                R = Y-np.dot(Dhat,X)
    #                Dhat[:,i] = Dhat[:,i] + alpha*np.dot(R,X[i,:])
                Dhat = Dhat + alpha*np.dot(Y-np.dot(Dhat,X),X.T) #Parallel dictionary update...
                Dhat = preprocessing.normalize(Dhat,norm='l2',axis=0)
                
                e[j,epoch] = np.linalg.norm(Y-np.dot(Dhat,X),'fro')**2
                count = FindDistanceBetweenDictionaries(D,Dhat)
                count_success[j,epoch] = count
        #####################################################
        # Applying Projected Gradient Descent with momentum #
        #####################################################
        if(FlagPGDMom==True):
            v = np.zeros((drows,dcols))
            gamma = 0.5
            X = omp(D,Y,sparsity)            
            for j in range(iterations):
#                X = omp(Dhat_momen,Y,sparsity)
    #            for i in range(dcols):
    #                R = Y-np.dot(Dhat_momen,X)
    #                v[:,i] = gamma*v[:,i] + alpha*np.dot(R,X[i,:])
    #                Dhat_momen[:,i] = Dhat_momen[:,i] + v[:,i]
                v = gamma*v - alpha*np.dot(Y-np.dot(Dhat_momen,X),X.T)
                Dhat_momen = Dhat_momen - v
                    
                Dhat_momen = preprocessing.normalize(Dhat_momen,norm='l2',axis=0)        
                e_momen[j,epoch] = np.linalg.norm(Y-np.dot(Dhat_momen,X),'fro')**2
                count_momen = FindDistanceBetweenDictionaries(D,Dhat_momen)
                count_success_momen[j,epoch] = count_momen
        #####################################################
        # Applying Gradient Descent with back tracking line #
        # search algorithm                                  #
        #####################################################
        if(FlagGDBTLS==True):
            alpha = 1
            beta = np.random.rand()
            eta = np.random.rand()*0.5
            Grad = np.zeros((drows,dcols))
            
            X = omp(D,Y,sparsity)
            for j in range(iterations):
                alpha = 1
#                X = omp(Dhat_GDBTLS,Y,sparsity)
                Dhat_GDtemp = deepcopy(Dhat_GDBTLS)
                
                #################################################################
                # Back Tracking line search Algorithm (BTLS) to find optimal    #
                # value of alpha                                                #
                #################################################################
                Grad = -np.dot(Y-np.dot(Dhat_GDBTLS,X),X.T)
                oldfunc = np.linalg.norm(Y-np.dot(Dhat_GDBTLS,X),'fro')**2 
                newfunc = np.linalg.norm(Y-np.dot(Dhat_GDtemp,X),'fro')**2 
                while(~(newfunc <= oldfunc-eta*alpha*np.sum(Grad**2))):
                    alpha = beta*alpha
                    Dhat_GDtemp = deepcopy(Dhat_GDBTLS)            
                    Dhat_GDtemp = Dhat_GDtemp + alpha*np.dot(Y-np.dot(Dhat_GDtemp,X),X.T)
                    Dhat_GDtemp = preprocessing.normalize(Dhat_GDtemp,norm='l2',axis=0)
                    newfunc = np.linalg.norm(Y-np.dot(Dhat_GDtemp,X),'fro')**2
                    if(alpha < 1e-9):
                        break
                #################################################################
                #################################################################
                Dhat_GDBTLS = Dhat_GDBTLS + alpha*np.dot(Y-np.dot(Dhat_GDBTLS,X),X.T)
                Dhat_GDBTLS = preprocessing.normalize(Dhat_GDBTLS,norm='l2',axis=0)
                
                e_GDBTLS[j,epoch] = np.linalg.norm(Y-np.dot(Dhat_GDBTLS,X),'fro')**2
                count_GDBTLS = FindDistanceBetweenDictionaries(D,Dhat_GDBTLS)
                count_success_GDBTLS[j,epoch] = count_GDBTLS
                
        #####################################################
        # Applying Gradient Descent with back tracking line #
        # search algorithm with regularization on X         #
        #####################################################
        if(FlagRGDBTLS==True):
            alpha = 1
            mu = 0.01 
#            beta = np.random.rand()
#            eta = np.random.rand()*0.5
#            Grad = np.zeros((drows,dcols))
#            mu = 0.01 
            
            X = omp(D,Y,sparsity)
            for j in range(iterations):
                alpha = 1
#                X = omp(Dhat_RGDBTLS,Y,sparsity)
                Dhat_RGDtemp = deepcopy(Dhat_RGDBTLS)
                
                #################################################################
                # Back Tracking line search Algorithm (BTLS) to find optimal    #
                # value of alpha                                                #
                #################################################################
                Grad = -np.dot(Y-np.dot(Dhat_RGDBTLS,X),X.T)
                oldfunc = np.linalg.norm(Y-np.dot(Dhat_RGDBTLS,X),'fro')**2 + mu*np.linalg.norm(X,'fro')**2
                newfunc = np.linalg.norm(Y-np.dot(Dhat_RGDtemp,X),'fro')**2 + mu*np.linalg.norm(X,'fro')**2
                while(~(newfunc <= oldfunc-eta*alpha*np.sum(Grad**2))):
                    alpha = beta*alpha
                    Dhat_RGDtemp = deepcopy(Dhat_RGDBTLS)            
                    Dhat_RGDtemp = Dhat_RGDtemp + alpha*np.dot(Y-np.dot(Dhat_RGDtemp,X),X.T)
                    Dhat_RGDtemp = preprocessing.normalize(Dhat_RGDtemp,norm='l2',axis=0)
                    newfunc = np.linalg.norm(Y-np.dot(Dhat_RGDtemp,X),'fro')**2 + mu*np.linalg.norm(X,'fro')**2
                    if(alpha < 1e-9):
                        break
                #################################################################
                #################################################################
                Dhat_RGDBTLS = Dhat_RGDBTLS + alpha*np.dot(Y-np.dot(Dhat_RGDBTLS,X),X.T)
                Dhat_RGDBTLS = preprocessing.normalize(Dhat_RGDBTLS,norm='l2',axis=0)
                ########## Update X Considering same sparsity pattern############
                Omega = X!=0
                ColUpdate = np.sum(Omega,axis=0)!=0
                YI = deepcopy(Y[:,ColUpdate])
                DI = deepcopy(Dhat_RGDBTLS)
                XI = deepcopy(X[:,ColUpdate])
                OmegaI = deepcopy(Omega[:,ColUpdate])
                OmegaL = np.sum(Omega,axis=0)
                mu_sqrt = np.sqrt(mu)
                
                for cn in range(ycols):
                    L = deepcopy(OmegaL[cn])
                    X[OmegaI[:,cn],cn] = np.linalg.lstsq(np.append(DI[:,OmegaI[:,cn]],\
                      np.diag(mu_sqrt*np.ones((L,))),axis=0),\
                        np.append(YI[:,cn],np.zeros((L,)),axis=0))[0]
                #################################################################
                e_RGDBTLS[j,epoch] = np.linalg.norm(Y-np.dot(Dhat_RGDBTLS,X),'fro')**2
                count_RGDBTLS = FindDistanceBetweenDictionaries(D,Dhat_RGDBTLS)
                count_success_RGDBTLS[j,epoch] = count_RGDBTLS
        ############################################
        # Applying MOD Algorithm                   #
        ############################################     
        if(FlagMOD==True):
            X = omp(D,Y,sparsity)
            for j in range(iterations):
    #            X = omp(Dhat_MOD,Y,sparsity)
                Dhat_MOD = np.dot(Y,np.linalg.pinv(X))
                Dhat_MOD = preprocessing.normalize(Dhat_MOD,norm='l2',axis=0)
        
                count_MOD = FindDistanceBetweenDictionaries(D,Dhat_MOD)
                count_success_MOD[j,epoch] = count_MOD
                e_MOD[j,epoch] = np.linalg.norm(Y-np.dot(Dhat_MOD,X),'fro')**2
        ############################################
        # Applying KSVD Algorithm                  #
        ############################################        
        if(FlagKSVD==True):
            X = omp(D,Y,sparsity)
            for j in range(iterations):
    #            X = omp(Dhat_KSVD,Y,sparsity)
                Dhat_KSVD,X = KSVD(Y,Dhat_KSVD,X)
        
                count_KSVD = FindDistanceBetweenDictionaries(D,Dhat_KSVD)
                count_success_KSVD[j,epoch] = count_KSVD
                e_KSVD[j,epoch] = np.linalg.norm(Y-np.dot(Dhat_KSVD,X),'fro')**2

        #############################################
        # Applying Regularized SimCo Algorithm      #
        #############################################
        if(FlagRSimCo==True):
            class IPara():
                pass
            IPara = IPara()
            IPara.I = range(D.shape[1])
            IPara.mu = 0.01
            IPara.dispN = 20 
            IPara.DebugFlag = 0
            IPara.itN = 1
            IPara.gmin = 1e-5; # the minimum value of gradient
            IPara.Lmin = 1e-6; # t4-t1 should be larger than Lmin
            IPara.t4 = 1e-2;   # the initial value of t4
            IPara.rNmax = 3;   # the number of iterative refinement in Part B in DictLineSearch03.m
        
            X = omp(D,Y,sparsity)
            for j in range(iterations):
    #            X = omp(Dhat_RSimCo,Y,sparsity)
                Dhat_RSimCo,X,_ = DictUpdate03(Y,Dhat_RSimCo,X,IPara)
        
                count_RSimCo = FindDistanceBetweenDictionaries(D,Dhat_RSimCo)
                count_success_RSimCo[j,epoch] = count_RSimCo
                e_RSimCo[j,epoch] = np.linalg.norm(Y-np.dot(Dhat_RSimCo,X),'fro')**2
        #############################################
        # Applying Primitive SimCo Algorithm        #
        #############################################    
        if(FlagPSimCo==True):
            IPara.mu = 0
            X = omp(D,Y,sparsity)
            for j in range(iterations):
    #            X = omp(Dhat_PSimCo,Y,sparsity)
                Dhat_PSimCo,X,_ = DictUpdate03(Y,Dhat_PSimCo,X,IPara)
        
                count_PSimCo = FindDistanceBetweenDictionaries(D,Dhat_PSimCo)
                count_success_PSimCo[j,epoch] = count_PSimCo
                e_PSimCo[j,epoch] = np.linalg.norm(Y-np.dot(Dhat_PSimCo,X),'fro')**2
        #############################################
        #############################################
        print 'epoch: ',epoch,'completed'

    plt.close('all')
    if FlagPGD==True:     plt.plot(np.sum(count_success,axis=1)/epochs,'b',label = 'PGD')
    if FlagPGDMom==True:  plt.plot(np.sum(count_success_momen,axis=1)/epochs,'r',label = 'PGD_Momentum')
    if FlagMOD==True:     plt.plot(np.sum(count_success_MOD,axis=1)/epochs,'g',label = 'MOD')
    if FlagKSVD==True:    plt.plot(np.sum(count_success_KSVD,axis=1)/epochs,'y',label = 'KSVD')
    if FlagRSimCo==True:  plt.plot(np.sum(count_success_RSimCo,axis=1)/epochs,'m',label = 'RSimCo')
    if FlagPSimCo==True:  plt.plot(np.sum(count_success_PSimCo,axis=1)/epochs,'c',label = 'PSimCo')
    if FlagGDBTLS==True:  plt.plot(np.sum(count_success_GDBTLS,axis=1)/epochs,':',label = 'GDBTLS')
    if FlagRGDBTLS==True: plt.plot(np.sum(count_success_RGDBTLS,axis=1)/epochs,'--',label = 'R_GDBTLS')

    plt.legend()
    plt.xlabel('iteration number')
    plt.ylabel('Success Counts in iteration')
    plt.title('Dictionary Learning Algorithms applied on Syhthetic data')
    
    plt.figure()
    if FlagPGD==True:     plt.plot(np.sum(e,axis=1)/epochs,'b',label = 'PGD')
    if FlagPGDMom==True:  plt.plot(np.sum(e_momen,axis=1)/epochs,'r',label = 'PGD_Momentum')
    if FlagMOD==True:     plt.plot(np.sum(e_MOD,axis=1)/epochs,'g',label = 'MOD')
    if FlagKSVD==True:    plt.plot(np.sum(e_KSVD,axis=1)/epochs,'y',label = 'KSVD')
    if FlagRSimCo==True:  plt.plot(np.sum(e_RSimCo,axis=1)/epochs,'m',label = 'RSimCo')
    if FlagPSimCo==True:  plt.plot(np.sum(e_PSimCo,axis=1)/epochs,'c',label = 'PSimCo')
    if FlagGDBTLS==True:  plt.plot(np.sum(e_GDBTLS,axis=1)/epochs,':',label = 'GDBTLS')
    if FlagRGDBTLS==True: plt.plot(np.sum(e_RGDBTLS,axis=1)/epochs,'--',label = 'R_GDBTLS')
    
    plt.legend()
    plt.xlabel('iteration number')
    plt.ylabel('Error: Sum of squares')
            
    toc = time.time()
    print 'Total Time Taken by code: ','%.2f' %((toc-tic)/60.0),'min'
