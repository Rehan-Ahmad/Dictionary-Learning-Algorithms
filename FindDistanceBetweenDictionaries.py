# -*- coding: utf-8 -*-
"""
Created on Fri Jul 07 11:42:02 2017

@author: Rehan Ahmad

 D   : Original Dictionary
 Dhat: Predicted/Estimated Dictionary 
 
 Finding the matched atoms between original and predicted Dictionary.
 Sweeping through the columns of original Dictionary and finding closest
 column in predicted Dictionary.

 Reference: 
  "The K-SVD: An Algorithm for Designing of Overcomplete Dictionaries for Sparse
  Representation", written by M. Aharon, M. Elad, and A.M. Bruckstein and 
  appeared in the IEEE Trans. On Signal Processing, Vol. 54, no. 11, 
  pp. 4311-4322, November 2006. 

"""
import numpy as np
from copy import deepcopy
def FindDistanceBetweenDictionaries(D,Dhat):
    catchCounter = 0 
    totalDistances = 0
    Dnew = np.ndarray((D.shape[0],D.shape[1]))
    for i in range(Dhat.shape[1]):
        Dnew[:,i] = deepcopy(np.sign(Dhat[0,i])*Dhat[:,i])
    for i in range(Dhat.shape[1]):
        d = deepcopy(np.sign(D[0,i])*D[:,i])
        distances = np.sum((Dnew-np.tile(np.reshape(d,(-1,1)),(1,Dhat.shape[1])))**2,axis=0)
        index = np.argmin(distances)
        errorOfElement = 1 - np.abs(np.dot(Dnew[:,index],d))
        totalDistances = totalDistances + errorOfElement
        catchCounter = catchCounter + (errorOfElement < 0.01)
    return catchCounter
