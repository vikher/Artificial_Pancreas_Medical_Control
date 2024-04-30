#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 23:14:44 2021

@author: prantogg
"""
import pandas as pd
import numpy as np
from numpy import diff
import pickle
import statistics
#from scipy.fft import fft


test = pd.read_csv('test.csv', header = None)


def findLocalMaximaMinima(arr):  
 
    mx = []  
    mn = []
    n = len(arr)
    # Checking whether the first point is  
    # local maxima or minima or neither  
    if(arr[0] > arr[1]):  
        mx.append(0)  
    elif(arr[0] < arr[1]):  
        mn.append(0)  
    # Iterating over all points to check  
    # local maxima and local minima  
    for i in range(1, n-1):  
        # Condition for local maxima 
        if(arr[i-1] > arr[i] < arr[i + 1]):  
            mn.append(i)  
        elif(arr[i-1] < arr[i] > arr[i + 1]):  
            mx.append(i)  
  
    if(arr[-1] > arr[-2]):  
        mx.append(n-1)  
    elif(arr[-1] < arr[-2]):  
        mn.append(n-1) 
     
    return (mx,mn)


testFeat = []
for i in range(len(test)):
    
    # To calculate T
    high = max(test.iloc[i])
    peak = test.iloc[i].loc[test.iloc[i] == high].index[0]
    T = (peak)*5
    
    # range
    rng = high - min(test.iloc[i])
    
    # To calculate dG normalized
    dGn = (high - test.iloc[i][0]) / test.iloc[i][0]
    
    # Fast Fourier Transform features
    farr = 20*np.log10(np.abs(np.fft.fft(test.iloc[i])))
    (mx,mn) = findLocalMaximaMinima(farr)
    
    f1 = mx[1]
    pf1 = farr[f1]
    
    # if len(mx) <= 2:
    #     f2 = mx[1] - 1
    # else:
    #     f2 = mx[2]
    # pf2 = farr[f2]
    
    
    # differential
    
    d = diff(test.iloc[0])/1
    d1 = max(d) - min(d)
    dd = diff(d)/1
    d2 = max(dd) - min(dd)
    
    # Standard Deviation
    stD = statistics.stdev(test.iloc[i])
    
    featset = [high,rng,T,dGn,d1,d2,stD,f1,pf1]
    testFeat.append(featset)
    
testF = pd.DataFrame(testFeat)




classifier = pickle.load(open('final_model.sav', 'rb'))
y_pred = classifier.predict(testF)
result = pd.DataFrame(y_pred)
result.to_csv('Result.csv', index = False, header = False)
















