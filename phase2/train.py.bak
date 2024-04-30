#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:00:56 2021

@author: prantogg
"""

import pandas as pd
import numpy as np
from numpy import diff
from datetime import timedelta 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import statistics
#from scipy.fft import fft
#from matplotlib import pyplot as plt


df = pd.read_csv('InsulinData.csv')
insulinD = df.filter(['Index','Date','Time','BWZ Carb Input (grams)'])
del df

insulinD['Date'] = insulinD['Date'] + ' ' + insulinD['Time']
insulinD['Date'] = pd.to_datetime(insulinD['Date'])
insulinD.drop(['Time'], axis = 1, inplace = True)
insulinD['BWZ Carb Input (grams)'].replace(0, np.nan, inplace = True)

mealtimes = insulinD.loc[insulinD['BWZ Carb Input (grams)'].notnull(), 'Date']
pd.to_datetime(mealtimes)

meals1 = pd.Series()
_ = 0
for i in range((len(mealtimes)-1),0,-1):
    if (mealtimes.iloc[i-1] - mealtimes.iloc[i]) < timedelta(hours=2):
        pass
    else:
        meals1.loc[_] = mealtimes.iloc[i]
        _ += 1    
meals1.loc[_] = mealtimes.iloc[0]
pd.to_datetime(meals1)




df2 = pd.read_csv('Insulin_patient2.csv')
insulinD2 = df2.filter(['Index','Date','Time','BWZ Carb Input (grams)'])
del df2

insulinD2['Date'] = insulinD2['Date'] + ' ' + insulinD2['Time']
insulinD2['Date'] = pd.to_datetime(insulinD2['Date'])
insulinD2.drop(['Time'], axis = 1, inplace = True)
insulinD2['BWZ Carb Input (grams)'].replace(0, np.nan, inplace = True)

mealtimes2 = insulinD2.loc[insulinD2['BWZ Carb Input (grams)'].notnull(), 'Date']
pd.to_datetime(mealtimes2)

meals2 = pd.Series()
_ = 0
for i in range((len(mealtimes2)-1),0,-1):
    if (mealtimes2.iloc[i-1] - mealtimes2.iloc[i]) < timedelta(hours=2):
        pass
    else:
        meals2.loc[_] = mealtimes2.iloc[i]
        _ += 1    
meals2.loc[_] = mealtimes2.iloc[0]
pd.to_datetime(meals2)

meals = meals1.append(meals2, ignore_index = True)










df = pd.read_csv('CGMData.csv')
cgmD = df.filter(['Index', 'Date', 'Time', 'Sensor Glucose (mg/dL)'])
del df

cgmD['Date'] = cgmD['Date'] + ' ' + cgmD['Time']
cgmD['Date'] = pd.to_datetime(cgmD['Date'])
cgmD.drop(['Time'], axis = 1, inplace = True)

df2 = pd.read_csv('CGM_patient2.csv')
cgmD2 = df2.filter(['Index', 'Date', 'Time', 'Sensor Glucose (mg/dL)'])
del df2

cgmD2['Date'] = cgmD2['Date'] + ' ' + cgmD2['Time']
cgmD2['Date'] = pd.to_datetime(cgmD2['Date'])
cgmD2.drop(['Time'], axis = 1, inplace = True)
cgmD2.drop([16478,16479,16480,16481,16482,16483,16484,16485,16486,16487,16488,16489], inplace=True)
   


datalstm = []
for i in range(len(meals1)):
    filt = (cgmD['Date'] > (meals1.iloc[i]-timedelta(hours=0.5))) & (cgmD['Date'] <= (meals1.iloc[i]+timedelta(hours=2)))
    temp1 = cgmD.loc[filt, 'Sensor Glucose (mg/dL)'].tolist()
    temp1.reverse()
    datalstm.append(temp1)

for i in range(len(meals2)):
    filt = (cgmD2['Date'] >= (meals2.iloc[i]-timedelta(hours=0.5))) & (cgmD2['Date'] <= (meals2.iloc[i]+timedelta(hours=2)))
    temp2 = cgmD2.loc[filt, 'Sensor Glucose (mg/dL)'].tolist()
    temp2.reverse()
    datalstm.append(temp2)
mealD = pd.DataFrame(datalstm, dtype = 'float32')
dlenm = [len(i) for i in datalstm]
#datalstm.clear()







nomeals1 = pd.Series()
_ = 0
for i in range(len(mealtimes)-1,0,-1):
    if (mealtimes.iloc[i-1]-mealtimes.iloc[i]) > timedelta(hours=4):
        nomeals1.loc[_] = mealtimes.iloc[i]
        _ += 1
    else:
        pass
nomeals1.loc[_] = mealtimes.iloc[0]
pd.to_datetime(nomeals1)

nomeals2 = pd.Series()
_ = 0
for i in range(len(mealtimes2)-1,0,-1):
    if (mealtimes2.iloc[i-1]-mealtimes2.iloc[i]) > timedelta(hours=4):
        nomeals2.loc[_] = mealtimes2.iloc[i]
        _ += 1
    else:
        pass
nomeals2.loc[_] = mealtimes2.iloc[0]
pd.to_datetime(nomeals2)


datalstnm = []
nomealT1 = pd.Series()
_ = 0
sum = 0
for i in range(len(nomeals1)-1):
    itr = (nomeals1.iloc[i+1]-nomeals1.iloc[i]-timedelta(hours=2)) // timedelta(hours=2)
    sum = sum + itr
    for j in range(itr):
        s = 2*(j+1)
        e = s+2
        nomealT1.loc[_] = nomeals1.iloc[i]+timedelta(hours=s)
        _ += 1
        filt = (cgmD['Date'] > (nomeals1.iloc[i]+timedelta(hours=s))) & (cgmD['Date'] <= (nomeals1.iloc[i]+timedelta(hours=e)))
        temp = cgmD.loc[filt, 'Sensor Glucose (mg/dL)'].tolist()
        temp.reverse()
        datalstnm.append(temp)

nomealT2 = pd.Series()
_ = 0
sum = 0
for i in range(len(nomeals2)-1):
    itr = (nomeals2.iloc[i+1]-nomeals2.iloc[i]-timedelta(hours=2)) // timedelta(hours=2)
    sum = sum + itr
    for j in range(itr):
        s = 2*(j+1)
        e = s+2
        nomealT2.loc[_] = nomeals2.iloc[i]+timedelta(hours=s)
        _ += 1
        filt = (cgmD2['Date'] > (nomeals2.iloc[i]+timedelta(hours=s))) & (cgmD2['Date'] <= (nomeals2.iloc[i]+timedelta(hours=e)))
        temp = cgmD2.loc[filt, 'Sensor Glucose (mg/dL)'].tolist()
        temp.reverse()
        datalstnm.append(temp)
nomealD = pd.DataFrame(datalstnm, dtype = 'float32')
nomealD.drop([i for i in range(24,37)],axis='columns', inplace = True)
dlennm = [len(i) for i in datalstnm]

nomealT = nomealT1.append(nomealT2, ignore_index = True)










Threshold = 0.15

meal_ncount = mealD.isnull().sum(axis=1)
nmeal_ncount = nomealD.isnull().sum(axis=1)


dropi_meal = meal_ncount.loc[meal_ncount > (Threshold*30)].index
mealD.drop(dropi_meal, inplace=True)
meals.drop(dropi_meal, inplace=True)

dropi_nomeal = nmeal_ncount.loc[nmeal_ncount > (Threshold*24)].index
nomealD.drop(dropi_nomeal, inplace=True)
nomealT.drop(dropi_nomeal, inplace=True)

for i in range(len(mealD)):
    mealD.iloc[i].fillna(method='ffill', inplace=True)
    mealD.iloc[i].fillna(method='bfill', inplace=True)
    
for i in range(len(nomealD)):
    nomealD.iloc[i].fillna(method='ffill', inplace=True)
    nomealD.iloc[i].fillna(method='bfill', inplace=True)



#plt.plot(20*np.log10(np.abs(np.fft.rfft(nomealD.iloc[0]))))



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
  
#findLocalMaximaMinima(len(20*np.log10(np.abs(np.fft.rfft(mealD.iloc[3])))), 20*np.log10(np.abs(np.fft.rfft(mealD.iloc[3]))))


mealFeat = []
for i in range(len(mealD)):
    
    # To calculate T
    high = max(mealD.iloc[i].loc[mealD.iloc[i].index > 6])
    top = max(mealD.iloc[i])
    peak = mealD.iloc[i].loc[mealD.iloc[i] == high].index[0]
    T = (peak-6)*5
    
    # To calculate dG normalized
    dGn = (high - mealD.iloc[i].loc[6]) / mealD.iloc[i].loc[6]
    
    # Fast Fourier Transform features
    farr = 20*np.log10(np.abs(np.fft.fft(mealD.iloc[i])))
    (mx,mn) = findLocalMaximaMinima(farr)
    
    f1 = mx[1]
    pf1 = farr[f1]
    
    # if len(mx) <= 2:
    #     f2 = mx[1] - 1
    # else:
    #     f2 = mx[2]
    # pf2 = farr[f2]
    
    
    # differential
    
    d = diff(mealD.iloc[i])/1
    d1 = max(d) - min(d)
    dd = diff(d)/1
    d2 = max(dd) - min(dd)
    
    # Standard Deviation
    stD = statistics.stdev(mealD.iloc[i])
    
    
    featset = np.array([top,T,dGn,d1,d2,stD,f1,pf1])
    featset.astype('float32')
    mealFeat.append(featset)
    
mealF = pd.DataFrame(mealFeat, dtype = 'float32')
    



nomealFeat = []
for i in range(len(nomealD)):
    
    # To calculate T
    high = max(nomealD.iloc[i])
    peak = nomealD.iloc[i].loc[nomealD.iloc[i] == high].index[0]
    T = (peak)*5
    
    # To calculate dG normalized
    dGn = (high - nomealD.iloc[i].loc[0]) / nomealD.iloc[i].loc[0]
    
    # Fast Fourier Transform features
    farr = 20*np.log10(np.abs(np.fft.fft(nomealD.iloc[i])))
    (mx,mn) = findLocalMaximaMinima(farr)
    
    f1 = mx[1]
    pf1 = farr[f1]
    
    # if len(mx) <= 2:
    #     f2 = mx[1] - 1
    # else:
    #     f2 = mx[2]
    # pf2 = farr[f2]
    
    
    # differential
    
    d = diff(nomealD.iloc[i])/1
    d1 = max(d) - min(d)
    dd = diff(d)/1
    d2 = max(dd) - min(dd)
    
     # Standard Deviation
    stD = statistics.stdev(nomealD.iloc[i])
    
    
    featset = np.array([high,T,dGn,d1,d2,stD,f1,pf1])
    featset.astype('float32')
    nomealFeat.append(featset)
    
nomealF = pd.DataFrame(nomealFeat, dtype = 'float32')













Dataset = mealF.append(nomealF, ignore_index = True)
labelV = pd.Series(np.append(np.ones(818), np.zeros(2370)))

X_train, X_test, y_train, y_test = train_test_split(Dataset, labelV, test_size=0.20)


# classifier = DecisionTreeClassifier()
# classifier.fit(X_train, y_train)


# from sklearn.linear_model import LogisticRegression
# LR = LogisticRegression().fit(X_train, y_train)


from sklearn.ensemble import GradientBoostingClassifier
GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(X_train, y_train)



#y_pred = classifier.predict(X_test)
#y1_pred = LR.predict(X_test)
y2_pred = GBC.predict(X_test)


pickle.dump(GBC, open('final_model.sav', 'wb'))


#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))





















