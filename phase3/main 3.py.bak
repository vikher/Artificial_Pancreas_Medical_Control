#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 18:50:12 2021

@author: prantogg
"""

import pandas as pd
import numpy as np
import statistics
from datetime import timedelta
from numpy import diff
from sklearn.cluster import KMeans
from numpy import asarray
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import contingency_matrix










df = pd.read_csv('InsulinData.csv')
insulinD = df.filter(['Index','Date','Time','BWZ Carb Input (grams)'])
del df

df = pd.read_csv('CGMData.csv')
cgmD = df.filter(['Index', 'Date', 'Time', 'Sensor Glucose (mg/dL)'])
del df

cgmD['Date'] = cgmD['Date'] + ' ' + cgmD['Time']
cgmD['Date'] = pd.to_datetime(cgmD['Date'])
cgmD.set_index('Date', inplace=True)
cgmD.drop(['Index','Time'], axis = 1, inplace = True)


insulinD['Date'] = insulinD['Date'] + ' ' + insulinD['Time']
insulinD['Date'] = pd.to_datetime(insulinD['Date'])
insulinD.set_index('Date', inplace=True)
insulinD.drop(['Index','Time'], axis = 1, inplace = True)
insulinD['BWZ Carb Input (grams)'].replace(0, np.nan, inplace = True)





mealtimes = insulinD.loc[insulinD['BWZ Carb Input (grams)'].notnull()]
mealtimes = pd.Series(mealtimes.index)
pd.to_datetime(mealtimes)

mealT = pd.Series()
_ = 0
for i in range((len(mealtimes)-1),0,-1):
    if (mealtimes.iloc[i-1] - mealtimes.iloc[i]) < timedelta(hours=2):
        pass
    else:
        mealT.loc[_] = mealtimes.iloc[i]
        _ += 1    
mealT.loc[_] = mealtimes.iloc[0]
pd.to_datetime(mealT)





datalst = []
for i in range(len(mealT)):
    filt = (cgmD.index > (mealT.iloc[i]-timedelta(hours=0.5))) & (cgmD.index <= (mealT.iloc[i]+timedelta(hours=2)))
    temp = cgmD.loc[filt, 'Sensor Glucose (mg/dL)'].tolist()
    temp.reverse()
    datalst.append(temp)
mealD = pd.DataFrame(datalst, dtype = 'float32')
dlen = [len(i) for i in datalst]




Threshold = 0.15
meal_ncount = mealD.isnull().sum(axis=1)
dropi_meal = meal_ncount.loc[meal_ncount > (Threshold*30)].index
mealD.drop(dropi_meal, inplace=True)
mealT.drop(dropi_meal, inplace=True)
for i in range(len(mealD)):
    mealD.iloc[i].fillna(method='ffill', inplace=True)
    mealD.iloc[i].fillna(method='bfill', inplace=True)
    



mealCarb = insulinD.loc[mealT]
mealCarb.dropna(inplace=True)
mealCarb.columns = ['Carbs']





def findLocalMaximaMinima(arr):  
    mx = []  
    mn = []
    n = len(arr) 
    if(arr[0] > arr[1]):  
        mx.append(0)  
    elif(arr[0] < arr[1]):  
        mn.append(0)  
    for i in range(1, n-1):  
        if(arr[i-1] > arr[i] < arr[i + 1]):  
            mn.append(i)  
        elif(arr[i-1] < arr[i] > arr[i + 1]):  
            mx.append(i)  
    if(arr[-1] > arr[-2]):  
        mx.append(n-1)  
    elif(arr[-1] < arr[-2]):  
        mn.append(n-1) 
    return (mx,mn)


mealFeat = []
for i in range(len(mealD)):
    
    # To calculate T
    high = max(mealD.iloc[i].loc[mealD.iloc[i].index > 6])
    top = max(mealD.iloc[i])
    peak = mealD.iloc[i].loc[mealD.iloc[i] == high].index[0]
    T = (peak-6)*5
    
    # To calculate dG normalized
    dGnN = (high - mealD.iloc[i].loc[6]) / mealD.iloc[i].loc[6]
    
    # To calculate dG
    dGn = (high - mealD.iloc[i].loc[6])
    
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
    
    
    featset = np.array([high,dGnN,d1,mealCarb.iloc[i]])
    mealFeat.append(featset)
    
mealF = pd.DataFrame(mealFeat, dtype = 'float32')



binNo = pd.Series()
_ = 0
for i in mealCarb.index:
    num = 1 + (mealCarb.loc[i, 'Carbs']-min(mealCarb['Carbs']))//20
    binNo.loc[_] = num
    _ += 1
binNo = binNo.astype('int_')




data = asarray(mealF.assign(Carbs = mealCarb['Carbs'].tolist()))
data = asarray(mealF)
scaler = StandardScaler()
# transform data
scaled = scaler.fit_transform(data)



# from sklearn.decomposition import PCA
# pca = PCA(2)
# df = pca.fit_transform(mealF)


km = KMeans(n_clusters=6, init='random', n_init=10, max_iter=300,)
km_labels = km.fit_predict(scaled)

dbsc = DBSCAN(eps = .38, min_samples = 6).fit(scaled)
db_labels = dbsc.fit_predict(scaled)
dbu_labels = np.unique(db_labels)

km_SSE = km.inertia_

db_centroids = [[scaled[db_labels==i,0].mean(),scaled[db_labels==i,1].mean(),scaled[db_labels==i,2].mean(),scaled[db_labels==i,3].mean()] for i in range(0,6)]

dbclust_SSE = []
for i in range(len(dbu_labels)-1):
    sse = 0
    for j in scaled[db_labels==i]:
        sse = sse + np.sum(np.square(db_centroids[i] - j))
    dbclust_SSE.append(sse)

db_SSE = np.sum(dbclust_SSE)


km_cont = pd.DataFrame(contingency_matrix(km_labels, binNo))
db_cont = pd.DataFrame(contingency_matrix(db_labels, binNo))
db_cont.drop(0,axis=0,inplace=True)

kmclust_tot = [km_cont.iloc[i].sum() for i in range(km_cont.shape[0])]
dbclust_tot = [db_cont.iloc[i].sum() for i in range(db_cont.shape[0])]
km_tot = np.sum(kmclust_tot)
db_tot = np.sum(dbclust_tot)

kmclust_Entropy = []
for i in range(km_cont.shape[0]):
    ent = 0
    for j in km_cont.iloc[i]:
        if j==0:
            pass
        else:
            ent = ent + ((j/kmclust_tot[i])*np.log2(j/kmclust_tot[i]))
    ent = ent * (-1)
    kmclust_Entropy.append(ent)
km_Entropy = 0
for i in kmclust_Entropy:
        if i==0:
            pass
        else:
            km_Entropy = km_Entropy + ((i/km_tot)*np.log2(i/km_tot))
km_Entropy = km_Entropy * (-1)


dbclust_Entropy = []
for i in range(db_cont.shape[0]):
    ent = 0
    for j in db_cont.iloc[i]:
        if j==0:
            pass
        else:
            ent = ent + ((j/dbclust_tot[i])*np.log2(j/dbclust_tot[i]))
    ent = ent * (-1)
    dbclust_Entropy.append(ent)
db_Entropy = 0
for i in dbclust_Entropy:
        if i==0:
            pass
        else:
            db_Entropy = db_Entropy + ((i/db_tot)*np.log2(i/db_tot))
db_Entropy = db_Entropy * (-1)


km_Purity = np.sum([km_cont.iloc[i].max() for i in range(km_cont.shape[0])])/km_tot
db_Purity = np.sum([db_cont.iloc[i].max() for i in range(db_cont.shape[0])])/db_tot

result = pd.DataFrame([[km_SSE,db_SSE,km_Entropy,db_Entropy,km_Purity,db_Purity]] )
result.to_csv('Result.csv', index=False, header=False)









#Getting unique labels
 
# u_labels = np.unique(km_labels)
 
#plotting the results:
 
# for i in u_labels:
#     filtered_label = df[km_labels == i]
#     plt.scatter( filtered_label[:,0] , filtered_label[:,1], label = i)
#     plt.scatter(km.cluster_centers_[i,0],km.cluster_centers_[i,1], marker='*')
# plt.legend()
# plt.show()







