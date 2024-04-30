#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 12:17:09 2021

@author: prantogg
"""

import pandas as pd

df = pd.read_csv('CGMData.csv')
cgm = df.filter(['Index','Date','Time','Sensor Glucose (mg/dL)'])
del df
cgm['Datetime'] = cgm['Date'] + ' ' + cgm['Time']
cgm['Time'] = cgm['Date'] + ' ' + cgm['Time']
cgm['Datetime'] = pd.to_datetime(cgm['Datetime'])
cgm['Date'] = pd.to_datetime(cgm['Date'])
cgm['Time'] = pd.to_datetime(cgm['Time'])
#print(cgm)

df = pd.read_csv('InsulinData.csv')
idata = df.filter(['Index','Date','Time','Alarm'])
del df
idata['Datetime'] = idata['Date'] + ' ' + idata['Time']
idata['Time'] = idata['Date'] + ' ' + idata['Time']
idata['Datetime'] = pd.to_datetime(idata['Datetime'])
idata['Date'] = pd.to_datetime(idata['Date'])
idata['Time'] = pd.to_datetime(idata['Time'])
#print(idata)


# Extract Auto Mode time
filt = (idata['Alarm'] == 'AUTO MODE ACTIVE PLGM OFF')
x = idata[filt]
date = x.iloc[-1,-1]
del idata
#print(date)


# Segregate AUTO and MANUAL
filt = (cgm['Datetime'] > date)
cgm_a = cgm[filt].copy()
cgm_a.set_index('Datetime', inplace=True)
filt = (cgm['Datetime'] <= date)
cgm_m = cgm[filt].copy()
cgm_m.set_index('Datetime', inplace=True)
del cgm



# Remove Auto Mode data with < 80% data
nullValue = cgm_a['Sensor Glucose (mg/dL)'].notnull().resample('D').sum()
indexList = nullValue[nullValue < 231].index.tolist()
delList = []
for i in indexList:
    a = str(i)
    delList.append(a[:10])

for i in delList:
    killList = cgm_a[i].index.tolist()
    cgm_a.drop(killList, inplace=True)



# Remove Manual Mode data with < 80% data
nullValue = cgm_m['Sensor Glucose (mg/dL)'].notnull().resample('D').sum()
indexList = nullValue[nullValue < 231].index.tolist()
delList = []
for i in indexList:
    a = str(i)
    delList.append(a[:10])

for i in delList:
    killList = cgm_m[i].index.tolist()
    cgm_m.drop(killList, inplace=True)


del nullValue
del delList
del killList


Auto_data = pd.DataFrame()


# Populate NaN values and compute metrics for AUTO Mode
meanValue = cgm_a['Sensor Glucose (mg/dL)'].resample('D').mean()
indexList = meanValue.index.tolist()
for i in range(len(indexList)):
    filt = (cgm_a['Date'] == indexList[i])
    Whole_day =  cgm_a[filt].copy()
    Whole_day.fillna(meanValue[indexList[i]], inplace=True)
    
    date = str(indexList[i])[:10]
    dateDay6am = pd.Timestamp(date + ' 06:00:00')
    dateDay12am = pd.Timestamp(date + ' 23:59:59')
    dateNight12am = pd.Timestamp(date + ' 00:00:00')
    
    filt = (Whole_day['Time'] >= dateDay6am) & (Whole_day['Time'] <= dateDay12am)
    day = Whole_day[filt].copy()
    
    filt = (Whole_day['Time'] >= dateNight12am) & (Whole_day['Time'] < dateDay6am)
    night = Whole_day[filt].copy()
    
    
    filt = night['Sensor Glucose (mg/dL)'] < 54
    N_m6 = len(night[filt])/2.88
    filt = night['Sensor Glucose (mg/dL)'] < 70
    N_m5 = len(night[filt])/2.88
    filt = (night['Sensor Glucose (mg/dL)'] >= 70) & (night['Sensor Glucose (mg/dL)'] <= 150)
    N_m4 = len(night[filt])/2.88
    filt = (night['Sensor Glucose (mg/dL)'] >= 70) & (night['Sensor Glucose (mg/dL)'] <= 180)
    N_m3 = len(night[filt])/2.88
    filt = night['Sensor Glucose (mg/dL)'] > 250
    N_m2 = len(night[filt])/2.88
    filt = night['Sensor Glucose (mg/dL)'] > 180
    N_m1 = len(night[filt])/2.88
    
    
    filt = day['Sensor Glucose (mg/dL)'] < 54
    D_m6 = len(day[filt])/2.88
    filt = day['Sensor Glucose (mg/dL)'] < 70
    D_m5 = len(day[filt])/2.88
    filt = (day['Sensor Glucose (mg/dL)'] >= 70) & (day['Sensor Glucose (mg/dL)'] <= 150)
    D_m4 = len(day[filt])/2.88
    filt = (day['Sensor Glucose (mg/dL)'] >= 70) & (day['Sensor Glucose (mg/dL)'] <= 180)
    D_m3 = len(day[filt])/2.88
    filt = day['Sensor Glucose (mg/dL)'] > 250
    D_m2 = len(day[filt])/2.88
    filt = day['Sensor Glucose (mg/dL)'] > 180
    D_m1 = len(day[filt])/2.88
    
    
    filt = Whole_day['Sensor Glucose (mg/dL)'] < 54
    W_m6 = len(Whole_day[filt])/2.88
    filt = Whole_day['Sensor Glucose (mg/dL)'] < 70
    W_m5 = len(Whole_day[filt])/2.88
    filt = (Whole_day['Sensor Glucose (mg/dL)'] >= 70) & (Whole_day['Sensor Glucose (mg/dL)'] <= 150)
    W_m4 = len(Whole_day[filt])/2.88
    filt = (Whole_day['Sensor Glucose (mg/dL)'] >= 70) & (Whole_day['Sensor Glucose (mg/dL)'] <= 180)
    W_m3 = len(Whole_day[filt])/2.88
    filt = Whole_day['Sensor Glucose (mg/dL)'] > 250
    W_m2 = len(Whole_day[filt])/2.88
    filt = Whole_day['Sensor Glucose (mg/dL)'] > 180
    W_m1 = len(Whole_day[filt])/2.88
    
    
    Auto_data[i] = [N_m1, N_m2, N_m3, N_m4, N_m5, N_m6, D_m1, D_m2, D_m3, D_m4, D_m5, D_m6, W_m1, W_m2, W_m3, W_m4, W_m5, W_m6]
    
    
#Auto_data.to_csv('Auto_data.csv')





Manual_data = pd.DataFrame()


# Populate NaN values and compute metrics for AUTO Mode
meanValue = cgm_m['Sensor Glucose (mg/dL)'].resample('D').mean()
indexList = meanValue.index.tolist()
for i in range(len(indexList)):
    filt = (cgm_m['Date'] == indexList[i])
    Whole_day =  cgm_m[filt].copy()
    Whole_day.fillna(meanValue[indexList[i]], inplace=True)
    
    date = str(indexList[i])[:10]
    dateDay6am = pd.Timestamp(date + ' 06:00:00')
    dateDay12am = pd.Timestamp(date + ' 23:59:59')
    dateNight12am = pd.Timestamp(date + ' 00:00:00')
    
    filt = (Whole_day['Time'] >= dateDay6am) & (Whole_day['Time'] <= dateDay12am)
    day = Whole_day[filt].copy()
    
    filt = (Whole_day['Time'] >= dateNight12am) & (Whole_day['Time'] < dateDay6am)
    night = Whole_day[filt].copy()
    
    
    filt = night['Sensor Glucose (mg/dL)'] < 54
    N_m6 = len(night[filt])/2.88
    filt = night['Sensor Glucose (mg/dL)'] < 70
    N_m5 = len(night[filt])/2.88
    filt = (night['Sensor Glucose (mg/dL)'] >= 70) & (night['Sensor Glucose (mg/dL)'] <= 150)
    N_m4 = len(night[filt])/2.88
    filt = (night['Sensor Glucose (mg/dL)'] >= 70) & (night['Sensor Glucose (mg/dL)'] <= 180)
    N_m3 = len(night[filt])/2.88
    filt = night['Sensor Glucose (mg/dL)'] > 250
    N_m2 = len(night[filt])/2.88
    filt = night['Sensor Glucose (mg/dL)'] > 180
    N_m1 = len(night[filt])/2.88
    
    
    filt = day['Sensor Glucose (mg/dL)'] < 54
    D_m6 = len(day[filt])/2.88
    filt = day['Sensor Glucose (mg/dL)'] < 70
    D_m5 = len(day[filt])/2.88
    filt = (day['Sensor Glucose (mg/dL)'] >= 70) & (day['Sensor Glucose (mg/dL)'] <= 150)
    D_m4 = len(day[filt])/2.88
    filt = (day['Sensor Glucose (mg/dL)'] >= 70) & (day['Sensor Glucose (mg/dL)'] <= 180)
    D_m3 = len(day[filt])/2.88
    filt = day['Sensor Glucose (mg/dL)'] > 250
    D_m2 = len(day[filt])/2.88
    filt = day['Sensor Glucose (mg/dL)'] > 180
    D_m1 = len(day[filt])/2.88
    
    
    filt = Whole_day['Sensor Glucose (mg/dL)'] < 54
    W_m6 = len(Whole_day[filt])/2.88
    filt = Whole_day['Sensor Glucose (mg/dL)'] < 70
    W_m5 = len(Whole_day[filt])/2.88
    filt = (Whole_day['Sensor Glucose (mg/dL)'] >= 70) & (Whole_day['Sensor Glucose (mg/dL)'] <= 150)
    W_m4 = len(Whole_day[filt])/2.88
    filt = (Whole_day['Sensor Glucose (mg/dL)'] >= 70) & (Whole_day['Sensor Glucose (mg/dL)'] <= 180)
    W_m3 = len(Whole_day[filt])/2.88
    filt = Whole_day['Sensor Glucose (mg/dL)'] > 250
    W_m2 = len(Whole_day[filt])/2.88
    filt = Whole_day['Sensor Glucose (mg/dL)'] > 180
    W_m1 = len(Whole_day[filt])/2.88
    
    
    Manual_data[i] = [N_m1, N_m2, N_m3, N_m4, N_m5, N_m6, D_m1, D_m2, D_m3, D_m4, D_m5, D_m6, W_m1, W_m2, W_m3, W_m4, W_m5, W_m6]
    
    
#Manual_data.to_csv('Manual_data.csv')
metrics = [list(Manual_data.mean(axis=1))]
metrics.append(list(Auto_data.mean(axis=1)))

output = pd.DataFrame(metrics)
output.to_csv('Results.csv', index=False, header=False)







