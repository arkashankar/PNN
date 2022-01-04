# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 22:08:55 2019

@author: Vikram
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import statistics
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import math
from sklearn.model_selection import KFold 

dataset=pd.read_excel('process safety dataset.xlsx')
X=dataset.iloc[:,1:7].values
dfx = pd.DataFrame(X)

XD = X.copy()
dfxd = pd.DataFrame(XD)

for j in range(6):
    if j==0:
        for i in range(len(XD)):
            if XD[i][j]=='Incident (injury/ Property Damage etc)':
                XD[i][j]=1
            elif XD[i][j]=='Near Miss':
                XD[i,j]=2
                
    elif j==1:
        for i in range(len(XD)):
            if XD[i][j]=='Inside Road':
                XD[i][j]=1
            elif XD[i][j]=='Mines':
                XD[i][j]=2
            elif XD[i][j]=='Outside Works':
                XD[i][j]=3
            elif XD[i][j]=='Workplace':
                XD[i][j]=4
                
    elif j==2:
        for i in range(len(XD)):
            if XD[i][j]=='Environment discharge':
                XD[i][j]=1
            elif XD[i][j]=='Equipment/Property Damage':
                XD[i][j]=2
            elif XD[i][j]=='injury':
                XD[i][j]=3
            elif XD[i][j]=='Medical Case':
                XD[i][j]=4
            elif XD[i][j]=='Toxic Release':
                XD[i][j]=5
                
    elif j==3:
        for i in range(len(XD)):
            if XD[i][j]=='activity related hazard':
                XD[i][j]=1
            elif XD[i][j]=='chemical':
                XD[i][j]=2
            elif XD[i][j]=='coke':
                XD[i][j]=3
            elif XD[i][j]=='dust and steam':
                XD[i][j]=4
            elif XD[i][j]=='gas':
                XD[i][j]=5
            elif XD[i][j]=='High pressure material':
                XD[i][j]=6
            elif XD[i][j]=='Hot metal/ steel/slag':
                XD[i][j]=7
            elif XD[i][j]=='Uncontained water/oil':
                XD[i][j]=8
                
    elif j==4:
        for i in range(len(XD)):
            if XD[i][j]=='Abnormal process parameter':
                XD[i][j]=1
            elif XD[i][j]=='damaged/degraded/poorly maintained equipment':
                XD[i,j]=2
            elif XD[i][j]=='failure/malfunctioning of equipment':
                XD[i,j]=3
            elif XD[i][j]=='improper material/material quality/equipment design':
                XD[i,j]=4
            elif XD[i][j]=='improper work by operator/worker':
                XD[i,j]=5
            elif XD[i][j]=='improper working environment':
                XD[i,j]=6
            elif XD[i][j]=='improper/exceeded process/operating parameters':
                XD[i,j]=7
            elif XD[i][j]=='Inadequate/ unavailable SOP':
                XD[i,j]=8
            elif XD[i][j]=='lack of commnication/supervision':
                XD[i,j]=9
            elif XD[i][j]=='presence of unwanted/flammable material':
                XD[i,j]=10
                
    elif j==5:
        for i in range(len(XD)):
            if XD[i][j]=='dashing/collision':
                XD[i][j]=1
            elif XD[i][j]=='degradation/breaking of equiptment':
                XD[i][j]=2
            elif XD[i][j]=='derailment':
                XD[i,j]=3
            elif XD[i][j]=='Disturbance in process parameters/equipment shutdown':
                XD[i,j]=4
            elif XD[i][j]=='explosion':
                XD[i,j]=5
            elif XD[i][j]=='exposure to toxic gas/ high temperature':
                XD[i,j]=6
            elif XD[i][j]=='fall of material':
                XD[i,j]=7
            elif XD[i][j]=='Fire/ flame generation':
                XD[i,j]=8
            elif XD[i][j]=='Leakage of hazardous material':
                XD[i,j]=9
            elif XD[i][j]=='Overflow of hazardous material':
                XD[i,j]=10
            elif XD[i][j]=='spillage of hazardous material':
                XD[i,j]=11
            elif XD[i][j]=='splash':
                XD[i,j]=12
            elif XD[i][j]=='water logging/flooding situation':
                XD[i,j]=13

#making X_complete to get true vales from it
X_complete = XD.copy()
dfxcomplete = pd.DataFrame(X_complete)

for j in range(6):
    for i in range(10):
        for k in range(4):
            y=random.randint(i*60, 60+i*60)
            XD[y][j]='?'

XCR = []
XIR = []

for i in range(len(XD)):
    for j in range(0,6):
        if XD[i][j]=='?':
            XIR.append(XD[i][:].copy())
            j=j-1
            break
    if j==5:
        XCR.append(XD[i][:].copy())

dfxcr=pd.DataFrame(XCR)
dfxir=pd.DataFrame(XIR)

#forming a mode list containing mode values for each attrubute
mode = []

for i in range (0,6):
    z = statistics.mode(XD[:,i])
    mode.append(z)

XIM = []

for i in range(len(XIR)):
    XIM.append(XIR[i][:].copy())

#Filling all the missing values in XIM with the mode value of corresponding attribute
for i in range(len(XIM)):
    for j in range(6):
        if XIM[i][j]=='?':
            XIM[i][j]=mode[j]

dfxim = pd.DataFrame(XIM)

X1 = []
for i in range(len(XIR)):
    if XIR[i][0]=='?':
        X1.append(XIR[i][:].copy())
        

for i in range(len(X1)):
    for j in range(1,6):
        if X1[i][j]=='?':
            X1[i][j]=mode[j]

dfx1=pd.DataFrame(X1)

X_true = []

for i in range(len(XD)):
    if XD[i][0]=='?':
        X_true.append(X_complete[i][0])
        
dfxtrue = pd.DataFrame(X_true)

XPR = []
for i in range(len(X1)):
    XPR.append(X1[i][1:6].copy())

dfxpr = pd.DataFrame(XPR)

XTR = []
for i in range(len(XCR)):
    XTR.append(XCR[i][1:6].copy())
    
dfxtr = pd.DataFrame(XTR)

#Applying LabelEncoder and OneHotEncoder to convert XTR into Indicator Matrix
labelencoder = LabelEncoder()
XTR[:][4] = labelencoder.fit_transform(XTR[:][4])
onehotencoder = OneHotEncoder(categorical_features = [4])
XTR = onehotencoder.fit_transform(XTR).toarray()
dfxtr = pd.DataFrame(XTR)

XTR[:][16] = labelencoder.fit_transform(XTR[:][16])
onehotencoder = OneHotEncoder(categorical_features = [16])
XTR = onehotencoder.fit_transform(XTR).toarray()
dfxtr = pd.DataFrame(XTR)

XTR[:][25] = labelencoder.fit_transform(XTR[:][25])
onehotencoder = OneHotEncoder(categorical_features = [25])
XTR = onehotencoder.fit_transform(XTR).toarray()
dfxtr = pd.DataFrame(XTR)

XTR[:][32] = labelencoder.fit_transform(XTR[:][32])
onehotencoder = OneHotEncoder(categorical_features = [32])
XTR = onehotencoder.fit_transform(XTR).toarray()
dfxtr = pd.DataFrame(XTR)

XTR[:][35] = labelencoder.fit_transform(XTR[:][35])
onehotencoder = OneHotEncoder(categorical_features = [35])
XTR = onehotencoder.fit_transform(XTR).toarray()
dfxtr = pd.DataFrame(XTR)

#making X_M for encoding XIR replaced with mode values
X_M = []

for i in range(len(XIR)):
    X_M.append(XIR[i][1:].copy())
    
for i in range(len(X_M)):
    for j in range(5):
        if X_M[i][j]=='?':
            X_M[i][j]=mode[j+1]

dfxm=pd.DataFrame(X_M)

X_M[:][4] = labelencoder.fit_transform(X_M[:][4])
onehotencoder = OneHotEncoder(categorical_features = [4])
X_M = onehotencoder.fit_transform(X_M).toarray()
dfxm = pd.DataFrame(X_M)

X_M[:][16] = labelencoder.fit_transform(X_M[:][16])
onehotencoder = OneHotEncoder(categorical_features = [16])
X_M = onehotencoder.fit_transform(X_M).toarray()
dfxm = pd.DataFrame(X_M)

X_M[:][24] = labelencoder.fit_transform(X_M[:][24])
onehotencoder = OneHotEncoder(categorical_features = [24])
X_M = onehotencoder.fit_transform(X_M).toarray()
dfxm = pd.DataFrame(X_M)

X_M[:][31] = labelencoder.fit_transform(X_M[:][31])
onehotencoder = OneHotEncoder(categorical_features = [31])
X_M = onehotencoder.fit_transform(X_M).toarray()
dfxm = pd.DataFrame(X_M)

X_M[:][34] = labelencoder.fit_transform(X_M[:][34])
onehotencoder = OneHotEncoder(categorical_features = [34])
X_M = onehotencoder.fit_transform(X_M).toarray()
dfxm = pd.DataFrame(X_M)

Xa = []
Xb = []
for i in range(len(XCR)):
    if XCR[i][0]==1:
        Xa.append(XTR[i][:].copy())
    else:
        Xb.append(XTR[i][:].copy())
        
dfxa = pd.DataFrame(Xa)
dfxb = pd.DataFrame(Xb)

from sklearn.model_selection import KFold 
kf= KFold(n_splits=10, random_state=42, shuffle=False)

PCP1 = []

X1_pred = []
#sigma = [0.0001,0.0011,0.0081,0.0011,0.2831]
#sigma = [0.7,0.9999,0.1291,0.8161,0.9999]
#sigma = [0.35005,0.50005,0.0686,0.4086,0.6415]
sigma = 0.5
p = 2
pi=3.14
all_pred_values = []
z = 0
for train_index, test_index in kf.split(X_M):
    X_test = []
    for i in range(test_index[0],test_index[0]+len(test_index)):
        if XIR[i][0]=='?':
            X_test.append(X_M[i][:].copy())
    dfx_test = pd.DataFrame(X_test)       

    for k in range(len(X_test)):
        sum_of_each_class = {}
        
        sum_of_all_pattern_nodes=0
        for j in range(len(Xa)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxa.iloc[j][m])*(dfx_test.iloc[k][m]-dfxa.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes += temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xa))
        sum_of_each_class[0] = sum_of_all_pattern_nodes
        
        sum_of_all_pattern_nodes=0
        for j in range(0,len(Xb)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxb.iloc[j][m])*(dfx_test.iloc[k][m]-dfxb.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes+=temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xb))
        sum_of_each_class[1] = sum_of_all_pattern_nodes
    
        max_sum_among_all_classes = max(sum_of_each_class[0],sum_of_each_class[1])
        if max_sum_among_all_classes==sum_of_each_class[0] :
            classified_class=1
        elif max_sum_among_all_classes==sum_of_each_class[1] :
            classified_class=2
        
        all_pred_values.append(classified_class)
        
    correct_prediction1=0
    incorrect_prediction1=0
    for i in range(z,z+len(X_test)):
        if all_pred_values[i]==X_true[i]:
            correct_prediction1+=1
        else:
            incorrect_prediction1+=1
    
    z+=len(X_test)
    percentage=correct_prediction1*100/(correct_prediction1+incorrect_prediction1)
    PCP1.append(percentage)

dfpcp1 = pd.DataFrame(PCP1)




#Taking attribute 2 as output
#making X_M for encoding XIR replaced with mode values
XTR = []
for i in range(len(XCR)):
    XTR.append(XCR[i][[0,2,3,4,5]].copy())
    
XTR[:][4] = labelencoder.fit_transform(XTR[:][4])
onehotencoder = OneHotEncoder(categorical_features = [4])
XTR = onehotencoder.fit_transform(XTR).toarray()
dfxtr = pd.DataFrame(XTR)
 
XTR[:][16] = labelencoder.fit_transform(XTR[:][16])
onehotencoder = OneHotEncoder(categorical_features = [16])
XTR = onehotencoder.fit_transform(XTR).toarray()
dfxtr = pd.DataFrame(XTR)  

XTR[:][25] = labelencoder.fit_transform(XTR[:][25])
onehotencoder = OneHotEncoder(categorical_features = [25])
XTR = onehotencoder.fit_transform(XTR).toarray()
dfxtr = pd.DataFrame(XTR)

XTR[:][32] = labelencoder.fit_transform(XTR[:][32])
onehotencoder = OneHotEncoder(categorical_features = [32])
XTR = onehotencoder.fit_transform(XTR).toarray()
dfxtr = pd.DataFrame(XTR)

XTR[:][36] = labelencoder.fit_transform(XTR[:][36])
onehotencoder = OneHotEncoder(categorical_features = [36])
XTR = onehotencoder.fit_transform(XTR).toarray()
dfxtr = pd.DataFrame(XTR)

X_M = []

for i in range(len(XIR)):
    X_M.append(XIR[i][[0,2,3,4,5]].copy())
 
for i in range(len(X_M)):
    if X_M[i][0]=='?':
        X_M[i][0]=mode[0]
    
for i in range(len(X_M)):
    for j in range(1,5):
        if X_M[i][j]=='?':
            X_M[i][j]=mode[j+1]

dfxm=pd.DataFrame(X_M)

X_M[:][4] = labelencoder.fit_transform(X_M[:][4])
onehotencoder = OneHotEncoder(categorical_features = [4])
X_M = onehotencoder.fit_transform(X_M).toarray()
dfxm = pd.DataFrame(X_M)

X_M[:][16] = labelencoder.fit_transform(X_M[:][16])
onehotencoder = OneHotEncoder(categorical_features = [16])
X_M = onehotencoder.fit_transform(X_M).toarray()
dfxm = pd.DataFrame(X_M)

X_M[:][24] = labelencoder.fit_transform(X_M[:][24])
onehotencoder = OneHotEncoder(categorical_features = [24])
X_M = onehotencoder.fit_transform(X_M).toarray()
dfxm = pd.DataFrame(X_M)

X_M[:][31] = labelencoder.fit_transform(X_M[:][31])
onehotencoder = OneHotEncoder(categorical_features = [31])
X_M = onehotencoder.fit_transform(X_M).toarray()
dfxm = pd.DataFrame(X_M)

X_M[:][34] = labelencoder.fit_transform(X_M[:][34])
onehotencoder = OneHotEncoder(categorical_features = [34])
X_M = onehotencoder.fit_transform(X_M).toarray()
dfxm = pd.DataFrame(X_M)

Xa = []
Xb = []
Xc = []
Xd = []
for i in range(len(XCR)):
    if XCR[i][1]==1:
        Xa.append(XTR[i][:].copy())
    elif XCR[i][1]==2:
        Xb.append(XTR[i][:].copy())
    elif XCR[i][1]==3:
        Xc.append(XTR[i][:].copy())
    elif XCR[i][1]==4:
        Xd.append(XTR[i][:].copy())
        
dfxa = pd.DataFrame(Xa)
dfxb = pd.DataFrame(Xb)
dfxc = pd.DataFrame(Xc)
dfxd = pd.DataFrame(Xd)

X_true = []

for i in range(len(XD)):
    if XD[i][1]=='?':
        X_true.append(X_complete[i][1])
        
dfxtrue = pd.DataFrame(X_true)

from sklearn.model_selection import KFold 
kf= KFold(n_splits=10, random_state=42, shuffle=False)

PCP2 = []

X1_pred = []
#sigma = [0.0001,0.0011,0.0081,0.0011,0.2831]
#sigma = [0.7,0.9999,0.1291,0.8161,0.9999]
#sigma = [0.35005,0.50005,0.0686,0.4086,0.6415]
sigma = 0.5
p = 2
pi=3.14
all_pred_values = []
z = 0
for train_index, test_index in kf.split(X_M):
    X_test = []
    for i in range(test_index[0],test_index[0]+len(test_index)):
        if XIR[i][1]=='?':
            X_test.append(X_M[i][:].copy())
    dfx_test = pd.DataFrame(X_test)       

    for k in range(len(X_test)):
        sum_of_each_class = {}
        
        if len(Xa)!=0:
            sum_of_all_pattern_nodes=0
            for j in range(len(Xa)):
                temp=0
                for m in range (len(X_test[0])):
                    temp+=(dfx_test.iloc[k][m]-dfxa.iloc[j][m])*(dfx_test.iloc[k][m]-dfxa.iloc[j][m])
                temp=-temp
                temp=temp/(2*sigma*sigma)
                temp=math.exp(temp)
                sum_of_all_pattern_nodes += temp
            sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xa))
            sum_of_each_class[0] = sum_of_all_pattern_nodes
        else:
            sum_of_each_class[0]=0
        
        sum_of_all_pattern_nodes=0
        for j in range(0,len(Xb)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxb.iloc[j][m])*(dfx_test.iloc[k][m]-dfxb.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes+=temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xb))
        sum_of_each_class[1] = sum_of_all_pattern_nodes
        
        sum_of_all_pattern_nodes=0
        for j in range(0,len(Xc)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxc.iloc[j][m])*(dfx_test.iloc[k][m]-dfxc.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes+=temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xc))
        sum_of_each_class[2] = sum_of_all_pattern_nodes
        
        sum_of_all_pattern_nodes=0
        for j in range(0,len(Xd)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxd.iloc[j][m])*(dfx_test.iloc[k][m]-dfxd.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes+=temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xd))
        sum_of_each_class[3] = sum_of_all_pattern_nodes
    
        max_sum_among_all_classes = max(sum_of_each_class[0],sum_of_each_class[1],sum_of_each_class[2],sum_of_each_class[3])
        if max_sum_among_all_classes==sum_of_each_class[0] :
            classified_class=1
        elif max_sum_among_all_classes==sum_of_each_class[1] :
            classified_class=2
        elif max_sum_among_all_classes==sum_of_each_class[2] :
            classified_class=3
        elif max_sum_among_all_classes==sum_of_each_class[3] :
            classified_class=4
        
        all_pred_values.append(classified_class)
        
    correct_prediction1=0
    incorrect_prediction1=0
    for i in range(z,z+len(X_test)):
        if all_pred_values[i]==X_true[i]:
            correct_prediction1+=1
        else:
            incorrect_prediction1+=1
    
    z+=len(X_test)
    percentage=correct_prediction1*100/(correct_prediction1+incorrect_prediction1)
    PCP2.append(percentage)

dfpcp2 = pd.DataFrame(PCP2)




#Taking attribute 3 as output
#making X_M for encoding XIR replaced with mode values
XTR = []
for i in range(len(XCR)):
    XTR.append(XCR[i][[0,1,3,4,5]].copy())
    
XTR[:][len(XTR[0])-1] = labelencoder.fit_transform(XTR[:][len(XTR[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(XTR[0])-1])
XTR = onehotencoder.fit_transform(XTR).toarray()
dfxtr = pd.DataFrame(XTR)
 
XTR[:][len(XTR[0])-1] = labelencoder.fit_transform(XTR[:][len(XTR[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(XTR[0])-1])
XTR = onehotencoder.fit_transform(XTR).toarray()
dfxtr = pd.DataFrame(XTR)

XTR[:][len(XTR[0])-1] = labelencoder.fit_transform(XTR[:][len(XTR[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(XTR[0])-1])
XTR = onehotencoder.fit_transform(XTR).toarray()
dfxtr = pd.DataFrame(XTR)

XTR[:][len(XTR[0])-1] = labelencoder.fit_transform(XTR[:][len(XTR[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(XTR[0])-1])
XTR = onehotencoder.fit_transform(XTR).toarray()
dfxtr = pd.DataFrame(XTR)

XTR[:][len(XTR[0])-1] = labelencoder.fit_transform(XTR[:][len(XTR[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(XTR[0])-1])
XTR = onehotencoder.fit_transform(XTR).toarray()
dfxtr = pd.DataFrame(XTR)

X_M = []

for i in range(len(XIR)):
    X_M.append(XIR[i][[0,1,3,4,5]].copy())
 
for i in range(len(X_M)):
    if X_M[i][0]=='?':
        X_M[i][0]=mode[0]
    if X_M[i][1]=='?':
        X_M[i][1]=mode[1]
    
for i in range(len(X_M)):
    for j in range(2,5):
        if X_M[i][j]=='?':
            X_M[i][j]=mode[j+1]

dfxm=pd.DataFrame(X_M)

X_M[:][len(X_M[0])-1] = labelencoder.fit_transform(X_M[:][len(X_M[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(X_M[0])-1])
X_M = onehotencoder.fit_transform(X_M).toarray()
dfxm = pd.DataFrame(X_M)

X_M[:][len(X_M[0])-1] = labelencoder.fit_transform(X_M[:][len(X_M[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(X_M[0])-1])
X_M = onehotencoder.fit_transform(X_M).toarray()
dfxm = pd.DataFrame(X_M)

X_M[:][len(X_M[0])-1] = labelencoder.fit_transform(X_M[:][len(X_M[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(X_M[0])-1])
X_M = onehotencoder.fit_transform(X_M).toarray()
dfxm = pd.DataFrame(X_M)

X_M[:][len(X_M[0])-1] = labelencoder.fit_transform(X_M[:][len(X_M[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(X_M[0])-1])
X_M = onehotencoder.fit_transform(X_M).toarray()
dfxm = pd.DataFrame(X_M)

X_M[:][len(X_M[0])-1] = labelencoder.fit_transform(X_M[:][len(X_M[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(X_M[0])-1])
X_M = onehotencoder.fit_transform(X_M).toarray()
dfxm = pd.DataFrame(X_M)

Xa = []
Xb = []
Xc = []
Xd = []
Xe = []
for i in range(len(XCR)):
    if XCR[i][2]==1:
        Xa.append(XTR[i][:].copy())
    elif XCR[i][2]==2:
        Xb.append(XTR[i][:].copy())
    elif XCR[i][2]==3:
        Xc.append(XTR[i][:].copy())
    elif XCR[i][2]==4:
        Xd.append(XTR[i][:].copy())
    elif XCR[i][2]==5:
        Xe.append(XTR[i][:].copy())
        
dfxa = pd.DataFrame(Xa)
dfxb = pd.DataFrame(Xb)
dfxc = pd.DataFrame(Xc)
dfxd = pd.DataFrame(Xd)
dfxe = pd.DataFrame(Xe)

X_true = []

for i in range(len(XD)):
    if XD[i][2]=='?':
        X_true.append(X_complete[i][2])
        
dfxtrue = pd.DataFrame(X_true)

from sklearn.model_selection import KFold 
kf= KFold(n_splits=10, random_state=42, shuffle=False)

PCP3 = []

X1_pred = []
#sigma = [0.0001,0.0011,0.0081,0.0011,0.2831]
#sigma = [0.7,0.9999,0.1291,0.8161,0.9999]
#sigma = [0.35005,0.50005,0.0686,0.4086,0.6415]
sigma = 0.5
p = 2
pi=3.14
all_pred_values = []
z = 0
for train_index, test_index in kf.split(X_M):
    X_test = []
    for i in range(test_index[0],test_index[0]+len(test_index)):
        if XIR[i][2]=='?':
            X_test.append(X_M[i][:].copy())
    dfx_test = pd.DataFrame(X_test)       

    for k in range(len(X_test)):
        sum_of_each_class = {}

        sum_of_each_class[0]=0
        sum_of_all_pattern_nodes=0
        for j in range(len(Xa)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxa.iloc[j][m])*(dfx_test.iloc[k][m]-dfxa.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes += temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xa))
        sum_of_each_class[0] = sum_of_all_pattern_nodes
        
        sum_of_each_class[1]=0
        sum_of_all_pattern_nodes=0
        for j in range(0,len(Xb)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxb.iloc[j][m])*(dfx_test.iloc[k][m]-dfxb.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes+=temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xb))
        sum_of_each_class[1] = sum_of_all_pattern_nodes
        
        sum_of_each_class[2]=0
        sum_of_all_pattern_nodes=0
        for j in range(0,len(Xc)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxc.iloc[j][m])*(dfx_test.iloc[k][m]-dfxc.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes+=temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xc))
        sum_of_each_class[2] = sum_of_all_pattern_nodes
        
        sum_of_each_class[3]=0
        sum_of_all_pattern_nodes=0
        for j in range(0,len(Xd)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxd.iloc[j][m])*(dfx_test.iloc[k][m]-dfxd.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes+=temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xd))
        sum_of_each_class[3] = sum_of_all_pattern_nodes
        
        sum_of_each_class[4]=0
        sum_of_all_pattern_nodes=0
        for j in range(0,len(Xe)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxe.iloc[j][m])*(dfx_test.iloc[k][m]-dfxe.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes+=temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xe))
        sum_of_each_class[3] = sum_of_all_pattern_nodes
    
        max_sum_among_all_classes = max(sum_of_each_class[0],sum_of_each_class[1],sum_of_each_class[2],sum_of_each_class[3],sum_of_each_class[4])
        if max_sum_among_all_classes==sum_of_each_class[0] :
            classified_class=1
        elif max_sum_among_all_classes==sum_of_each_class[1] :
            classified_class=2
        elif max_sum_among_all_classes==sum_of_each_class[2] :
            classified_class=3
        elif max_sum_among_all_classes==sum_of_each_class[3] :
            classified_class=4
        elif max_sum_among_all_classes==sum_of_each_class[4] :
            classified_class=5
        
        all_pred_values.append(classified_class)
        
    correct_prediction1=0
    incorrect_prediction1=0
    for i in range(z,z+len(X_test)):
        if all_pred_values[i]==X_true[i]:
            correct_prediction1+=1
        else:
            incorrect_prediction1+=1
    
    z+=len(X_test)
    percentage=correct_prediction1*100/(correct_prediction1+incorrect_prediction1)
    PCP3.append(percentage)

dfpcp3 = pd.DataFrame(PCP3)




#Taking attribute 4 as output
#making X_M for encoding XIR replaced with mode values
XTR = []
for i in range(len(XCR)):
    XTR.append(XCR[i][[0,1,2,4,5]].copy())
    
XTR[:][len(XTR[0])-1] = labelencoder.fit_transform(XTR[:][len(XTR[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(XTR[0])-1])
XTR = onehotencoder.fit_transform(XTR).toarray()
dfxtr = pd.DataFrame(XTR)
 
XTR[:][len(XTR[0])-1] = labelencoder.fit_transform(XTR[:][len(XTR[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(XTR[0])-1])
XTR = onehotencoder.fit_transform(XTR).toarray()
dfxtr = pd.DataFrame(XTR)

XTR[:][len(XTR[0])-1] = labelencoder.fit_transform(XTR[:][len(XTR[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(XTR[0])-1])
XTR = onehotencoder.fit_transform(XTR).toarray()
dfxtr = pd.DataFrame(XTR)

XTR[:][len(XTR[0])-1] = labelencoder.fit_transform(XTR[:][len(XTR[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(XTR[0])-1])
XTR = onehotencoder.fit_transform(XTR).toarray()
dfxtr = pd.DataFrame(XTR)

XTR[:][len(XTR[0])-1] = labelencoder.fit_transform(XTR[:][len(XTR[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(XTR[0])-1])
XTR = onehotencoder.fit_transform(XTR).toarray()
dfxtr = pd.DataFrame(XTR)

X_M = []

for i in range(len(XIR)):
    X_M.append(XIR[i][[0,1,2,4,5]].copy())
 
for i in range(len(X_M)):
    if X_M[i][0]=='?':
        X_M[i][0]=mode[0]
    if X_M[i][1]=='?':
        X_M[i][1]=mode[1]
    if X_M[i][2]=='?':
        X_M[i][2]=mode[2]
    
    
for i in range(len(X_M)):
    for j in range(3,5):
        if X_M[i][j]=='?':
            X_M[i][j]=mode[j+1]

dfxm=pd.DataFrame(X_M)

X_M[:][len(X_M[0])-1] = labelencoder.fit_transform(X_M[:][len(X_M[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(X_M[0])-1])
X_M = onehotencoder.fit_transform(X_M).toarray()
dfxm = pd.DataFrame(X_M)

X_M[:][len(X_M[0])-1] = labelencoder.fit_transform(X_M[:][len(X_M[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(X_M[0])-1])
X_M = onehotencoder.fit_transform(X_M).toarray()
dfxm = pd.DataFrame(X_M)

X_M[:][len(X_M[0])-1] = labelencoder.fit_transform(X_M[:][len(X_M[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(X_M[0])-1])
X_M = onehotencoder.fit_transform(X_M).toarray()
dfxm = pd.DataFrame(X_M)

X_M[:][len(X_M[0])-1] = labelencoder.fit_transform(X_M[:][len(X_M[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(X_M[0])-1])
X_M = onehotencoder.fit_transform(X_M).toarray()
dfxm = pd.DataFrame(X_M)

X_M[:][len(X_M[0])-1] = labelencoder.fit_transform(X_M[:][len(X_M[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(X_M[0])-1])
X_M = onehotencoder.fit_transform(X_M).toarray()
dfxm = pd.DataFrame(X_M)

Xa = []
Xb = []
Xc = []
Xd = []
Xe = []
Xf = []
Xg = []
Xh = []
for i in range(len(XCR)):
    if XCR[i][3]==1:
        Xa.append(XTR[i][:].copy())
    elif XCR[i][3]==2:
        Xb.append(XTR[i][:].copy())
    elif XCR[i][3]==3:
        Xc.append(XTR[i][:].copy())
    elif XCR[i][3]==4:
        Xd.append(XTR[i][:].copy())
    elif XCR[i][3]==5:
        Xe.append(XTR[i][:].copy())
    elif XCR[i][3]==6:
        Xf.append(XTR[i][:].copy())
    elif XCR[i][3]==7:
        Xg.append(XTR[i][:].copy())
    elif XCR[i][3]==8:
        Xh.append(XTR[i][:].copy())
        
dfxa = pd.DataFrame(Xa)
dfxb = pd.DataFrame(Xb)
dfxc = pd.DataFrame(Xc)
dfxd = pd.DataFrame(Xd)
dfxe = pd.DataFrame(Xe)
dfxf = pd.DataFrame(Xf)
dfxg = pd.DataFrame(Xg)
dfxh = pd.DataFrame(Xh)

X_true = []

for i in range(len(XD)):
    if XD[i][3]=='?':
        X_true.append(X_complete[i][3])
        
dfxtrue = pd.DataFrame(X_true)

from sklearn.model_selection import KFold 
kf= KFold(n_splits=10, random_state=42, shuffle=False)

PCP4 = []

X1_pred = []
#sigma = [0.0001,0.0011,0.0081,0.0011,0.2831]
#sigma = [0.7,0.9999,0.1291,0.8161,0.9999]
#sigma = [0.35005,0.50005,0.0686,0.4086,0.6415]
sigma = 0.5
p = 2
pi=3.14
all_pred_values = []
z = 0
for train_index, test_index in kf.split(X_M):
    X_test = []
    for i in range(test_index[0],test_index[0]+len(test_index)):
        if XIR[i][3]=='?':
            X_test.append(X_M[i][:].copy())
    dfx_test = pd.DataFrame(X_test)       

    for k in range(len(X_test)):
        sum_of_each_class = {}

        sum_of_each_class[0]=0
        sum_of_all_pattern_nodes=0
        for j in range(len(Xa)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxa.iloc[j][m])*(dfx_test.iloc[k][m]-dfxa.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes += temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xa))
        sum_of_each_class[0] = sum_of_all_pattern_nodes
        
        sum_of_each_class[1]=0
        sum_of_all_pattern_nodes=0
        for j in range(0,len(Xb)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxb.iloc[j][m])*(dfx_test.iloc[k][m]-dfxb.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes+=temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xb))
        sum_of_each_class[1] = sum_of_all_pattern_nodes
        
        sum_of_each_class[2]=0
        sum_of_all_pattern_nodes=0
        for j in range(0,len(Xc)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxc.iloc[j][m])*(dfx_test.iloc[k][m]-dfxc.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes+=temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xc))
        sum_of_each_class[2] = sum_of_all_pattern_nodes
        
        sum_of_each_class[3]=0
        sum_of_all_pattern_nodes=0
        for j in range(0,len(Xd)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxd.iloc[j][m])*(dfx_test.iloc[k][m]-dfxd.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes+=temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xb))
        sum_of_each_class[3] = sum_of_all_pattern_nodes
        
        sum_of_each_class[4]=0
        sum_of_all_pattern_nodes=0
        for j in range(0,len(Xe)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxe.iloc[j][m])*(dfx_test.iloc[k][m]-dfxe.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes+=temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xd))
        sum_of_each_class[3] = sum_of_all_pattern_nodes

        sum_of_each_class[5]=0
        sum_of_all_pattern_nodes=0
        for j in range(len(Xf)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxf.iloc[j][m])*(dfx_test.iloc[k][m]-dfxf.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes += temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xe))
        sum_of_each_class[5] = sum_of_all_pattern_nodes

        sum_of_each_class[6]=0
        sum_of_all_pattern_nodes=0
        for j in range(len(Xg)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxg.iloc[j][m])*(dfx_test.iloc[k][m]-dfxg.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes += temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xf))
        sum_of_each_class[6] = sum_of_all_pattern_nodes
        
        sum_of_each_class[7]=0
        sum_of_all_pattern_nodes=0
        for j in range(len(Xh)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxh.iloc[j][m])*(dfx_test.iloc[k][m]-dfxh.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes += temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xg))
        sum_of_each_class[7] = sum_of_all_pattern_nodes
    
        max_sum_among_all_classes = max(sum_of_each_class[0],sum_of_each_class[1],sum_of_each_class[2],sum_of_each_class[3],sum_of_each_class[4],sum_of_each_class[5],sum_of_each_class[6],sum_of_each_class[7])
        if max_sum_among_all_classes==sum_of_each_class[0] :
            classified_class=1
        elif max_sum_among_all_classes==sum_of_each_class[1] :
            classified_class=2
        elif max_sum_among_all_classes==sum_of_each_class[2] :
            classified_class=3
        elif max_sum_among_all_classes==sum_of_each_class[3] :
            classified_class=4
        elif max_sum_among_all_classes==sum_of_each_class[4] :
            classified_class=5
        elif max_sum_among_all_classes==sum_of_each_class[5] :
            classified_class=6
        elif max_sum_among_all_classes==sum_of_each_class[6] :
            classified_class=7
        elif max_sum_among_all_classes==sum_of_each_class[7] :
            classified_class=8
        
        all_pred_values.append(classified_class)
        
    correct_prediction1=0
    incorrect_prediction1=0
    for i in range(z,z+len(X_test)):
        if all_pred_values[i]==X_true[i]:
            correct_prediction1+=1
        else:
            incorrect_prediction1+=1
    
    z+=len(X_test)
    percentage=correct_prediction1*100/(correct_prediction1+incorrect_prediction1)
    PCP4.append(percentage)

dfpcp4 = pd.DataFrame(PCP4)


#Taking attribute 5 as output
#making X_M for encoding XIR replaced with mode values
XTR = []
for i in range(len(XCR)):
    XTR.append(XCR[i][[0,1,2,3,5]].copy())
    
XTR[:][len(XTR[0])-1] = labelencoder.fit_transform(XTR[:][len(XTR[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(XTR[0])-1])
XTR = onehotencoder.fit_transform(XTR).toarray()
dfxtr = pd.DataFrame(XTR)
 
XTR[:][len(XTR[0])-1] = labelencoder.fit_transform(XTR[:][len(XTR[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(XTR[0])-1])
XTR = onehotencoder.fit_transform(XTR).toarray()
dfxtr = pd.DataFrame(XTR)

XTR[:][len(XTR[0])-1] = labelencoder.fit_transform(XTR[:][len(XTR[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(XTR[0])-1])
XTR = onehotencoder.fit_transform(XTR).toarray()
dfxtr = pd.DataFrame(XTR)

XTR[:][len(XTR[0])-1] = labelencoder.fit_transform(XTR[:][len(XTR[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(XTR[0])-1])
XTR = onehotencoder.fit_transform(XTR).toarray()
dfxtr = pd.DataFrame(XTR)

XTR[:][len(XTR[0])-1] = labelencoder.fit_transform(XTR[:][len(XTR[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(XTR[0])-1])
XTR = onehotencoder.fit_transform(XTR).toarray()
dfxtr = pd.DataFrame(XTR)

X_M = []

for i in range(len(XIR)):
    X_M.append(XIR[i][[0,1,2,3,5]].copy())
 
for i in range(len(X_M)):
    if X_M[i][0]=='?':
        X_M[i][0]=mode[0]
    if X_M[i][1]=='?':
        X_M[i][1]=mode[1]
    if X_M[i][2]=='?':
        X_M[i][2]=mode[2]
    if X_M[i][3]=='?':
        X_M[i][3]=mode[3]
    
for i in range(len(X_M)):
    for j in range(4,5):
        if X_M[i][j]=='?':
            X_M[i][j]=mode[j+1]

dfxm=pd.DataFrame(X_M)

X_M[:][len(X_M[0])-1] = labelencoder.fit_transform(X_M[:][len(X_M[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(X_M[0])-1])
X_M = onehotencoder.fit_transform(X_M).toarray()
dfxm = pd.DataFrame(X_M)

X_M[:][len(X_M[0])-1] = labelencoder.fit_transform(X_M[:][len(X_M[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(X_M[0])-1])
X_M = onehotencoder.fit_transform(X_M).toarray()
dfxm = pd.DataFrame(X_M)

X_M[:][len(X_M[0])-1] = labelencoder.fit_transform(X_M[:][len(X_M[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(X_M[0])-1])
X_M = onehotencoder.fit_transform(X_M).toarray()
dfxm = pd.DataFrame(X_M)

X_M[:][len(X_M[0])-1] = labelencoder.fit_transform(X_M[:][len(X_M[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(X_M[0])-1])
X_M = onehotencoder.fit_transform(X_M).toarray()
dfxm = pd.DataFrame(X_M)

X_M[:][len(X_M[0])-1] = labelencoder.fit_transform(X_M[:][len(X_M[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(X_M[0])-1])
X_M = onehotencoder.fit_transform(X_M).toarray()
dfxm = pd.DataFrame(X_M)

Xa = []
Xb = []
Xc = []
Xd = []
Xe = []
Xf = []
Xg = []
Xh = []
Xi = []
Xj = []
for i in range(len(XCR)):
    if XCR[i][4]==1:
        Xa.append(XTR[i][:].copy())
    elif XCR[i][4]==2:
        Xb.append(XTR[i][:].copy())
    elif XCR[i][4]==3:
        Xc.append(XTR[i][:].copy())
    elif XCR[i][4]==4:
        Xd.append(XTR[i][:].copy())
    elif XCR[i][4]==5:
        Xe.append(XTR[i][:].copy())
    elif XCR[i][4]==6:
        Xf.append(XTR[i][:].copy())
    elif XCR[i][4]==7:
        Xg.append(XTR[i][:].copy())
    elif XCR[i][4]==8:
        Xh.append(XTR[i][:].copy())
    elif XCR[i][4]==9:
        Xi.append(XTR[i][:].copy())
    elif XCR[i][4]==9:
        Xj.append(XTR[i][:].copy())
        
dfxa = pd.DataFrame(Xa)
dfxb = pd.DataFrame(Xb)
dfxc = pd.DataFrame(Xc)
dfxd = pd.DataFrame(Xd)
dfxe = pd.DataFrame(Xe)
dfxf = pd.DataFrame(Xf)
dfxg = pd.DataFrame(Xg)
dfxh = pd.DataFrame(Xh)
dfxi = pd.DataFrame(Xi)
dfxj = pd.DataFrame(Xj)

X_true = []

for i in range(len(XD)):
    if XD[i][4]=='?':
        X_true.append(X_complete[i][4])
        
dfxtrue = pd.DataFrame(X_true)

from sklearn.model_selection import KFold 
kf= KFold(n_splits=10, random_state=42, shuffle=False)

PCP5 = []

X1_pred = []
#sigma = [0.0001,0.0011,0.0081,0.0011,0.2831]
#sigma = [0.7,0.9999,0.1291,0.8161,0.9999]
#sigma = [0.35005,0.50005,0.0686,0.4086,0.6415]
sigma = 0.5
p = 2
pi=3.14
all_pred_values = []
z = 0
for train_index, test_index in kf.split(X_M):
    X_test = []
    for i in range(test_index[0],test_index[0]+len(test_index)):
        if XIR[i][4]=='?':
            X_test.append(X_M[i][:].copy())
    dfx_test = pd.DataFrame(X_test)       

    for k in range(len(X_test)):
        sum_of_each_class = {}

        sum_of_each_class[0]=0
        sum_of_all_pattern_nodes=0
        for j in range(len(Xa)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxa.iloc[j][m])*(dfx_test.iloc[k][m]-dfxa.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes += temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xa))
        sum_of_each_class[0] = sum_of_all_pattern_nodes
        
        sum_of_each_class[1]=0
        sum_of_all_pattern_nodes=0
        for j in range(0,len(Xb)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxb.iloc[j][m])*(dfx_test.iloc[k][m]-dfxb.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes+=temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xb))
        sum_of_each_class[1] = sum_of_all_pattern_nodes
        
        sum_of_each_class[2]=0
        sum_of_all_pattern_nodes=0
        for j in range(0,len(Xc)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxc.iloc[j][m])*(dfx_test.iloc[k][m]-dfxc.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes+=temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xc))
        sum_of_each_class[2] = sum_of_all_pattern_nodes
        
        sum_of_each_class[3]=0
        sum_of_all_pattern_nodes=0
        for j in range(0,len(Xd)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxd.iloc[j][m])*(dfx_test.iloc[k][m]-dfxd.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes+=temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xd))
        sum_of_each_class[3] = sum_of_all_pattern_nodes
        
        sum_of_each_class[4]=0
        sum_of_all_pattern_nodes=0
        for j in range(0,len(Xe)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxe.iloc[j][m])*(dfx_test.iloc[k][m]-dfxe.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes+=temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xe))
        sum_of_each_class[3] = sum_of_all_pattern_nodes

        sum_of_each_class[5]=0
        sum_of_all_pattern_nodes=0
        for j in range(len(Xf)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxf.iloc[j][m])*(dfx_test.iloc[k][m]-dfxf.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes += temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xf))
        sum_of_each_class[5] = sum_of_all_pattern_nodes

        sum_of_each_class[6]=0
        sum_of_all_pattern_nodes=0
        for j in range(len(Xg)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxg.iloc[j][m])*(dfx_test.iloc[k][m]-dfxg.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes += temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xg))
        sum_of_each_class[6] = sum_of_all_pattern_nodes
        
        sum_of_each_class[7]=0
        sum_of_all_pattern_nodes=0
        for j in range(len(Xh)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxh.iloc[j][m])*(dfx_test.iloc[k][m]-dfxh.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes += temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xh))
        sum_of_each_class[7] = sum_of_all_pattern_nodes
        
        sum_of_each_class[8]=0
        sum_of_all_pattern_nodes=0
        for j in range(len(Xi)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxi.iloc[j][m])*(dfx_test.iloc[k][m]-dfxi.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes += temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xi))
        sum_of_each_class[8] = sum_of_all_pattern_nodes

        if len(Xj)==0:
            sum_of_each_class[9]=0
        else:
            sum_of_all_pattern_nodes=0
            for j in range(len(Xj)):
                temp=0
                for m in range (len(X_test[0])):
                    temp+=(dfx_test.iloc[k][m]-dfxj.iloc[j][m])*(dfx_test.iloc[k][m]-dfxj.iloc[j][m])
                temp=-temp
                temp=temp/(2*sigma*sigma)
                temp=math.exp(temp)
                sum_of_all_pattern_nodes += temp
            sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xj))
            sum_of_each_class[9] = sum_of_all_pattern_nodes
    
        max_sum_among_all_classes = max(sum_of_each_class[0],sum_of_each_class[1],sum_of_each_class[2],sum_of_each_class[3],sum_of_each_class[4],sum_of_each_class[5],sum_of_each_class[6],sum_of_each_class[7],sum_of_each_class[8],sum_of_each_class[9])
        if max_sum_among_all_classes==sum_of_each_class[0] :
            classified_class=1
        elif max_sum_among_all_classes==sum_of_each_class[1] :
            classified_class=2
        elif max_sum_among_all_classes==sum_of_each_class[2] :
            classified_class=3
        elif max_sum_among_all_classes==sum_of_each_class[3] :
            classified_class=4
        elif max_sum_among_all_classes==sum_of_each_class[4] :
            classified_class=5
        elif max_sum_among_all_classes==sum_of_each_class[5] :
            classified_class=6
        elif max_sum_among_all_classes==sum_of_each_class[6] :
            classified_class=7
        elif max_sum_among_all_classes==sum_of_each_class[7] :
            classified_class=8
        elif max_sum_among_all_classes==sum_of_each_class[8] :
            classified_class=9
        elif max_sum_among_all_classes==sum_of_each_class[9] :
            classified_class=10
        
        all_pred_values.append(classified_class)
        
    correct_prediction1=0
    incorrect_prediction1=0
    for i in range(z,z+len(X_test)):
        if all_pred_values[i]==X_true[i]:
            correct_prediction1+=1
        else:
            incorrect_prediction1+=1
    
    z+=len(X_test)
    percentage=correct_prediction1*100/(correct_prediction1+incorrect_prediction1)
    PCP5.append(percentage)

dfpcp5 = pd.DataFrame(PCP5)



#Taking attribute 6 as output
#making X_M for encoding XIR replaced with mode values
XTR = []
for i in range(len(XCR)):
    XTR.append(XCR[i][[0,1,2,3,4]].copy())
    
XTR[:][len(XTR[0])-1] = labelencoder.fit_transform(XTR[:][len(XTR[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(XTR[0])-1])
XTR = onehotencoder.fit_transform(XTR).toarray()
dfxtr = pd.DataFrame(XTR)
 
XTR[:][len(XTR[0])-1] = labelencoder.fit_transform(XTR[:][len(XTR[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(XTR[0])-1])
XTR = onehotencoder.fit_transform(XTR).toarray()
dfxtr = pd.DataFrame(XTR)

XTR[:][len(XTR[0])-1] = labelencoder.fit_transform(XTR[:][len(XTR[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(XTR[0])-1])
XTR = onehotencoder.fit_transform(XTR).toarray()
dfxtr = pd.DataFrame(XTR)

XTR[:][len(XTR[0])-1] = labelencoder.fit_transform(XTR[:][len(XTR[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(XTR[0])-1])
XTR = onehotencoder.fit_transform(XTR).toarray()
dfxtr = pd.DataFrame(XTR)

XTR[:][len(XTR[0])-1] = labelencoder.fit_transform(XTR[:][len(XTR[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(XTR[0])-1])
XTR = onehotencoder.fit_transform(XTR).toarray()
dfxtr = pd.DataFrame(XTR)

X_M = []

for i in range(len(XIR)):
    X_M.append(XIR[i][[0,1,2,3,4]].copy())
 
for i in range(len(X_M)):
    if X_M[i][0]=='?':
        X_M[i][0]=mode[0]
    if X_M[i][1]=='?':
        X_M[i][1]=mode[1]
    if X_M[i][2]=='?':
        X_M[i][2]=mode[2]
    if X_M[i][3]=='?':
        X_M[i][3]=mode[3]
    if X_M[i][4]=='?':
        X_M[i][4]=mode[4]
    

dfxm=pd.DataFrame(X_M)

X_M[:][len(X_M[0])-1] = labelencoder.fit_transform(X_M[:][len(X_M[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(X_M[0])-1])
X_M = onehotencoder.fit_transform(X_M).toarray()
dfxm = pd.DataFrame(X_M)

X_M[:][len(X_M[0])-1] = labelencoder.fit_transform(X_M[:][len(X_M[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(X_M[0])-1])
X_M = onehotencoder.fit_transform(X_M).toarray()
dfxm = pd.DataFrame(X_M)

X_M[:][len(X_M[0])-1] = labelencoder.fit_transform(X_M[:][len(X_M[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(X_M[0])-1])
X_M = onehotencoder.fit_transform(X_M).toarray()
dfxm = pd.DataFrame(X_M)

X_M[:][len(X_M[0])-1] = labelencoder.fit_transform(X_M[:][len(X_M[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(X_M[0])-1])
X_M = onehotencoder.fit_transform(X_M).toarray()
dfxm = pd.DataFrame(X_M)

X_M[:][len(X_M[0])-1] = labelencoder.fit_transform(X_M[:][len(X_M[0])-1])
onehotencoder = OneHotEncoder(categorical_features = [len(X_M[0])-1])
X_M = onehotencoder.fit_transform(X_M).toarray()
dfxm = pd.DataFrame(X_M)

Xa = []
Xb = []
Xc = []
Xd = []
Xe = []
Xf = []
Xg = []
Xh = []
Xi = []
Xj = []
Xk = []
Xl = []
Xm = []
for i in range(len(XCR)):
    if XCR[i][5]==1:
        Xa.append(XTR[i][:].copy())
    elif XCR[i][5]==2:
        Xb.append(XTR[i][:].copy())
    elif XCR[i][5]==3:
        Xc.append(XTR[i][:].copy())
    elif XCR[i][5]==4:
        Xd.append(XTR[i][:].copy())
    elif XCR[i][5]==5:
        Xe.append(XTR[i][:].copy())
    elif XCR[i][5]==6:
        Xf.append(XTR[i][:].copy())
    elif XCR[i][5]==7:
        Xg.append(XTR[i][:].copy())
    elif XCR[i][5]==8:
        Xh.append(XTR[i][:].copy())
    elif XCR[i][5]==9:
        Xi.append(XTR[i][:].copy())
    elif XCR[i][5]==10:
        Xj.append(XTR[i][:].copy())
    elif XCR[i][5]==11:
        Xj.append(XTR[i][:].copy())
    elif XCR[i][5]==12:
        Xj.append(XTR[i][:].copy())
    elif XCR[i][5]==13:
        Xj.append(XTR[i][:].copy())
        
dfxa = pd.DataFrame(Xa)
dfxb = pd.DataFrame(Xb)
dfxc = pd.DataFrame(Xc)
dfxd = pd.DataFrame(Xd)
dfxe = pd.DataFrame(Xe)
dfxf = pd.DataFrame(Xf)
dfxg = pd.DataFrame(Xg)
dfxh = pd.DataFrame(Xh)
dfxi = pd.DataFrame(Xi)
dfxj = pd.DataFrame(Xj)
dfxk = pd.DataFrame(Xk)
dfxl = pd.DataFrame(Xl)
dfxm = pd.DataFrame(Xm)

X_true = []

for i in range(len(XD)):
    if XD[i][5]=='?':
        X_true.append(X_complete[i][5])
        
dfxtrue = pd.DataFrame(X_true)

from sklearn.model_selection import KFold 
kf= KFold(n_splits=10, random_state=42, shuffle=False)

PCP6 = []

X1_pred = []
#sigma = [0.0001,0.0011,0.0081,0.0011,0.2831]
#sigma = [0.7,0.9999,0.1291,0.8161,0.9999]
#sigma = [0.35005,0.50005,0.0686,0.4086,0.6415]
sigma = 0.1
p = 2
pi=3.14
all_pred_values = []
z = 0
for train_index, test_index in kf.split(X_M):
    X_test = []
    for i in range(test_index[0],test_index[0]+len(test_index)):
        if XIR[i][5]=='?':
            X_test.append(X_M[i][:].copy())
    dfx_test = pd.DataFrame(X_test)       

    for k in range(len(X_test)):
        sum_of_each_class = {}

        sum_of_each_class[0]=0
        sum_of_all_pattern_nodes=0
        for j in range(len(Xa)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxa.iloc[j][m])*(dfx_test.iloc[k][m]-dfxa.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes += temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xa))
        sum_of_each_class[0] = sum_of_all_pattern_nodes
        
        sum_of_each_class[1]=0
        sum_of_all_pattern_nodes=0
        for j in range(0,len(Xb)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxb.iloc[j][m])*(dfx_test.iloc[k][m]-dfxb.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes+=temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xb))
        sum_of_each_class[1] = sum_of_all_pattern_nodes
        
        sum_of_each_class[2]=0
        sum_of_all_pattern_nodes=0
        for j in range(0,len(Xc)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxc.iloc[j][m])*(dfx_test.iloc[k][m]-dfxc.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes+=temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xc))
        sum_of_each_class[2] = sum_of_all_pattern_nodes
        
        sum_of_each_class[3]=0
        sum_of_all_pattern_nodes=0
        for j in range(0,len(Xd)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxd.iloc[j][m])*(dfx_test.iloc[k][m]-dfxd.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes+=temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xd))
        sum_of_each_class[3] = sum_of_all_pattern_nodes
        
        sum_of_each_class[4]=0
        sum_of_all_pattern_nodes=0
        for j in range(0,len(Xe)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxe.iloc[j][m])*(dfx_test.iloc[k][m]-dfxe.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes+=temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xe))
        sum_of_each_class[3] = sum_of_all_pattern_nodes

        sum_of_each_class[5]=0
        sum_of_all_pattern_nodes=0
        for j in range(len(Xf)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxf.iloc[j][m])*(dfx_test.iloc[k][m]-dfxf.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes += temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xf))
        sum_of_each_class[5] = sum_of_all_pattern_nodes

        sum_of_each_class[6]=0
        sum_of_all_pattern_nodes=0
        for j in range(len(Xg)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxg.iloc[j][m])*(dfx_test.iloc[k][m]-dfxg.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes += temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xg))
        sum_of_each_class[6] = sum_of_all_pattern_nodes
        
        sum_of_each_class[7]=0
        sum_of_all_pattern_nodes=0
        for j in range(len(Xh)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxh.iloc[j][m])*(dfx_test.iloc[k][m]-dfxh.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes += temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xh))
        sum_of_each_class[7] = sum_of_all_pattern_nodes
        
        sum_of_each_class[8]=0
        sum_of_all_pattern_nodes=0
        for j in range(len(Xi)):
            temp=0
            for m in range (len(X_test[0])):
                temp+=(dfx_test.iloc[k][m]-dfxi.iloc[j][m])*(dfx_test.iloc[k][m]-dfxi.iloc[j][m])
            temp=-temp
            temp=temp/(2*sigma*sigma)
            temp=math.exp(temp)
            sum_of_all_pattern_nodes += temp
        sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xi))
        sum_of_each_class[8] = sum_of_all_pattern_nodes

        if len(Xj)==0:
            sum_of_each_class[9]=0
        else:
            sum_of_all_pattern_nodes=0
            for j in range(len(Xj)):
                temp=0
                for m in range (len(X_test[0])):
                    temp+=(dfx_test.iloc[k][m]-dfxj.iloc[j][m])*(dfx_test.iloc[k][m]-dfxj.iloc[j][m])
                temp=-temp
                temp=temp/(2*sigma*sigma)
                temp=math.exp(temp)
                sum_of_all_pattern_nodes += temp
            sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xj))
            sum_of_each_class[9] = sum_of_all_pattern_nodes

        if len(Xk)==0:
            sum_of_each_class[10]=0
        else:
            sum_of_all_pattern_nodes=0
            for j in range(len(Xk)):
                temp=0
                for m in range (len(X_test[0])):
                    temp+=(dfx_test.iloc[k][m]-dfxk.iloc[j][m])*(dfx_test.iloc[k][m]-dfxk.iloc[j][m])
                temp=-temp
                temp=temp/(2*sigma*sigma)
                temp=math.exp(temp)
                sum_of_all_pattern_nodes += temp
            sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xk))
            sum_of_each_class[10] = sum_of_all_pattern_nodes

        if len(Xl)==0:
            sum_of_each_class[11]=0
        else:
            sum_of_all_pattern_nodes=0
            for j in range(len(Xl)):
                temp=0
                for m in range (len(X_test[0])):
                    temp+=(dfx_test.iloc[k][m]-dfxl.iloc[j][m])*(dfx_test.iloc[k][m]-dfxl.iloc[j][m])
                temp=-temp
                temp=temp/(2*sigma*sigma)
                temp=math.exp(temp)
                sum_of_all_pattern_nodes += temp
            sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xl))
            sum_of_each_class[11] = sum_of_all_pattern_nodes

        if len(Xm)==0:
            sum_of_each_class[12]=0
        else:
            sum_of_all_pattern_nodes=0
            for j in range(len(Xm)):
                temp=0
                for m in range (len(X_test[0])):
                    temp+=(dfx_test.iloc[k][m]-dfxm.iloc[j][m])*(dfx_test.iloc[k][m]-dfxm.iloc[j][m])
                temp=-temp
                temp=temp/(2*sigma*sigma)
                temp=math.exp(temp)
                sum_of_all_pattern_nodes += temp
            sum_of_all_pattern_nodes=sum_of_all_pattern_nodes/(pow(2*pi, p/2)*pow(sigma,p)*len(Xm))
            sum_of_each_class[12] = sum_of_all_pattern_nodes
    
        max_sum_among_all_classes = max(sum_of_each_class[0],sum_of_each_class[1],sum_of_each_class[2],sum_of_each_class[3],sum_of_each_class[4],sum_of_each_class[5],sum_of_each_class[6],sum_of_each_class[7],sum_of_each_class[8],sum_of_each_class[9],sum_of_each_class[10]),sum_of_each_class[11],sum_of_each_class[12]
        if max_sum_among_all_classes==sum_of_each_class[0] :
            classified_class=1
        elif max_sum_among_all_classes==sum_of_each_class[1] :
            classified_class=2
        elif max_sum_among_all_classes==sum_of_each_class[2] :
            classified_class=3
        elif max_sum_among_all_classes==sum_of_each_class[3] :
            classified_class=4
        elif max_sum_among_all_classes==sum_of_each_class[4] :
            classified_class=5
        elif max_sum_among_all_classes==sum_of_each_class[5] :
            classified_class=6
        elif max_sum_among_all_classes==sum_of_each_class[6] :
            classified_class=7
        elif max_sum_among_all_classes==sum_of_each_class[7] :
            classified_class=8
        elif max_sum_among_all_classes==sum_of_each_class[8] :
            classified_class=9
        elif max_sum_among_all_classes==sum_of_each_class[9] :
            classified_class=10
        elif max_sum_among_all_classes==sum_of_each_class[10] :
            classified_class=11
        elif max_sum_among_all_classes==sum_of_each_class[11] :
            classified_class=12
        elif max_sum_among_all_classes==sum_of_each_class[12] :
            classified_class=13
        
        all_pred_values.append(classified_class)
        
    correct_prediction1=0
    incorrect_prediction1=0
    for i in range(z,z+len(X_test)):
        if all_pred_values[i]==X_true[i]:
            correct_prediction1+=1
        else:
            incorrect_prediction1+=1
    
    z+=len(X_test)
    percentage=correct_prediction1*100/(correct_prediction1+incorrect_prediction1)
    PCP6.append(percentage)

dfpcp6 = pd.DataFrame(PCP6)