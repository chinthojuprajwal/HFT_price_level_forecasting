# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 13:51:27 2021

@author: prajwal
"""
#importing packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
import scipy.stats as stats
import pylab
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
import timeit
from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
import random
#%% Reading CSV
data = pd.read_csv('D:/UIUC_courses/IE HFT/stock_prediction_svr/data/book.csv')

#%% Convert time to nanoseconds and set up dat vector with price level,ask and prices and sizes and time
rnd=int(random.random()*100000)
df2=data[rnd:rnd+10000]
df1=df2['COLLECTION_TIME'].values
deltaTime=0
sumi=0
timest=[0]
price=[]
cl=[]
consttime=10000
sum1=0
k=0
rest=[]
rest.append(list(df2.iloc[0])[4:])

kprev=1
price.append((df2['BID_PRICE_1'].values[k]*df2['BID_SIZE_1'].values[k]+df2['BID_PRICE_2'].values[k]*df2['BID_SIZE_2'].values[k]
                 +df2['BID_PRICE_3'].values[k]*df2['BID_SIZE_3'].values[k]+df2['ASK_PRICE_3'].values[k]*df2['ASK_SIZE_3'].values[k]
                 +df2['ASK_PRICE_2'].values[k]*df2['ASK_SIZE_2'].values[k]+df2['ASK_PRICE_1'].values[k]*df2['ASK_SIZE_1'].values[k])/
                (df2['ASK_SIZE_1'].values[k]+df2['ASK_SIZE_2'].values[k]+df2['ASK_SIZE_3'].values[k]+df2['BID_SIZE_1'].values[k]+
                 df2['BID_SIZE_2'].values[k]
                 +df2['BID_SIZE_3'].values[k]))
for k in range(1,int(len(df2))):

        price.append((df2['BID_PRICE_1'].values[k]*df2['BID_SIZE_1'].values[k]+df2['BID_PRICE_2'].values[k]*df2['BID_SIZE_2'].values[k]
                      +df2['BID_PRICE_3'].values[k]*df2['BID_SIZE_3'].values[k]+df2['ASK_PRICE_3'].values[k]*df2['ASK_SIZE_3'].values[k]
                      +df2['ASK_PRICE_2'].values[k]*df2['ASK_SIZE_2'].values[k]+df2['ASK_PRICE_1'].values[k]*df2['ASK_SIZE_1'].values[k])/
                     (df2['ASK_SIZE_1'].values[k]+df2['ASK_SIZE_2'].values[k]+df2['ASK_SIZE_3'].values[k]+df2['BID_SIZE_1'].values[k]+
                      df2['BID_SIZE_2'].values[k]
                      +df2['BID_SIZE_3'].values[k]))
        rest = np.append(rest,[list(df2.iloc[k])[4:]],axis= 0)
        
        head1,sep1,tail1=df1[k-1].partition('.')
        head2,sep2,tail2=df1[k].partition('.')
        firstStamp = pd.to_datetime(head1, format='%Y-%m-%dT%H:%M:%S')
        lastStamp = pd.to_datetime(head2, format='%Y-%m-%dT%H:%M:%S')
        deltaTime = deltaTime+(lastStamp - firstStamp).total_seconds()*1e9+(int(tail2)-int(tail1))
        timest.append(deltaTime)

        sumi=sumi+1
        if (price[-1]-price[-2])/price[-2]==0:
            price=price[:-1]
            timest=timest[:-1]
            rest=rest[:-1]

cl=np.transpose(np.vstack([price,np.transpose(rest),timest]))


#%% Setting up features and target variables
def processData(data,lb):
    X,Y = [],[]
    for i in range(len(data)-lb-1):
        X.append(np.append(data[i:i+lb,:].reshape(-1),data[i+lb,13].reshape(-1),axis=0))
        Y.append(((data[(i+lb+1),0]-data[(i+lb),0])/data[(i+lb),0])*1e5)
        first_val=X[i][(0)*14+13]
        for k in range(lb):
            X[i][(k)*14+13]=X[i][(k)*14+13]-first_val
        X[i][-1]=X[i][-1]-first_val
    return np.array(X),np.array(Y)
sz=20
X,y = processData(cl,sz)


#%%Setting up model and training
score=[]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
model = Pipeline([('scaler', StandardScaler()), ('model', RandomForestRegressor(max_depth=3,n_estimators=150))])
param_grid={'model__max_depth': [3]}
gscv = GridSearchCV(estimator=model, param_grid=param_grid,cv=5)
gscv.fit(X_train,y_train)
print(gscv.best_params_)
#%% predicting training data and accuracy reporting
y_predict=0
y_predict=gscv.best_estimator_.predict(X_train)
plt.plot((y_train),'r',label='actual')
plt.plot((y_predict),'g',label='predicted')
plt.title('Actual vs predicted price change ratio - RF- train')
plt.ylim(-7, 7)
plt.legend()
c=0
w=0
for i in range(0,len(y_train)):
    if y_predict[i]*y_train[i]>0:
        c=c+1
    if y_predict[i]*y_train[i]<0:
        w=w+1
        
print('correct percentage=',str(c/len(y_train)))
print('wrong percentage=',str(w/len(y_train)))


#%% predicting test and accuracy reporting

y_predict=gscv.best_estimator_.predict(X_test)
plt.plot((y_test),'r',label='actual')
plt.plot((y_predict),'g',label='predicted')
plt.title('Actual vs predicted price change ratio - RF-test')
plt.ylim(-7, 7)
plt.legend()
c=0
w=0
for i in range(0,len(y_test)):
    if y_predict[i]*y_test[i]>0:
        c=c+1
    if y_predict[i]*y_test[i]<0:
        w=w+1
        
print('correct percentage=',str(c/len(y_test)))
print('wrong percentage=',str(w/len(y_test)))

