# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:39:43 2016

@author: ARSHABH SEMWAL
"""
import numpy as np
import pandas as pd
import quandl , math
from sklearn import cross_validation , preprocessing , svm
from sklearn.linear_model import LinearRegression


df=quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PTC']=(df['Adj. High'] -df['Adj. Close'])/(df['Adj. Close'])*100.0
df['PTC_change']=(df['Adj. Close'] -df['Adj. Open'])/(df['Adj. Open'])*100.0
df=df[['Adj. Close','HL_PTC','PTC_change','Adj. Volume']]

forecast_col='Adj. Close'
df.fillna(-99999, inplace=True)
forecast_out = int(math.ceil(0.01*len(df)))
print(len(df))
df['label']=df[forecast_col].shift(-forecast_out)

X=np.array(df.drop(['label'],1))
X=preprocessing.scale(X)
X=X[:-forecast_out]
X_lately=X[-forecast_out:]

df.dropna(inplace=True)

y=np.array(df['label'])
y=np.array(df['label'])


#print(len(X), len(y))
X_train , X_test ,y_train , y_test = cross_validation.train_test_split(X,y,test_size=0.2)
classifier=LinearRegression(n_jobs=-1)#can try n_jobs inside paranthesis
#clf=svm.SVR(kernel='poly') # support vector machine
#clf.fit(X_train , y_train)
#accuracy1=clf.score(X_test , y_test)
classifier.fit(X_train , y_train)
accuracy = classifier.score(X_test , y_test)
# fit is synonymous with test data and score is synonymous wiht train data model
print(accuracy)
forecast_set = classifier.predict(X_lately)
print(forecast_set ,forecast_out)

