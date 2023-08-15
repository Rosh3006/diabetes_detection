import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("diabetes.csv")
pd.set_option('display.max_columns',None)
print(df.head(10))
print(df.describe())
print(df.isnull().sum())


X=df.drop(columns=['Outcome'])
Y=df["Outcome"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.30)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(x_train)
x_train= sc.transform(x_train)
x_test= sc.transform(x_test)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
print("Accuracy of Linear Regression:",lr.score(x_test,y_test)*100)

from sklearn.linear_model import LogisticRegression
lor=LogisticRegression()
lor.fit(x_train,y_train)
print("Accuracy of Logistic Regression:",lor.score(x_test,y_test)*100)

from sklearn.neighbors import KNeighborsClassifier
knc=KNeighborsClassifier()
knc.fit(x_train,y_train)
print("Accuracy of KNC:",knc.score(x_test,y_test)*100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
print("Accuracy of Decision Tree:",dt.score(x_test,y_test)*100)

from sklearn import svm
classifer=svm.SVC(kernel='linear')
classifer.fit(x_train,y_train)
print("Accuracy of SVC:",classifer.score(x_test,y_test)*100)

from sklearn import svm
regression=svm.SVR(kernel='linear')
regression.fit(x_train,y_train)
print("Accuracy of SVR:",regression.score(x_test,y_test)*100)

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=100,random_state=0)
regressor.fit(x_train,y_train)
print("Accuracy of Random Forest Regressor:",regressor.score(x_test,y_test)*100)

from sklearn.ensemble import RandomForestClassifier
RandomForest = RandomForestClassifier(oob_score=True)
RandomForest.fit(x_train,y_train)
print(RandomForest.oob_score_)
print("Accuracy of Random Forest Classifier:",RandomForest.score(x_test,y_test)*100)

from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)
print("Accuracy of Naive Bayes Classifier:",classifier.score(x_test,y_test)*100)

import pickle
filename="saveModelDiabetes.sav"
pickle.dump(classifer,open(filename,'wb'))
load_model=pickle.load(open(filename,'rb'))
y=load_model.predict([[5,168,64,0,0,32.9,0.135,41]])
print(y)
