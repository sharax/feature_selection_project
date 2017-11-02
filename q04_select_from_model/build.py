# %load q04_select_from_model/build.py
# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')

np.random.seed(9)
# Yourr solution code here
def select_from_model(data):
    X=data.iloc[:,:-1]
    y=data.iloc[:,-1]
    model=RandomForestClassifier()
    c=model.fit(X,y)
    model1=SelectFromModel(c ,prefit=True)#embedded
    d=model1.get_support()
    e=[]
    for i in range(0,len(d)):
        if d[i]==True:
            e.append(i)
    f=[]
    for i in e:
        f.append(list(X)[i])
    return f

#select_from_model(data)
