# %load q03_rf_rfe/build.py
# Default imports
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# Your solution code here

def rf_rfe(data):
    X=data.iloc[:,:-1]
    y=data.iloc[:,-1]
    model=RandomForestClassifier()
    rfe=RFE(model)
    rfe=rfe.fit(X,y)
    c=rfe.ranking_
    d=[]
    e=[]
    for i in range(0,len(c)):
        if c[i]==1:
            d.append(i)
    for j in d:
        e.append(list(X)[j])
    return e

#model=RandomForest
