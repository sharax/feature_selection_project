# Default imports


import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
import numpy as np

# Write your solution here:
finallist=[]
def percentile_k_features(df,k=20):
    features=df.iloc[:,:-1]
   # print features.head()
    headerlist=list(features)
    #print headerlist
    target=df.loc[:,'SalePrice']
    #print target.head()

    model=SelectPercentile(f_regression,k)
    x_new=model.fit_transform(features,target)
    #print x_new
    scores = -np.log10(model.pvalues_)
    #print scores
    indexes=model.get_support(indices=True)
    #print indexes
    scoreslist=[]
    for i in indexes:
        scoreslist.append(scores[i])

    #print scoreslist
    zipped=zip(scoreslist,indexes)
    zipped.sort(reverse=True)
    indexes_unzipped=[index for (s,index) in zipped]
    #print indexes_unzipped
    scores_unzipped=[scores for (scores,i) in zipped]
    #print scores_unzipped

    for i in indexes_unzipped:
        finallist.append(headerlist[i])
    #print finallist
    return finallist
