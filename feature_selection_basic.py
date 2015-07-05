# -*- coding: utf-8 -*-
"""
Created on Fri May 22 10:37:52 2015

@author: gajendrakatuwal
"""
from sklearn.svm import SVC
from sklearn.linear_model import RandomizedLasso
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
import numpy as np
import pandas as pd
import pickle
import os
import re
from time import time

#%% Settings
np.random.seed(0)

#%% Helper Functions
def remove_highly_correlated(df,method='pearson',threshold=0.98):
    df_corr = df.corr(method=method)
    # create a mask to ignore self-
    mask = np.ones(df_corr.columns.size) - np.eye(df_corr.columns.size)
    df_corr = mask * df_corr
    
    drops = []
    # loop through each variable
    for col in df_corr.columns.values:
        # if we've already determined to drop the current variable, continue
        if np.in1d([col],drops):
           continue
        # find all the variables that are highly correlated with the current variable 
        # and add them to the drop list 
        corr = df_corr[abs(df_corr[col]) > threshold].index
        drops = np.union1d(drops, corr)
 
    print "\nDropping", drops.shape[0], "highly correlated features...\n", drops
    df.drop(drops, axis=1, inplace=True)

#%% Data
data = pd.read_csv(os.path.join('..','..','data','FS_imputed.csv'),index_col=0)
data = data[data['sex']==1]

y = data['control']
data = data.iloc[:,10:]
intensity_columns = filter(lambda x:"_intensity" in x,data.iloc[:,10:].columns)
data.drop(intensity_columns,axis=1,inplace=True) # remove intensity columns

attributes = ['_volume','_area','_thickness$','_thicknessstd','_foldind','_meancurv','_gauscurv','_all']
# attributes = ['_volume','_area']
skf = StratifiedKFold(y, n_folds=10,shuffle=True,random_state=np.random.seed(0))

t0=time() 
dict_for_attribute={}

for attribute in attributes:
    print(attribute)
    print('------------------------------------------')
    ## selecting morphometric feature specifice columns
    if attribute is "_all":
        continue
    else:
        attribute_columns=filter(lambda x:re.search(attribute,x), data.iloc[:,10:].columns)
        X=data[attribute_columns[:20]]
        
    remove_highly_correlated(X,threshold=0.98)
    print(X.columns.values)
    list_dicts=list()
    for train_index, test_index in skf:
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(X_train.shape)
        if feature_selection == "randomized_lasso":
            feature_selector=RandomizedLasso(sample_fraction=0.5,n_resampling=50,verbose=False,n_jobs=-1)
        elif feature_selection == "RFECV_linearSVM":
#            print(feature_selection % "selected")
            feature_selector = RFECV(SVC(kernel="linear"),step=1,cv=StratifiedKFold(y,5),scoring="accuracy")
        else:
            print("Options are: randomized_lasso, RFECV_linearSVM")
            
        feature_selector.fit(X_train,y_train)
        result = {'X_train':X_train,'y_train':y_train,'X_test':X_test,'y_test':y_test,'feature_selector':feature_selector}
        list_dicts.append(result)
        
        
    dict_for_attribute[attribute] = list_dicts
    print("done in %0.3fs" % (time()-t0))

