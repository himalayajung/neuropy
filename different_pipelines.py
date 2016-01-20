# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:10:49 2015

@author: gajendrakatuwal
"""

from sklearn import svm 
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from scipy.stats import randint as sp_randint
#from scipy.stats import randint
import numpy as np
import pandas as pd
import pickle
import os
from time import time
from sklearn.pipeline import Pipeline
import math

np.random.seed(0)
#%%
pheno = pd.read_csv(os.path.join('..','..','data','ABIDE_IQ.csv'),index_col=0)
pheno = pheno[['control','age','sex','site','VIQ','PIQ','FIQ','ADOS']]
#print(pheno.describe())
#mode_pars=pickle.load( open( "mode_parameters.p", "rb" ) )

#vertices=pickle.load( open( "vertices_horizontal.p", "rb" ) )
#vertices=pickle.load( open( "vertices.p", "rb" ) )

zernike = pickle.load( open( "zernike_moments_native.p", "rb" ) )



sub_cortical_structures = ["BrStem","L_Accu","R_Accu","L_Amyg","R_Amyg","L_Caud",
                         "R_Caud","L_Hipp","R_Hipp","L_Pall","R_Pall","L_Puta",
                         "R_Puta","L_Thal","R_Thal"]

#sub_cortical_structures=["L_Hipp","R_Hipp"]
#%%
## Feature Extraction
feature_linearSVC = svm.LinearSVC(penalty="l1",dual=False)
feature_RFECV = RFECV(feature_linearSVC,step=0.05,cv=10)
#feature_PCA=PCA(n_components=n_components)
#%%
svc = svm.SVC()
param_grid = dict(C=range(1,10,2),gamma=np.logspace(-6, 1, 10))

sites = pheno['site'].unique()
list_dfs = list()
for site in sites: 
    print(site)
    list_scores = list()
    for sub_cortical in sub_cortical_structures:
        print('----------------------------')
        print(sub_cortical)
        #    d=mode_pars[sub_cortical]
    #    d=vertices[sub_cortical]
        d = zernike[sub_cortical]
        df = pheno.join(d)
    #    nan_inds= pd.isnull(X).any(1).nonzero()[0]
#        X=d.iloc[:,:10]
    #    X=d.iloc[200:300,:10]
#        X=d[df['site'] == site].iloc[:,:10]
        X = d[df['site'] == site]
        X = X[pd.notnull(X).any(1)]
        y = df['control'].loc[X.index]
      
        #%% RF
        mtry = np.sqrt(X.shape[1]).round()
    #    mtry=np.sqrt(n_components).round()
        rf = RandomForestClassifier(n_estimators=5000)
        gbm = GradientBoostingClassifier(n_estimators=10000,learning_rate=0.001)
        # Parameter Grids
        param_grid_rf = dict(max_features=np.arange(int(mtry-round(mtry/2)),int(mtry+round(mtry/2)), 2 ) )
        param_grid_gbm = dict(max_depth= range(1,10))
    #    param_grid=dict(max_features=range(5,100,5))
        param_dist = {"max_features": sp_randint(5,100)}
        random_search_rf = RandomizedSearchCV(rf,param_distributions=param_dist,n_iter=40)
        grid_search_rf = GridSearchCV(estimator = rf, param_grid = param_grid_rf, cv = 10) 
        grid_search_gbm = GridSearchCV(estimator = gbm, param_grid =param_grid_gbm, cv = 10) 

        pipe1 = Pipeline([('feature_selection', feature_linearSVC),
                         ('classification', grid_search_rf)])

        pipe2 = Pipeline([('feature_selection', feature_RFECV),
                         ('classification', random_search_rf)])
                         
    #    pipe3 = Pipeline([('feature_selection', feature_PCA),
    #                      ('classification', grid_search_rf)])
        #%%
        #Nested cross-validation
        t0 = time()
    #    result=cross_val_score(pipe1, X, y,cv=10,verbose=0,n_jobs=-1)
        result=cross_val_score(grid_search_rf, X, y,cv=10,verbose=0,n_jobs=-1)
        # result=cross_val_score(grid_search_gbm, X, y,cv=10,verbose=0,n_jobs=-1)
        list_scores.append(result)
        print(result)
        print(result.mean())
        print("done in %0.3fs" % (time()-t0))
    df_scores=pd.DataFrame(list_scores)   
    df_scores.index=sub_cortical_structures
    list_dfs.append(df_scores) 
    
df_scores_site = pd.concat(list_dfs,keys=sites,axis=0) # cbind    
pickle.dump(df_scores_site, open( "saved_runs/zernike_native_rf_accuracy_sitewise.p", "wb" ) )
