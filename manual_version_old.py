# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 16:31:06 2021

@author: kampaimaging
"""
import pyupset as pyu
from pickle import load
import glob
import os.path
import os
import pickle
from matplotlib.ticker import StrMethodFormatter
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec

from scipy.optimize import curve_fit

import numpy as np

import pandas as pd

import sklearn # for the roc curv
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    adjusted_rand_score,
    adjusted_mutual_info_score
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import sklearn.cluster as cluster


import scikitplot as skplt
import skopt
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objs as go
import plotly.io as pio

import seaborn as sns
import umap
import hdbscan

from mlxtend.plotting import plot_decision_regions


from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import umap
#%%
dataset_new = pd.read_csv(r'F:\Neuropixel_TestData\pipeline_analysis\NeuropixelsAnalysis\dataset\dataset.csv')
print(dataset_new.columns)
print(dataset_new.shape)
#%%
#============================Classifier========================================
idxs = list(dataset_new.index.values)
metric_data = dataset_new[[    'n_spike', 
                               'syncSpike_2',
                                'syncSpike_4',
                              'firing_rate',
                              'presence_ratio',
                              'nn_hit_rate',
                              'nn_miss_rate',
                              'cumulative_drift']].values

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(metric_data,
                                                                                 dataset_new['gTruth'].values,
                                                                                 idxs, 
                                                                                 shuffle=True,
                                                                                 random_state=42)


#%% Tceate pipeline using  unsupervised feature preprocessing 

pipe = Pipeline([('impute', SimpleImputer(missing_values=np.nan, strategy='median')),
                 ('scaler', StandardScaler())])

#call the tranformer in data 
X_train_transform = pipe.fit_transform(X_train)

#%% Supervised UMAP 

supervised_embedder = umap.UMAP(
    min_dist=0.0, 
    n_neighbors=10, 
    n_components=2, # dimensions 
    random_state=42
    )
X_train_final = supervised_embedder.fit_transform(X_train_transform, y=y_train)
#X_train_final : fit and transformed with emedder
# fig, ax = plt.subplots()
# #fig.set_size_inches(18.5, 10.5)

# plt.scatter(X_train_final[:, 0], X_train_final[:, 1], s=20, c=y_train, cmap='Spectral', alpha=1.0)
# plt.setp(ax, xticks=[], yticks=[])

# cbar = plt.colorbar(boundaries=np.arange(3)-0.5)
# cbar.set_ticks(np.arange(2))

# plt.title(' Embedded via UMAP using Labels');

# plt.show()

#%% Metric Learning 
X_test_trans = pipe.transform(X_test) #to impute and normalise
X_test_final = supervised_embedder.transform(X_test_trans)

#%%
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neural_network import MLPClassifier 

from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
import lightgbm 
from lightgbm import LGBMRegressor, LGBMClassifier, Booster
from skopt import BayesSearchCV
seed=1
# CASH (Combined Algorithm Selection and Hyperparameter optimisation)
models = [
            'ADB',
            'GBC',
            'RFC',
            'KNC',
            'SVC',
            'MLP',
            'ExtraTreesClassifier',
            'xgboost',
           'lightgbm'
         ]
clfs = [
        AdaBoostClassifier(random_state=seed),
        GradientBoostingClassifier(random_state=seed),
        RandomForestClassifier(random_state=seed,n_jobs=-1),
        KNeighborsClassifier(n_jobs=-1),
        SVC(random_state=seed,probability=True),
        MLPClassifier(random_state=seed, max_iter=300,hidden_layer_sizes= (50, 100)),
        ExtraTreesClassifier(n_estimators=100, random_state=0),
        XGBClassifier(n_estimators=100, random_state=0),
        LGBMClassifier(random_state=0)
        ]
params = {
            models[0]:{'learning_rate':[1,2], 
                       'n_estimators':[50,100],
                       'algorithm':['SAMME','SAMME.R']
                       },#AdaB
    
            models[1]:{'learning_rate':[0.05,0.1],
                       'n_estimators':[100,150], 
                       'max_depth':[2,4],
                       'min_samples_split':[2,4],
                       'min_samples_leaf': [2,4]
                       }, #GBC
    
            models[2]:{'n_estimators':[100,150],
                       'criterion':['gini','entropy'],
                       'min_samples_split':[2,4],
                       'min_samples_leaf': [2,4]
                       }, #RFC
    
            models[3]:{'n_neighbors':[20,50], 
                       'weights':['distance','uniform'],
                       'leaf_size':[30]
                       }, #KNN
    
            models[4]: {'C':[0.5,2.5],
                       'kernel':['sigmoid','linear','poly','rbf']
                       }, #SVC
            
            models[5]: {
                         'activation': ['tanh', 'relu'],
                         'solver': ['sgd', 'adam'],
                         'alpha': [0.0001, 0.05],
                         'learning_rate': ['constant','adaptive']
                         }, #MLP
    
            models[6]:{'criterion':['gini', 'entropy'],  
                       'class_weight':['balanced', 'balanced_subsample']
                       }, #extratrees
    
             models[7]:{'max_depth':[2,4], 
                       'eta': [0.2,0.5], 
                       'sampling_method':['uniform','gradient_based'],
                       'grow_policy':['depthwise', 'lossguide']
                      }, #xgboost
                        
    
            models[8]:{'learning_rate':[0.05,0.15],
                       'n_estimators': [100,150]} #lightgbm
    
         }
test_scores = []
val_scores =[]
search_objects=[]
model_estimator = []  

for name, estimator in zip(models,clfs):
    print(name)
    
   
    clf = BayesSearchCV(estimator, params[name], scoring='accuracy', refit='True', n_jobs=-1, n_iter=20,cv=5)
    clf.fit(X_train_final, y_train)   # X_train_final, y_train X is train samples and y is the corresponding labels
   
    
    print("best params: " + str(clf.best_params_))
    print("best scores: " + str(clf.best_score_))
    print("best estimator " +  str(clf.best_estimator_))
    model_estimator.append(clf.best_estimator_)
    
    fig = plot_decision_regions(X=X_train_final, y=y_train, clf=clf, legend=2)
    plt.title(f'Decison boundary of {name} on clusters');
    plt.show()
    
    val_scores.append(clf.best_score_)
    #acc = accuracy_score(y_test, clf.predict(X_test_trans))
    #print("Accuracy: {:.4%}".format(acc))
    search_objects.append(clf)
    
    
    clfscore=clf.score(X_test_final, y_test)
    test_scores.append(clfscore)
#%%
max_value = max(val_scores)#Return the max value of the list
max_index = val_scores.index(max_value)
best_classifier = model_estimator[max_index]    
print("best_classifier is ", best_classifier)

#%%
clf = KNeighborsClassifier(n_jobs=-1, n_neighbors=44, weights='distance')
 
clf.fit(X_train_final,y_train)

score = clf.score(X_test_final, y_test) # report 
fig = plot_decision_regions(X=X_train_final, y=y_train, clf=clf, legend=2)
plt.title('Decison boundary')
plt.show()
print(score)

#%%

X_unseen = frames[7][[          'n_spike', 
                               'syncSpike_2',
                               'syncSpike_4',
                              'firing_rate',
                              'presence_ratio',
                              'nn_hit_rate',
                              'nn_miss_rate',
                              'cumulative_drift' ]].values


X_test_trans = pipe.transform(X_unseen) #to impute and normalise
X_test_final = supervised_embedder.transform(X_test_trans)
y_pred = clf.predict(X_test_final) #for every row 
frames[7]['is_noise']=y_pred

#%%

frames[2].to_csv(r'Y:\invivo_ephys\Neuropixels\RD10_2130_20210120\RD10_2130_20210120_g0\RD10_2130_20210120_g0_imec0\RD10_2130_20210120_g0_t0_imec0\imec0_ks2\classifier_output.csv')

#%%