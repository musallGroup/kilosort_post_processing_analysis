# -*- coding: utf-8 -*-
"""
@author: Jain 
"""
#%%
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from mlxtend.plotting import plot_decision_regions
from preprocessing import create_preprocessing_pipeline, preprocess_frame, get_groundTruth
from BayesianSearch import run_search
#%%



#%% Call Data
metric_data = dataset[['syncSpike_2',
                              'firing_rate',
                              'presence_ratio',
                              'nn_hit_rate',
                              'nn_miss_rate',
                              'cumulative_drift']].values
X = metric_data
y = dataset['label'].values

#%% Tceate pipeline using  unsupervised feature preprocessing 

preprocessing_pipeline = create_preprocessing_pipeline()

#plot_umap_embedding(X_train_final, y_train, 'Embedded via UMAP using Labels')
#plot_umap_embedding(X_test_final, y_test, 'Embedded via UMAP predicting Labels')

run_search(preprocessing_pipeline, X, y) # call function 
train_best_estimator(preprocessing_pipeline,X,y)

