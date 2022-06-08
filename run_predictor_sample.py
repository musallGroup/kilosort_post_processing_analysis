# -*- coding: utf-8 -*-

"""
@author: Jain 
"""
#cell 0 
import os
import pickle
from BayesianSearch import clfs, identify_best_estimator
from preprocessing import create_preprocessing_pipeline
import umap
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import json
import inspect
import numpy as np
import pandas as pd
from funcs_to_use import (
    calculate_confusion_matrix,
    get_preprocessing_umap_pipeline,
    split_data_to_X_y
)
#%% load path cell1
    
def run_predictor(test_dataframe, exp_dir):
# This applies the decoder to a given data table. Needs to include the metrics
# that the classifier was trained on.

    # cPath = os.path.dirname(inspect.getfile(run_predictor)); # find directory of this function and save pickle files there
    clf = pickle.load(open(exp_dir + 'classifier.pkl', 'rb'))
    used_metrics  = pickle.load(open(os.path.join(exp_dir, 'strict_metrics.pkl'), 'rb'))

    # check if all metrics are present
    foundMetrics = set(used_metrics) & set(list(test_dataframe.columns))
    if len(foundMetrics) != len(used_metrics):
        print('Missing metrics in dataset:')
        print(set(test_dataframe.columns).symmetric_difference(set(used_metrics)))
        raise ValueError('!! Columns in test dataset do not contain all required metrics !!') 

    X_unseen = test_dataframe[used_metrics]    
    y_pred = clf.predict(X_unseen) #for every row \
    y_prob  = clf.predict_proba(X_unseen)

    test_dataframe['is_noise']=y_pred
    test_dataframe['noise_probs']=y_prob[:1]

    # fig = plot_decision_regions(X=X_test_final, y=y_pred, clf=clf, legend=2)
    # # plt.savefig('decision-boundary.png')
    # plt.show()
    
    return y_pred, y_prob


def train_best_estimator(X, y, classifierPath, seed):
# this uses the classifier that has been previously identified as performing
# best and fits it to a new supervised UMAP projection based on X and y.

    # make sure not to use cluster_id as a metric
    if len(set(X.columns).intersection(set(['cluster_id']))) > 0:
        del X['cluster_id']
        
    used_metrics = list(X.columns)

    preprocess_umap_pipeline = get_preprocessing_umap_pipeline(seed)
    X_train_final = preprocess_umap_pipeline.fit_transform(X, y)
    
    # cPath = os.path.dirname(inspect.getfile(train_best_estimator)); # find directory of this function and save pickle files there
    config = json.load(open(classifierPath + '\incumbent_config.json'))
    clf = clfs[config['estimator']]
    params = config['params']
    clf.set_params(**params)
    clf.fit(X_train_final,y)

    pickle.dump(preprocess_umap_pipeline, open(classifierPath + '\preprocess_umap_pipeline.sav','wb'))
    pickle.dump(used_metrics, open(classifierPath + '\used_metrics.sav','wb'))
    pickle.dump(clf, open(classifierPath + '\classifier.sav', 'wb'))
   


def test_predictor(test_dataframe, classifierPath):
# This tests  the decoder to a given data table that has human labels. 
# Needs to include the metrics that the classifier was trained on.

    # cPath = os.path.dirname(inspect.getfile(run_predictor)); # find directory of this function and save pickle files there
    clf = pickle.load(open(classifierPath + '\classifier.sav', 'rb'))
    pipeline = pickle.load(open(classifierPath + '\preprocess_umap_pipeline.sav', 'rb'))
    used_metrics = pickle.load(open(classifierPath + '\used_metrics.sav', 'rb'))

    # check if all metrics are present
    foundMetrics = set(used_metrics) & set(list(test_dataframe.columns))
    if len(foundMetrics) != len(used_metrics):
        print('Missing metrics in dataset:')
        print(set(test_dataframe.columns).symmetric_difference(set(used_metrics)))
        raise ValueError('!! Columns in test dataset do not contain all required metrics !!')        

    X_unseen = test_dataframe[used_metrics]    
    X_test_final = pipeline.transform(X_unseen) #to impute and normalise and umap embed

    y_pred = clf.predict(X_test_final) #for every row 
    test_dataframe['is_noise']=y_pred

    fig = plot_decision_regions(X=X_test_final, y=test_dataframe['gTruth'].to_numpy(), clf=clf, legend=2)
    # plt.savefig('decision-boundary.png')
    plt.show()

    confusion_matrix = calculate_confusion_matrix(test_dataframe)
        
    return test_dataframe['is_noise'], confusion_matrix


#%%
# if __name__=='main':
    # dataset = pd.read_csv(r'D:\dataset.csv')
    # print(dataset.columns)
    # print(dataset.shape)
      
    # metric_data = dataset[[     
    #                              'syncSpike_2',
    #                              'syncSpike_4',
    #                               'firing_rate',
    #                               'presence_ratio',
    #                               'nn_hit_rate',
    #                               'nn_miss_rate',
    #                               'cumulative_drift' ]].values
    # X = metric_data
    # y = dataset['gTruth'].values
    
    # preprocessing_pipeline = create_preprocessing_pipeline()
    # run_search(preprocessing_pipeline, X, y) # call function 
    # train_best_estimator(preprocessing_pipeline,X,y)
#%%