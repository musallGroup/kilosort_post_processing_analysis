# -*- coding: utf-8 -*-
"""
Created on Mon May 30 14:01:18 2022

@author: jain
"""

import copy
from functools import partial
from itertools import filterfalse
import os
from typing import NamedTuple

import pandas as pd 

import matplotlib.pyplot as plt

import numpy as np

import umap.umap_ as umap

import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    VotingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier 
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from skopt import BayesSearchCV

from imblearn.over_sampling import SMOTE

from preprocessing import get_feature_columns,remove_miss_vals,get_roc_metrics, create_preprocessing_pipeline
import baseParams

def split_data_to_X_y(data):
    X = data.drop(['gTruth','cluster_id'], axis=1)
    y = data['gTruth'] 
    return X, y


def metrics_to_use(file_paths):
    "Use this function on the all the recordings given in fPath"
    "to find quality metrics in metrics_params with ROC-AUC values"
    
    "output is the quality metrics used for training"
    frames = get_feature_columns(file_paths, baseParams.get_QMetrics()) 
    gTruths = get_feature_columns(file_paths, ['group']) 

    #create a single large dataframe
    frame = pd.concat(frames[0],axis = 0)
    gTruth = pd.concat(gTruths[0],axis = 0)

    a = pd.get_dummies(gTruth['group'])
    frame['gTruth'] = a['noise']

    # data pre-processing
    frame = remove_miss_vals(frame)
    #calculating roc_metrics
    auc_vals_frame = get_roc_metrics(frame)

    keep_cols =  np.where((auc_vals_frame.roc_auc_val > 0.79) | (auc_vals_frame.roc_auc_val < 0.21))[0].tolist() # get metric with high AUC values 
    strict_metrics = frame.columns[keep_cols].values # added to baseParams.AUC_Metrics_fermino

    return strict_metrics

def get_leave_one_out_frame(file_paths,strict_metrics):
    """

    """
    keep_data_frames = []
    leave_out_dfs = []
    path_ids = list(range(len(file_paths)))
    for i in path_ids:
        leave_out_path = i
        keep_path_ids = list(filterfalse(lambda x: x == leave_out_path, path_ids))

        metrics_frames_merged_keep = get_merged_df_from_path_ids(file_paths, strict_metrics, keep_path_ids)
        metrics_frames_merged_leave = get_merged_df_from_path_ids(file_paths, strict_metrics, [leave_out_path])
        keep_data_frames.append(metrics_frames_merged_keep)
        leave_out_dfs.append(metrics_frames_merged_leave)
        
    return keep_data_frames, leave_out_dfs

def get_merged_df_from_path_ids(file_paths, strict_metrics, path_ids):
    metrics_frames, _ = get_feature_columns(np.array(file_paths)[path_ids], strict_metrics)
    gtruth_frames, _ = get_feature_columns(np.array(file_paths)[path_ids], ['group']) 

    metrics_frames_merged = pd.concat(metrics_frames,axis = 0)
    metrics_frames_merged = metrics_frames_merged.reset_index(drop=True)
        
    gtruth_frames_merged = pd.concat(gtruth_frames, axis = 0)
    gtruth_frames_merged = gtruth_frames_merged.reset_index(drop=True)
        
    ohe_gtruth_frames_merged = pd.get_dummies(gtruth_frames_merged['group'])
    ohe_gtruth_frames_merged = ohe_gtruth_frames_merged.reset_index(drop=True)

    metrics_frames_merged['gTruth'] = ohe_gtruth_frames_merged['noise']
    metrics_frames_merged = remove_miss_vals(metrics_frames_merged)

    return metrics_frames_merged


def params_to_estimator(params, name):
    clfs, _, _ = get_classifiers_and_params()
    estimator = clfs[name](**params)
    return estimator


def get_supervised_embedder(seed):
    supervised_embedder = umap.UMAP(min_dist=0.0, n_neighbors=10, n_components=2,random_state=seed)
    return supervised_embedder


def get_preprocessing_umap_pipeline(seed):
    preprocessing_pipeline = create_preprocessing_pipeline(seed)

    supervised_embedder = get_supervised_embedder(seed)

    preprocess_umap_pipeline = Pipeline([
    ('pre', preprocessing_pipeline),
    ('umap-embedder', supervised_embedder)
    ])
        
    return preprocess_umap_pipeline


def get_classifiers_and_params():
    clfs = {
            'AdaBoostClassifier': AdaBoostClassifier,
            'GradientBoostingClassifier': GradientBoostingClassifier,
            'RandomForestClassifier': RandomForestClassifier,
            'KNeighborsClassifier': KNeighborsClassifier,
            #'SVC': SVC,
            'MLPClassifier': MLPClassifier,
            'ExtraTreesClassifier': ExtraTreesClassifier,
            'XGBClassifier': partial(XGBClassifier, use_label_encoder=False),
            'LGBMClassifier': LGBMClassifier,
    }
    models =  list(clfs.keys())
              
    params = {
                'AdaBoostClassifier':{'learning_rate':[1,2], 
                           'n_estimators':[50,100],
                           'algorithm':['SAMME','SAMME.R']
                           },#AdaB
                'GradientBoostingClassifier':{'learning_rate':[0.05,0.1],
                           'n_estimators':[100,150], 
                           'max_depth':[2,4],
                           'min_samples_split':[2,4],
                           'min_samples_leaf': [2,4]
                           }, #GBC
                'RandomForestClassifier':{'n_estimators':[100,150],
                           'criterion':['gini','entropy'],
                           'min_samples_split':[2,4],
                           'min_samples_leaf': [2,4]
                           }, #RFC
                'KNeighborsClassifier':{'n_neighbors':[20,50], 
                           'weights':['distance','uniform'],
                           'leaf_size':[30]
                           }, #KNN
                'SVC': {
                        'C': [0.5,2.5],
                        'kernel': ['sigmoid','linear','rbf'],
                        'probability': [True]
                        }, #SVC
                'MLPClassifier': {
                             'activation': ['tanh', 'relu'],
                             'solver': ['sgd', 'adam'],
                             'alpha': [0.0001, 0.05],
                             'learning_rate': ['constant','adaptive']
                             }, #MLP
                'ExtraTreesClassifier':{'criterion':['gini', 'entropy'],  
                           'class_weight':['balanced', 'balanced_subsample']
                           }, #extratrees
                 'XGBClassifier':{'max_depth':[2,4], 
                           'eta': [0.2,0.5], 
                           'sampling_method':['uniform','gradient_based'],
                           'grow_policy':['depthwise', 'lossguide']
                          }, #xgboost
                'LGBMClassifier':{'learning_rate':[0.05,0.15],
                           'n_estimators': [100,150]} #lightgbm
             }
             
    return clfs,models,params


def get_nulls(training, testing):
    print("Training Data:")
    print(pd.isnull(training).sum())
    print("Testing Data:")
    print(pd.isnull(testing).sum())


def create_models_for_voting_clfs(X,y,preprocess_pipeline,clf_best_estimators,seed,kfold): #kfold=None
    classifiers = []

    for classifier in clf_best_estimators:
        clf_pipeline = Pipeline([
            ('pre-umap', preprocess_pipeline),
            ('estimator', classifier)])
        
        if kfold is not None:
            for i, (train_index, _) in enumerate(kfold.split(X, y)):
                X_train_in = X.iloc[train_index]
                y_train_in = y.iloc[train_index]
                print("running estimator : " , classifier )
                print("running split : ", i )
                fitted_clf  = clf_pipeline.fit(X_train_in,y_train_in)
                classifiers.append(copy.deepcopy(fitted_clf))
        else:
            fitted_clf  = clf_pipeline.fit(X,y)
            classifiers.append(copy.deepcopy(fitted_clf))

    return classifiers

def get_smote_resampled(X, y, seed):
    sm = SMOTE(random_state=seed)
    #ad list to test sizes

    # to get equal distribution for labels in y_ytain
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res

#def predict_voting_clfs_output(X,y,pipeline_config,clf_best_estimators,kfold):

def predict_using_voting_clf(eclf, X_test, y_test):

    y_preds = eclf.predict(X_test) 
    y_probs = eclf.predict_proba(X_test)
    
    return y_preds, y_probs

def plot_cm(y_true,y_pred, title,exp_dir):
    cm = confusion_matrix(y_true, y_pred, labels=None)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax,cmap='Blues');  #annot=True to annotate cells, ftm='g' to disable scientific notation
    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title(f'Confusion Matrix from {title}'); 
    ax.xaxis.set_ticklabels(['Neural', 'Noise']); ax.yaxis.set_ticklabels(['Neural', 'Noise']);
    plt.savefig(os.path.join(exp_dir, f"confusion_matrix_{title}")) 
    #plt.show()


class ClassificationPerformanceConfusionMatrix(NamedTuple):
    true_positive_rate: int
    false_positive_rate: int
    true_negative_rate: int
    false_negative_rate: int
    total_performance: int

def calculate_confusion_matrix(test_dataframe):
    # confusionMatrix = pd.DataFrame(columns=['TruePositive', 'FalsePositive', 'TrueNegative', 'FalseNegative', 'TotalPerf'])

    #true positive (correctly recognized noise clusters)    
    true_positive_rate = np.sum((test_dataframe['is_noise'] == 1) & (test_dataframe['gTruth'] == 1)) / len(test_dataframe['gTruth'])
    
    #false alarm rate (percent falsely labeled neural clusters)    
    false_positive_rate = np.sum((test_dataframe['is_noise'] == 1) & (test_dataframe['gTruth'] == 0)) / len(test_dataframe['gTruth'])
 
    #true negative (correctly recognized neural clusters)    
    true_negative_rate = np.sum((test_dataframe['is_noise'] == 0) & (test_dataframe['gTruth'] == 0)) / len(test_dataframe['gTruth'])
    
    #false negatove (missed noise clusters)    
    false_negative_rate = np.sum((test_dataframe['is_noise'] == 0) & (test_dataframe['gTruth'] == 1)) / len(test_dataframe['gTruth'])
 
    #total performance    
    total_performance = np.round(np.sum(test_dataframe['is_noise'] == test_dataframe['gTruth']) / len(test_dataframe['gTruth']), 2)
    return ClassificationPerformanceConfusionMatrix(
        true_positive_rate=true_positive_rate,
        false_positive_rate=false_positive_rate,
        true_negative_rate=true_negative_rate,
        false_negative_rate=false_negative_rate,
        total_performance=total_performance
        )


def fit_with_fitted_pipeline(X, y, fitted_pipeline):
    fitted_clf  = fitted_pipeline.fit(X, y)
    return fitted_clf

def predict_with_fitted_pipeline(dataframe, fitted_clf, metrics_trained_on):
    dataframe = remove_miss_vals(dataframe)
    dataframe = dataframe[metrics_trained_on]
    y_preds = fitted_clf.predict(dataframe)
    y_probs = fitted_clf.predict_proba(dataframe)
    return y_preds,y_probs
