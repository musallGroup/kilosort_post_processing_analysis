# -*- coding: utf-8 -*-
"""
Created on Mon May 30 14:01:18 2022

@author: jain
"""

import copy
from itertools import filterfalse

import pandas as pd 

import matplotlib.pyplot as plt

import numpy as np

import umap

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

def get_clf_best_estimators_list(X, y, preprocess_umap_pipeline: BaseEstimator, kfold, seed):
    "Using BayesSearchCV"

    clfs, models, params = get_classifiers_and_params(seed)
   
    X_res, y_res = get_smote_resampled(X, y, seed)

    X_transformed= preprocess_umap_pipeline.fit_transform(X_res)

    #del clf_best_estimators
    clf_best_estimators = []
      
    # run search for each model 
    for name in models:
        print(name)
        # add print statements
        estimator = clfs[name]
        clf = BayesSearchCV(estimator, params[name], scoring='accuracy', refit='True', n_jobs=-1, n_iter=20, cv=kfold) 
        clf.fit(X_transformed, y_res)   # X_train_final, y_train X is train samples and y is the corresponding labels
        
        print("best estimator " +  str(clf.best_estimator_))
        print("best params: " + str(clf.best_params_))
        clf_best_estimators.append(clf.best_estimator_)

    return clf_best_estimators


def get_supervised_embedder(seed):
    supervised_embedder = umap.UMAP(min_dist=0.0, n_neighbors=10, n_components=2,random_state=seed)
    return supervised_embedder


def get_preprocessing_umap_pipeline(seed):
    preprocessing_pipeline = create_preprocessing_pipeline()

    supervised_embedder = get_supervised_embedder(seed)

    preprocess_umap_pipeline = Pipeline([
    ('pre', preprocessing_pipeline),
    ('umap-embedder', supervised_embedder)
    ])
        
    return preprocess_umap_pipeline


def get_classifiers_and_params(seed):
    clfs = {
            'AdaBoostClassifier' : AdaBoostClassifier(random_state=seed),
            'GradientBoostingClassifier' :GradientBoostingClassifier(random_state=seed),
            'RandomForestClassifier' :RandomForestClassifier(random_state=seed,n_jobs=-1),
            'KNeighborsClassifier': KNeighborsClassifier(n_jobs=-1),
            'SVC': SVC(random_state=seed,probability=True),
            'MLPClassifier' :MLPClassifier(random_state=seed, max_iter=300,hidden_layer_sizes= (50, 100)),
            'ExtraTreesClassifier' : ExtraTreesClassifier(n_estimators=100, random_state=seed),
            'XGBClassifier' : XGBClassifier(n_estimators=100, random_state=seed,use_label_encoder=False),
            'LGBMClassifier' : LGBMClassifier(random_state=seed)
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
                'SVC': {'C':[0.5,2.5],
                           'kernel':['sigmoid','linear','poly','rbf']
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


def create_models_for_voting_clfs(X,y,preprocess_umap_pipeline,clf_best_estimators,kfold, seed):
 
    X_res, y_res = get_smote_resampled(X, y, seed)
    classifiers = []

    for classifier in clf_best_estimators:
        clf_pipeline = Pipeline([
            ('pre-umap', preprocess_umap_pipeline),
            ('estimator', classifier)])
        
        for i, (train_index, _) in enumerate(kfold.split(X_res, y_res)):
            X_train_in = X_res.iloc[train_index]
            y_train_in = y_res.iloc[train_index]
            print("running estimator : " , classifier )
            print("running split : ", i )
            fitted_clf  = clf_pipeline.fit(X_train_in,y_train_in)
            classifiers.append(copy.deepcopy(fitted_clf))

    return classifiers

def get_smote_resampled(X, y, seed):
    sm = SMOTE(random_state=seed)
    #ad list to test sizes

    # to get equal distribution for labels in y_ytain
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res

#def predict_voting_clfs_output(X,y,pipeline_config,clf_best_estimators,kfold):

def predict_using_voting_clf(classifiers, X_test, y_test):

    eclf = VotingClassifier(estimators= None, voting='soft',n_jobs=-1)
    eclf.estimators_ = classifiers
    eclf.le_ = LabelEncoder().fit(y_test) #https://stackoverflow.com/questions/42920148/using-sklearn-voting-ensemble-with-partial-fit/54610569#54610569
    eclf.classes = eclf.le_.classes_
    y_preds = eclf.predict(X_test) 
    y_probs = eclf.predict_proba(X_test)
    
    return y_preds, y_probs

def plot_cm(y_true,y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=None)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax,cmap='Blues');  #annot=True to annotate cells, ftm='g' to disable scientific notation
    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title(f'Confusion Matrix from {title}'); 
    ax.xaxis.set_ticklabels(['Neural', 'Noise']); ax.yaxis.set_ticklabels(['Neural', 'Noise']);
    #plt.savefig('Confusion Matrix from classifier (SUA).pdf') 
    
    plt.show()


def fit_with_fitted_pipeline(X,y,fitted_pipeline):
 
    X, y = shuffle(X, y, random_state=0)
    #kfold = StratifiedKFold(n_splits=8, shuffle = True,random_state = 42)
    sm = SMOTE(random_state=42)
    #ad list to test sizes
    X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)


    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    fitted_clf  = fitted_pipeline.fit(X_train_res, y_train_res)
    return fitted_clf

def predict_with_fitted_clf(dataframe,fitted_clf,metrics_trained_on):
    dataframe = remove_miss_vals(dataframe)
    dataframe = dataframe[metrics_trained_on]
    y_preds = fitted_clf.predict(dataframe)
    y_probs = fitted_clf.predict_proba(dataframe)
    
    return y_preds,y_probs
    






























