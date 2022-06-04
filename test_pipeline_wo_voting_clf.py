# -*- coding: utf-8 -*-
"""
Created on Sun May 29 23:32:32 2022

@author: jain
"""
targFolder = r'D:\SharedEcephys\Ferimos_data\FromFermino'

# get modules and params
import os
import random
import numpy as np
import pickle

# ML packages
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

from funcs_to_use import (
    fit_with_fitted_pipeline,
    get_preprocessing_umap_pipeline,
    metrics_to_use,
    get_leave_one_out_frame,
    plot_cm,
    predict_with_fitted_pipeline,
    split_data_to_X_y
)
from run_predictor_sample import identify_best_estimator


# path to default classifier (repository link)
defClassifierPath = r"C:\Users\jain\Documents\GitHub\kilosort_post_processing_analysis"
seed=0
CV_SPLITS = 2
np.random.seed(seed)
random.seed(seed)

# get recordings and keep the ones that have the cluster_group.tsv file
folderCheck = os.listdir(targFolder)
file_paths = []
for i, path in enumerate(folderCheck):
    if os.path.isfile(os.path.join(targFolder,path,'cluster_group.tsv')):
        file_paths.append(os.path.join(targFolder,path))

       
strict_metrics = metrics_to_use(file_paths) # quality metrics with auc-roc greater than threshold
keep_data_frames, leave_out_dfs = get_leave_one_out_frame(file_paths,strict_metrics)

X_train, y_train = split_data_to_X_y(keep_data_frames[0])
X_test, y_test = split_data_to_X_y(leave_out_dfs[0])

preprocess_umap_pipeline = get_preprocessing_umap_pipeline(seed)
kfold = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=seed)

classifierPath = r'D:\SharedEcephys\Ferimos_data\FromFermino\train_new_metric_framesnot5'
identify_best_estimator(keep_data_frames, strict_metrics, classifierPath, seed)
clf = pickle.load(open(classifierPath + '\classifier.sav', 'rb'))
# only load once as both are in the same pipeline
pipeline = pickle.load(open(classifierPath + '\preprocess_umap_pipeline.sav', 'rb'))
used_metrics = pickle.load(open(classifierPath + '\used_metrics.sav', 'rb'))

fitted_pipeline = Pipeline([
    ('pre-umap' , pipeline),
    ('clf' , clf)
    ])

fitted_clf = fit_with_fitted_pipeline(X_train, y_test, fitted_pipeline, seed)
y_preds,y_probs = predict_with_fitted_pipeline(X_test, fitted_clf, strict_metrics)

totalProbs = np.abs(y_probs[:,0] - y_probs[:,1])

plot_cm(y_test, y_preds, 'bayesian classifier')  
