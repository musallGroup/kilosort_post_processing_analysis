# -*- coding: utf-8 -*-
"""
Created on Sun May 29 23:32:32 2022

@author: jain
"""
# get modules and params
import os
import numpy as np
import pickle

# ML packages
from sklearn.pipeline import Pipeline

from funcs_to_use import (
    fit_with_fitted_pipeline,
    predict_with_fitted_pipeline,
)
from BayesianSearch import identify_best_estimator


def run_best_estimator(
    X_train,
    y_train,
    X_test,
   # y_test,
    preprocess_pipeline,
    kfold,
    seed,
    exp_dir,
    strict_metrics
):
    """
    fits the incumbent configuration(no voting classifier) found via bayes search
    on train data and predicts on test data.
    """
    best_estimator = identify_best_estimator(
        X=X_train, 
        y=y_train, 
        metrics=strict_metrics,
        exp_dir=exp_dir,
        seed=seed,
        kfold=kfold,
        preprocess_pipeline=preprocess_pipeline)
    # only load once as both are in the same pipeline

    fitted_pipeline = Pipeline([
        ('preprocess' , preprocess_pipeline),
        ('clf' , best_estimator)
        ])

    fitted_clf = fit_with_fitted_pipeline(X_train, y_train, fitted_pipeline)
    y_pred, y_prob = predict_with_fitted_pipeline(X_test, fitted_clf, strict_metrics)
    
    pickle.dump(fitted_clf, open(os.path.join(exp_dir, 'classifier.pkl'), 'wb'))
    pickle.dump(strict_metrics, open(os.path.join(exp_dir, 'strict_metrics.pkl'), 'wb'))
    return y_pred, y_prob, X_test
