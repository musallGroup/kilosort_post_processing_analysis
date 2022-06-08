import pickle
import os
import random
import copy
import argparse
from tkinter.tix import Tree

import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from funcs_to_use import (
    get_preprocessing_umap_pipeline,
    get_smote_resampled,
    metrics_to_use,
    get_leave_one_out_frame,
    create_models_for_voting_clfs,
    predict_using_voting_clf,
    plot_cm,
    split_data_to_X_y
)
from preprocessing import create_preprocessing_pipeline
from BayesianSearch import get_clf_best_estimators_per_model_family

def run_voting_classifier(
    X_train,
    y_train,
    X_test,
    y_test,
    preprocess_pipeline,
    kfold,
    seed,
    exp_dir,
    strict_metrics
    ):
    
    clf_best_estimators, _, _ = get_clf_best_estimators_per_model_family(
        X_train,
        y_train,
        preprocess_pipeline,
        copy.deepcopy(kfold),
        seed
    )

    classifiers = create_models_for_voting_clfs(X_train, y_train, preprocess_pipeline,clf_best_estimators, seed=seed,kfold = copy.deepcopy(kfold)) # copy.deepcopy(kfold))or kfold=None
    print("number of classifier models generated : " , len(classifiers))
    
    eclf = get_voting_classifier(y_test, classifiers)

    y_pred, y_prob = predict_using_voting_clf(eclf, X_test, y_test)
    pickle.dump(eclf, open(os.path.join(exp_dir, 'classifier.pkl'), 'wb'))
    pickle.dump(strict_metrics, open(os.path.join(exp_dir, 'strict_metrics.pkl'), 'wb'))

    return y_pred, y_prob

def get_voting_classifier(y, classifiers, n_jobs=-1, voting='soft'):
    eclf = VotingClassifier(estimators=None, voting=voting,n_jobs=n_jobs)
    eclf.estimators_ = classifiers
    eclf.le_ = LabelEncoder().fit(y) #https://stackoverflow.com/questions/42920148/using-sklearn-voting-ensemble-with-partial-fit/54610569#54610569
    eclf.classes = eclf.le_.classes_
    return eclf
