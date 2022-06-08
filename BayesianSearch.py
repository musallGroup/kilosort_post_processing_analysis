from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neural_network import MLPClassifier 
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
import lightgbm 
from lightgbm import LGBMRegressor, LGBMClassifier, Booster
from skopt import BayesSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlxtend.plotting import plot_decision_regions
import umap.umap_ as umap
import pickle
import json
import os
import inspect
import numpy as np

from funcs_to_use import get_classifiers_and_params, get_preprocessing_umap_pipeline, get_smote_resampled, get_supervised_embedder, params_to_estimator
from preprocessing import create_preprocessing_pipeline

def get_clf_best_estimators_per_model_family(
    X,
    y,
    preprocess_pipeline,
    kfold,
    seed,
    scoring_metric='balanced_accuracy',
    n_iter=20,
    n_jobs=-1,
    verbose=2):
    "Using BayesSearchCV"

    clfs, models, params = get_classifiers_and_params()

    X_transformed= preprocess_pipeline.fit_transform(X)

    #del clf_best_estimators
    clf_best_estimators = []
    best_val_scores = []
    best_params = []
    # run search for each model 
    for name in models:
        print(f"Starting BayesSearchCV for {name}")
        # add print statements
        estimator = clfs[name]
        init_args = {"random_state": seed} if name != "KNeighborsClassifier" else {}
        clf = BayesSearchCV(
            estimator(**init_args),
            params[name],
            scoring=scoring_metric,
            refit=False,
            n_jobs=n_jobs,
            n_iter=n_iter,
            cv=kfold,
            random_state=seed,
            verbose=verbose
            ) 
        clf.fit(X_transformed, y)   # X_train_final, y_train X is train samples and y is the corresponding labels
        # print("best estimator " +  str(clf.best_estimator_))
        print("best params: " + str(clf.best_params_))
        clf_best_estimators.append(params_to_estimator(clf.best_params_, name))
        best_val_scores.append(clf.best_score_)
        best_params.append(clf.best_params_)
        # fig = plot_decision_regions(X=X_train_final, y=y_train, clf=clf, legend=2)
        # plt.title(f'Decison boundary of {name} on clusters');
        # plt.show()

    return clf_best_estimators, best_val_scores, best_params


# run search with given dataset        
def identify_best_estimator(
    X,
    y,
    exp_dir,
    preprocess_pipeline,
    metrics,
    seed,
    kfold,
    scoring_metric='balanced_accuracy',
    n_iter=20,
    n_jobs=-1,
    verbose=2
):
    validate_data(X, metrics)

    print('performing bayesian search for finding the best classifier configuration')
    # CASH (Combined Algorithm Selection and Hyperparameter optimisation)
    # time passes
    best_estimators, val_scores, best_configs = get_clf_best_estimators_per_model_family(
                                                    X,
                                                    y,
                                                    preprocess_pipeline,
                                                    kfold,
                                                    seed,
                                                    scoring_metric=scoring_metric,
                                                    n_iter=n_iter,
                                                    n_jobs=n_jobs,
                                                    verbose=verbose)

    max_value = max(val_scores)  # Return the max value of the list
    max_index = val_scores.index(max_value)
    best_config = best_configs[max_index]    
    estimator=best_estimators[max_index]

    # save incumbent(best) config 
    incumbent_config= {
        'estimator_name': estimator.__class__.__name__,
        'params': best_config
    }
    
    print(f"best classifier is: {estimator} with configuration: ", best_config)
    json.dump(incumbent_config, open(os.path.join(exp_dir, 'incumbent_config.json'), 'w'))
    return estimator


def validate_data(X, metrics):
    # check if all metrics are present
    foundMetrics = set(metrics) & set(list(X.columns))
    if len(foundMetrics) != len(metrics):
        print('Missing metrics in dataset:')
        print(set(X.columns).symmetric_difference(set(metrics)))
        raise ValueError('!! Columns in dataset do not contain all required metrics !!')
    
    # make sure not to use cluster_id as a metric
    if len(set(X.columns).intersection(set(['cluster_id']))) > 0:
        del X['cluster_id']
    