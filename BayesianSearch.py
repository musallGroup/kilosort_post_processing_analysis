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

from funcs_to_use import get_classifiers_and_params, get_preprocessing_umap_pipeline, get_supervised_embedder

seed=1
# CASH (Combined Algorithm Selection and Hyperparameter optimisation)
clfs, models, params = get_classifiers_and_params(seed=seed)

# run search with given dataset        
def run_search(preprocessing_pipeline, X, y, classifierPath):
    print('performing bayesian search for best classifier')

    X_train, X_test, y_train, y_test = train_test_split(X, y) #(train and validate on 75%, test on 25% of data)
    
    preprocess_umap_pipeline = get_preprocessing_umap_pipeline(seed)
    X_train_final = preprocess_umap_pipeline.fit_transform(X_train, y=y_train)
    X_test_final = preprocess_umap_pipeline.transform(X_test)

    usedMetrics = list(X.columns)
    # cPath = os.path.dirname(inspect.getfile(run_search)); # find directory of this function and save pickle files there
    pickle.dump([preprocess_umap_pipeline, usedMetrics], open(classifierPath + '\crossVal_embedder.sav', 'wb'))

    # time passes
    test_scores = []
    val_scores =[]
    search_objects=[]
    best_configs = []  
    
    # run search for each model 
    for name in models:
        print(name)
        estimator = clfs[name]
        clf = BayesSearchCV(estimator, params[name], scoring='accuracy', refit='True', n_jobs=-1, n_iter=20,cv=5) 
        clf.fit(X_train_final, y_train)   # X_train_final, y_train X is train samples and y is the corresponding labels
        
        print("best params: " + str(clf.best_params_)) # best paramters for each model 
        print("best scores: " + str(clf.best_score_))
        print("best estimator " +  str(clf.best_estimator_))
        best_configs.append(clf.best_params_)
        val_scores.append(clf.best_score_)
        search_objects.append(clf)

        fig = plot_decision_regions(X=X_train_final, y=y_train, clf=clf, legend=2)
        plt.title(f'Decison boundary of {name} on clusters');
        plt.show()
        
        clfscore=clf.score(X_test_final, y_test) # X_test_final, y_test
        test_scores.append(clfscore)
    
    max_value = max(val_scores)#Return the max value of the list
    max_index = val_scores.index(max_value)
    best_config = best_configs[max_index]    
    estimator=models[max_index]
    
    # save incumbent(best) config 
    incumbent_config= {
        'estimator': estimator, 
        'params' : best_config
    }
    
    print("best_config with Configuration is ", incumbent_config)
    json.dump(incumbent_config, open(classifierPath + '\incumbent_config.json', 'w'))
    