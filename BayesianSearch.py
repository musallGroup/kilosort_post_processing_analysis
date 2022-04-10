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

seed=1
# CASH (Combined Algorithm Selection and Hyperparameter optimisation)
                   
# define search space, you can  add your own models 
clfs = {
        'AdaBoostClassifier' : AdaBoostClassifier(random_state=seed),
        'GradientBoostingClassifier' :GradientBoostingClassifier(random_state=seed),
        'RandomForestClassifier' :RandomForestClassifier(random_state=seed,n_jobs=-1),
        'KNeighborsClassifier': KNeighborsClassifier(n_jobs=-1),
        # 'SVC': SVC(random_state=seed,probability=True),
        'MLPClassifier' :MLPClassifier(random_state=seed, max_iter=300,hidden_layer_sizes= (50, 100)),
        'ExtraTreesClassifier' : ExtraTreesClassifier(n_estimators=100, random_state=0),
        'XGBClassifier' : XGBClassifier(n_estimators=100, random_state=0),
        'LGBMClassifier' : LGBMClassifier(random_state=0)
}
models =  list(clfs.keys())
# models = [ 'AdaBoostClassifier',
#            'GradientBoostingClassifier',
#            'RandomForestClassifier',
#            'KNeighborsClassifier',
#            'SVC',
#            'MLPClassifier',
#            'ExtraTreesClassifier',
#            'XGBClassifier',
#            'LGBMClassifier'] 
          
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
# run search with given dataset        
def run_search(preprocessing_pipeline, X, y, classifierPath):
    print('performing bayesian search for best classifier')

    X_train, X_test, y_train, y_test = train_test_split(X, y) #(train and validate on 75%, test on 25% of data)
    X_train_transform = preprocessing_pipeline.fit_transform(X_train)
    
    supervised_embedder = umap.UMAP(
    min_dist=0.0, 
    n_neighbors=10, 
    n_components=2, # dimensions 
    random_state=42)
    X_train_final = supervised_embedder.fit_transform(X_train_transform, y=y_train)
    X_test_trans = preprocessing_pipeline.transform(X_test) #to impute and normalise
    X_test_final = supervised_embedder.transform(X_test_trans)

    usedMetrics = list(X.columns)
    # cPath = os.path.dirname(inspect.getfile(run_search)); # find directory of this function and save pickle files there
    pickle.dump([supervised_embedder, usedMetrics], open(classifierPath + '\crossVal_embedder.sav', 'wb'))

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
    