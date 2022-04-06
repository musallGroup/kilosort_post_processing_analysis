# -*- coding: utf-8 -*-

"""
@author: Jain 
"""
#cell 0 
import os
import pickle
from BayesianSearch import clfs, run_search
from preprocessing import create_preprocessing_pipeline
import umap
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import json
import inspect

#%% load path cell1

def train_best_estimator(preprocessing_pipeline,X,y):
# this uses the classifier that has been previously identified as performing
# best and fits it to a new supervised UMAP projection based on X and y.

    # make sure not to use cluster_id as a metric
    if len(set(X.columns).intersection(set(['cluster_id']))) > 0:
        del X['cluster_id']
        
    usedMetrics = list(X.columns)

    supervised_embedder = umap.UMAP(
        min_dist=0.0, 
        n_neighbors=10, 
        n_components=2, # dimensions 
        random_state=42)    #print(type(supervised_embedder))
        
    X_train_transform = preprocessing_pipeline.fit_transform(X)
    X_train_final = supervised_embedder.fit_transform(X_train_transform, y)
    
    cPath = os.path.dirname(inspect.getfile(train_best_estimator)); # find directory of this function and save pickle files there
    config = json.load(open(cPath + '\incumbent_config.json'))
    clf = clfs[config['estimator']]
    params = config['params']
    clf.set_params(**params)
    clf.fit(X_train_final,y)
    
    
    pickle.dump(preprocessing_pipeline, open(cPath + '\preprocessing_pipeline.sav','wb'))
    pickle.dump([supervised_embedder, usedMetrics], open(cPath + '\embedder.sav','wb'))
    pickle.dump(clf, open(cPath + '\classifier.sav', 'wb'))
   
    
def run_predictor(test_dataframe):
# This applies the decoder to a given data table. Needs to include the metrics
# that the classifier was trained on.

    cPath = os.path.dirname(inspect.getfile(run_predictor)); # find directory of this function and save pickle files there
    clf = pickle.load(open(cPath + '\classifier.sav', 'rb'))
    pipeline =  pickle.load(open(cPath + '\preprocessing_pipeline.sav', 'rb'))
    embeder = pickle.load(open(cPath + '\embedder.sav', 'rb'))
    umap_embedder = embeder[0]
    usedMetrics = embeder[1]
    
    
    # check if all metrics are present
    foundMetrics = set(usedMetrics) & set(list(test_dataframe.columns))
    if len(foundMetrics) != len(usedMetrics):
        print('Missing metrics in dataset:')
        print(set(test_dataframe.columns).symmetric_difference(set(usedMetrics)))
        raise ValueError('!! Columns in test dataset do not contain all required metrics !!')
        
    
    X_unseen = test_dataframe[usedMetrics]    
    X_test_trans = pipeline.transform(X_unseen) #to impute and normalise
    X_test_final = umap_embedder.transform(X_test_trans)
    
    y_pred = clf.predict(X_test_final) #for every row 
    test_dataframe['is_noise']=y_pred
    fig = plot_decision_regions(X=X_test_final, y=y_pred, clf=clf, legend=2)
    plt.savefig('decision-boundary.png')
    
    return test_dataframe['is_noise']


def identify_best_estimator(dataset, metrics):
# this performs a new bayesian search to identify the best classifier in UMAP
# space. This is useful when retraining on a new flavor of data.

    # check if all metrics are present
    foundMetrics = set(metrics) & set(list(dataset.columns))
    if len(foundMetrics) != len(metrics):
        print('Missing metrics in dataset:')
        print(set(dataset.columns).symmetric_difference(set(metrics)))
        raise ValueError('!! Columns in dataset do not contain all required metrics !!')
    
    y = dataset['gTruth'].values #gTruth are the manual labels that are used for training
    X = dataset[dataset.columns.intersection(metrics)]    
    
    # make sure not to use cluster_id as a metric
    if len(set(X.columns).intersection(set(['cluster_id']))) > 0:
        del X['cluster_id']
        
    preprocessing_pipeline = create_preprocessing_pipeline()
    run_search(preprocessing_pipeline, X, y) # call function 
    train_best_estimator(preprocessing_pipeline,X,y)


#%%
# if __name__=='main':
    # dataset = pd.read_csv(r'D:\dataset.csv')
    # print(dataset.columns)
    # print(dataset.shape)
      
    # metric_data = dataset[[      #'n_spike', 
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