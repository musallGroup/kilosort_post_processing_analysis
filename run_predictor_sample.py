# -*- coding: utf-8 -*-

"""
@author: Jain 
"""
#cell 0 
import pandas as pd
import os
import pickle
from BayesianSearch import clfs
from sklearn.model_selection import train_test_split
from preprocessing import merge_frames, preprocess_frame,get_n_spikes,create_preprocessing_pipeline
from BayesianSearch import run_search
import pickle
import umap
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
#%% load path cell1

def train_best_estimator(preprocessing_pipeline,X,y):
    import json
    config = json.load(open('incumbent_config.json'))
    
    supervised_embedder = umap.UMAP(
        min_dist=0.0, 
        n_neighbors=10, 
        n_components=2, # dimensions 
        random_state=42)    #print(type(supervised_embedder))
        
    X_train_transform = preprocessing_pipeline.fit_transform(X)
    X_train_final = supervised_embedder.fit_transform(X_train_transform, y)
    
    clf = clfs[config['estimator']]
    params = config['params']
    clf.set_params(**params)
    clf.fit(X_train_final,y)
    
    #import os
    #os.mkdir('./pipeline')
    
    
    
    pickle.dump(preprocessing_pipeline, open('preprocessing_pipeline.sav','wb'))
    pickle.dump(supervised_embedder, open('embedder.sav', 'wb'))
    pickle.dump(clf, open('classifier.sav', 'wb'))
   
def run_predictor(test_dataframe):
    
    X_unseen = test_dataframe[[  'n_spike', 
                             'syncSpike_2',
                             'syncSpike_4',
                              'firing_rate',
                              'presence_ratio',
                              'nn_hit_rate',
                              'nn_miss_rate',
                              'cumulative_drift' ]].values

    
    pipeline =  pickle.load(open('preprocessing_pipeline.sav', 'rb'))
    umap_embedder = pickle.load(open('embedder.sav', 'rb'))
    clf=  pickle.load(open('classifier.sav', 'rb'))
    #X_test_final = pipeline.transform(X_unseen)
    X_test_trans = pipeline.transform(X_unseen) #to impute and normalise
    X_test_final = umap_embedder.transform(X_test_trans)
    y_pred = clf.predict(X_test_final) #for every row 
    test_dataframe['is_noise']=y_pred
    print(X_test_final.shape)
    print(y_pred.shape)
    fig = plot_decision_regions(X=X_test_final, y=y_pred, clf=clf, legend=2)
    plt.savefig('decision-boundry.png')
    
    
    
    
    return test_dataframe['is_noise']


#%%
if __name__=='main':
    dataset = pd.read_csv(r'dataset\dataset.csv')
    print(dataset.columns)
    print(dataset.shape)
      
    metric_data = dataset[[      'n_spike', 
                                 'syncSpike_2',
                                 'syncSpike_4',
                                  'firing_rate',
                                  'presence_ratio',
                                  'nn_hit_rate',
                                  'nn_miss_rate',
                                  'cumulative_drift' ]].values
    X = metric_data
    y = dataset['gTruth'].values
    
    preprocessing_pipeline = create_preprocessing_pipeline()
    run_search(preprocessing_pipeline, X, y) # call function 
    train_best_estimator(preprocessing_pipeline,X,y)
#%%