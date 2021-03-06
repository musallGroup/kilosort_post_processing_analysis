# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 13:47:19 2022

@author: jain
"""
fPath = r'D:\SharedEcephys\FromFerimos\TestData\Kilosort2_2021-10-06_000401'
# fPath = r'Z:\invivo_ephys\SharedEphys\FromGia\recording1'
# fPath = r'D:\test'

import os
import logging
import time
import numpy as np
import pandas as pd
from utils import load_kilosort_data 
from metrics import calculate_metrics
from id_noise_templates import id_noise_templates
from preprocessing import get_feature_columns, preprocess_frame, get_n_spikes
from run_predictor_sample import run_predictor, identify_best_estimator
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

import baseParams
baseParams.QMparams['quality_metrics_output_file'] = fPath + r'\metrics.csv'
baseParams.noiseParams['classifier_path'] = os.path.abspath(os.getcwd()) + r'\rf_classifier.pkl'
baseParams.noiseParams['noise_templates_output_file'] = fPath + r'\noiseModule.csv'

# import QMparams

os.chdir(fPath)
import params
  

start = time.time()

#%% load spike data
print("Loading data...")

spike_times, spike_clusters, spike_templates, amplitudes, templates, channel_map, clusterIDs, cluster_quality, pc_features, pc_feature_ind = \
    load_kilosort_data(fPath, params.sample_rate, use_master_clock = False, include_pcs = True)


#%% run quality metric module and save output
print('running quality metrics module')
Qmetrics = calculate_metrics(spike_times, 
    spike_clusters, 
    spike_templates,
    amplitudes, 
    channel_map, 
    pc_features, 
    pc_feature_ind,
    baseParams.QMparams)

Qmetrics.to_csv(baseParams.QMparams['quality_metrics_output_file'])

#%% #https://medium.com/@pratap_aish/how-do-i-run-my-matlab-functions-in-python-7d2b8b2fefd1
import matlab.engine
eng = matlab.engine.start_matlab()
eng.pC_getSyncMetric(fPath)

#%% run noise Classifier and save output
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
import umap
from sklearn.pipeline import Pipeline
from preprocessing import create_preprocessing_pipeline
# baseParams.useNoiseMetrics contains all the metrics of interest. 
# make sure this includes cluster_id
fPath = r'D:\SharedEcephys\Ferimos_data\FromFermino\Kilosort2_2021-03-13_180605'
# baseParams.useNoiseMetrics = list(set(baseParams.useNoiseMetrics))
# baseParams.useNoiseMetrics.append('cluster_id')
frame_rec0 = get_feature_columns([fPath], baseParams.useNoiseMetrics) # get metrics from recording

# get ground Truth labels # Assumption: labels are binary and in string. example: neural ,noise
# when manual labelling is in phy, the changes are reflected in cluster_group.csv 
#gTruth = get_feature_columns([fPath], ['group']) 

classifierPath = r'Y:\invivo_ephys\SharedEphys\FromFermino\new_clf'
#run_predictor(frames[0][0],classifierPath) # test classifier

preprocessing_pipeline = create_preprocessing_pipeline()

pipeline_oldClf = Pipeline([
    ('pre', preprocessing_pipeline),
    ('umap-embedder', umap.UMAP(min_dist=0.0, n_neighbors=10, n_components=2,random_state=4)),
    ('clf', KNeighborsClassifier(n_jobs=-1, n_neighbors=20, weights='distance'))
    ])
eclf = VotingClassifier(estimators= None, voting='soft',n_jobs=-1)
eclf.estimators_ = pipeline_oldClf
eclf.le_ = LabelEncoder().fit(y_train_res) #https://stackoverflow.com/questions/42920148/using-sklearn-voting-ensemble-with-partial-fit/54610569#54610569
eclf.classes_ = eclf.le_.classes_
y_pred_soft = eclf.predict(frame_rec0) 
y_probs = eclf.predict_proba(frame_rec0)



mapping = {False: 'neural', True: 'noise'}
labels = [mapping[value] for value in frames[0][0]['is_noise']]

df = pd.DataFrame(data={'cluster_id' : frames[0][0]['cluster_id'], 'group': labels})
df.to_csv(fPath + r'\cluster_noiseClassifier.csv')


#%% load ground truth from Phy labels and retrain classifier

gTruth = get_feature_columns([fPath], ['group']) # get metrics from recording
a = pd.get_dummies(gTruth[0][0]['group'])
frames[0][0]['gTruth'] = a['noise']

identify_best_estimator(frames[0][0], baseParams.useNoiseMetrics) # re-train classifier
run_predictor(frames[0][0]) # test classifier
  


#%% run noise_template module and save output
print('running noise_template module')
nClusterIDs = clusterIDs[clusterIDs <= templates.shape[0]]

cluster_ids, is_noise = id_noise_templates(nClusterIDs, 
                                           templates, 
                                           np.squeeze(channel_map), 
                                           baseParams.noiseParams)

mapping = {False: 'neural', True: 'noise'}
labels = [mapping[value] for value in is_noise]
df = pd.DataFrame(data={'cluster_id' : cluster_ids, 'group': labels})
df.to_csv(fPath + r'\noiseModule.csv')
                                 
#%%
import pandas as pd
df_noise = pd.read_csv(r'Y:\invivo_ephys\SharedEphys\TestData\Kilosort2_2021-10-06_000401\noiseClassifier_updated.csv')
del df_noise['Unnamed: 0']
df = pd.read_csv(r'Y:\invivo_ephys\SharedEphys\TestData\Kilosort2_2021-10-06_000401\metrics.csv')
df['noiseClassifier_updated'] = df_noise['group']
df.to_csv('Y:\invivo_ephys\SharedEphys\TestData\Kilosort2_2021-10-06_000401\metrics.csv')        
                    

                
                
                
             

