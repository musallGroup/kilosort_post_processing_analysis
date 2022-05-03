# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 13:47:19 2022

@author: jain
"""
fPath = r'Y:\invivo_ephys\SharedEphys\FromFermino\Kilosort2_2021-03-13_180605'
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
from preprocessing import merge_frames, preprocess_frame, get_n_spikes
from run_predictor_sample import run_predictor, identify_best_estimator
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from utilities import plot_roc_curve
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
df = pd.read_csv('metrics.csv')

# baseParams.useNoiseMetrics contains all the metrics of interest. 
# make sure this includes cluster_id



baseParams.useNoiseMetrics = list(set(baseParams.useNoiseMetrics))

frames = merge_frames([fPath], baseParams.useNoiseMetrics) # get metrics from recording

# get ground Truth labels # Assumption: labels are binary and in string. example: neural ,noise
# when manual labelling is in phy, the changes are reflected in cluster_group.csv 
gTruth = merge_frames([fPath], ['group']) 

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from preprocessing import create_preprocessing_pipeline

frame = frames[0][0]
def preprocess_dataframe(frame):
    # removing columns with high precentage of nans 
    if 'epoch_name' in frame.columns:
        frame.drop(['epoch_name'],axis = 1,inplace=True)
    if 'Var1' in frame.columns:
        frame.drop(['Var1'],axis = 1,inplace=True)
        
    percent_missing = frame.isnull().sum() * 100 / len(frame)
    p = percent_missing.where(percent_missing > 80.0).dropna()
    print("columns dropped : " ,  p)
    
    #interpolation to handle missing values
    trans_frame = frame.interpolate(axis = 1, limit =5, limit_direction= 'both' ) 
    percent_missing = trans_frame.isnull().sum() * 100 / len(trans_frame)
    # imputing the dataframe with    technique
    
    
    
    preprocessing_pipeline = create_preprocessing_pipeline()
    processed_frame = preprocessing_pipeline.fit_transform(trans_frame)
    return processed_frame

frames[0][0] = preprocess_dataframe(frames[0][0])

df_auc = plot_roc_curve(fPath,frames,gTruth) # get AUC values for each metric

rslt_df = df_auc.loc[(df_auc.roc_auc > 0.59) | (df_auc.roc_auc < 0.41)] # get metric with high AUC values 

rocMetrics = list(rslt_df.metric)
print(set(rocMetrics).intersection(set(baseParams.useNoiseMetrics)))
use_rocMetrics = set(rocMetrics).intersection(set(baseParams.useNoiseMetrics))

baseParams.useNoiseMetrics.append('cluster_id')
run_predictor(frames[0][0]) # test classifier

mapping = {False: 'neural', True: 'noise'}
labels = [mapping[value] for value in frames[0][0]['is_noise']]

df = pd.DataFrame(data={'cluster_id' : frames[0][0]['cluster_id'], 'group': labels})
df.to_csv(fPath + r'\noiseClassifier.csv')


#%% load ground truth from Phy labels and retrain classifier

gTruth = merge_frames([fPath], ['group']) # get metrics from recording
a = pd.get_dummies(gTruth[0][0]['group'])
frames[0][0]['gTruth'] = a['noise']

df_auc = plot_roc_curve(fPath,frames,gTruth)
identify_best_estimator(frames[0][0], baseParams.useNoiseMetrics) # re-train classifier
run_predictor(frames[0][0]) # test classifier
  
    
#%%

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

                    
                    
        
                    

                
                
                
             

