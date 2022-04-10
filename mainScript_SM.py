
# assign paths
fPath = r'\\naskampa\lts\invivo_ephys\SharedEphys\FromSylvia\SS088_2018-01-30_K1\SS088_2018-01-30_K1_g0\SS088_2018-01-30_K1_g0_imec0\SS088_2018-01-30_K1_g0_t0_imec\imec_ks2'
fPath = r'D:\SharedEphysData\FromSyliva\SS089_2018-02-07_K1'
# fPath = r'Z:\invivo_ephys\SharedEphys\FromGia\recording1'
# fPath = r'D:\test'
fPath = r'D:\SharedEphysData\FerminoData\KilosortOut\Kilosort2_2021-03-13_180605'
fPath = r'D:\SharedEphysData\FerminoData\KilosortOut\Kilosort2_2022-01-10_205701'


classifierPath = r'\\naskampa\lts\invivo_ephys\SharedEphys'
classifierName = r'\baseClassifier'


# get modules and params
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

import baseParams

# add paths to baseParams
baseParams.QMparams['quality_metrics_output_file'] = fPath + r'\metrics.csv'
baseParams.noiseParams['classifier_path'] = os.path.abspath(os.getcwd()) + r'\rf_classifier.pkl'
baseParams.noiseParams['noise_templates_output_file'] = fPath + r'\noiseModule.csv'


# baseParams.useNoiseMetrics['classifier_path'] = fPath + classifierName

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


#%% run noise decoder and save output
# make sure this includes cluster_id
baseParams.useNoiseMetrics.append('cluster_id')
baseParams.useNoiseMetrics = list(set(baseParams.useNoiseMetrics))

frames = merge_frames([fPath], baseParams.useNoiseMetrics) # get metrics from recording
run_predictor(frames[0][0]) # test classifier

mapping = {False: 'neural', True: 'noise'}
labels = [mapping[value] for value in frames[0][0]['is_noise']]

df = pd.DataFrame(data={'cluster_id' : frames[0][0]['cluster_id'], 'group': labels})
df.to_csv(fPath + r'\noiseClassifier.csv')









#%% load ground truth from Phy labels and retrain classifier

gTruth = merge_frames([fPath], ['group']) # get metrics from recording
a = pd.get_dummies(gTruth[0][0]['group'])
frames[0][0]['gTruth'] = a['noise']

identify_best_estimator(frames[0][0], baseParams.useNoiseMetrics) # re-train classifier
run_predictor(frames[0][0]) # test classifier

mapping = {False: 'neural', True: 'noise'}
labels = [mapping[value] for value in frames[0][0]['is_noise']]

df = pd.DataFrame(data={'cluster_id' : frames[0][0]['cluster_id'], 'group': labels})
df.to_csv(fPath + r'\noiseClassifier.csv')

    
#%%



                    
                    
        
                    

                
                
                
             

