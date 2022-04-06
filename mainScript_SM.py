

fPath = r'\\naskampa\lts\invivo_ephys\SharedEphys\FromSylvia\SS088_2018-01-30_K1\SS088_2018-01-30_K1_g0\SS088_2018-01-30_K1_g0_imec0\SS088_2018-01-30_K1_g0_t0_imec\imec_ks2'
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
from run_predictor_sample import run_predictor
import seaborn as sns
import matplotlib.pyplot as plt

import baseParams
baseParams.QMparams['quality_metrics_output_file'] = fPath + r'\metrics.csv'
baseParams.noiseParams['classifier_path'] = os.path.abspath(os.getcwd()) + r'\rf_classifier.pkl'
baseParams.noiseParams['noise_templates_output_file'] = fPath + r'\noiseModule.csv'

# import QMparams

os.chdir(fPath)
import params
  

start = time.time()

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




#%%
frame_metrics=[]

# files that can metrics of interest 
files = ['cluster_Amplitude.tsv', 'cluster_ContamPct.tsv', 'metrics.csv']

# find those files
def find(name,path):
    for root,dirs,files in os.walk(path):
        if name in files:
            return os.path.join(root,name)

#saving file paths
frame_metrics = []
for file in files:
    frame_metrics.append(find(file,fPath))

useNoiseMetrics = ['firing_rate',  'presence_ratio', 'isi_viol', 'amplitude_cutoff',
                   'isolation_distance', 'l_ratio' , 'd_prime','nn_hit_rate',
                   'nn_miss_rate', 'silhouette_score', 'max_drift',   'cumulative_drift' ,
                   'syncSpike_2', 'syncSpike_4', ]

#finding baseparames and saving them in dataframe
QM_features = pd.read_csv(frame_metrics[2],usecols = useNoiseMetrics)





                    
                    
        
                    

                
                
                
             






for path in kilosort_output_folder:
    frame_metrics.append(pd.read_csv(os.path.join(path, "metrics.csv"), index_col = 0))

n_spikes_per_clusters=[]
for path,metric in zip(kilosort_output_folder, frame_metrics):
    n_spikes_per_clusters.append( get_n_spikes(path, total_units=len(metric)))

for (key, frame), spike in zip(frames.items(), n_spikes_per_clusters):
    frame['n_spike'] = spike


for key, frame in frames.items():
    print(frame.columns)
    frames[key]=preprocess_frame(frame)
    
 #calling_predictor    
if __name__ == '__main__':
      
    run_predictor(frames[0][0])
    
    
#%% Categorise into SUA, MUA and Noise
focus_df = frames[0].copy()
focus_mask = (focus_df['isi_viol'] <= 0.5) & (focus_df['amplitude_cutoff'] <= 0.1)
not_noisy_mask = (focus_df['is_noise'] ==0)
focus_df.loc[(not_noisy_mask & focus_mask), 'category'] = "SUA"
focus_df.loc[(not_noisy_mask & ~focus_mask), 'category'] = "MUA"
focus_df.loc[~not_noisy_mask, 'category'] = "noise"






    