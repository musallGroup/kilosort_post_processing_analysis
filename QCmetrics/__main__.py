

fPath = r'D:\SharedEphysData\Filemail.com - Spike data\sorting'

import os
import logging
import time
from argschema import ArgSchemaParser
import numpy as np
import pandas as pd
from utils import load_kilosort_data
from metrics import calculate_metrics
# import QMparams

os.chdir(fPath)
import params

    
QMparams = {'isi_threshold': 0.0015,
    'min_isi': 0.000166,
    'num_channels_to_compare': 13,
    'max_spikes_for_unit': 500,
    'max_spikes_for_nn': 10000,
    'n_neighbors': 4,
    'n_silhouette': 10000,
    'drift_metrics_min_spikes_per_interval': 10,
    'drift_metrics_interval_s': 60,
    'quality_metrics_output_file': fPath + r'\metrics.csv',
    'include_pc_metrics': True}

   

print('ecephys spike sorting: quality metrics module')

start = time.time()

print("Loading data...")

spike_times, spike_clusters, spike_templates, amplitudes, templates, channel_map, clusterIDs, cluster_quality, pc_features, pc_feature_ind = \
    load_kilosort_data(fPath, params.sample_rate, use_master_clock = False, include_pcs = True)

# pc_features = None
# pc_feature_ind = None

Qmetrics = calculate_metrics(spike_times, 
    spike_clusters, 
    spike_templates,
    amplitudes, 
    channel_map, 
    pc_features, 
    pc_feature_ind,
    QMparams)
    

output_file = fPath + r'\metrics.csv'
Qmetrics.to_csv(output_file)
