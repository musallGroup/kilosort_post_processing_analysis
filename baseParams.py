# settings for quality metric module
QMparams = {'isi_threshold': 0.0015,
    'min_isi': 0.000166,
    'num_channels_to_compare': 13,
    'max_spikes_for_unit': 500,
    'max_spikes_for_nn': 10000,
    'n_neighbors': 4,
    'n_silhouette': 10000,
    'drift_metrics_min_spikes_per_interval': 10,
    'drift_metrics_interval_s': 60,
    'include_pc_metrics': True}


# settings for noise_template module
noiseParams = {
        'multiprocessing_worker_count': 10,
        'smoothed_template_amplitude_threshold': 0.2,
        'template_amplitude_threshold': 0.2,
        'smoothed_template_filter_width': 2,
        'min_spread_threshold': 2,
        'mid_spread_threshold': 16,
        'max_spread_threshold': 25,
        'channel_amplitude_thresh': 0.25,
        'peak_height_thresh': 0.2,
        'peak_prominence_thresh': 0.2,
        'peak_channel_range': 24,
        'peak_locs_std_thresh': 3.5,
        'min_temporal_peak_location': 10,
        'max_temporal_peak_location': 30,
        'template_shape_channel_range': 12,
        'wavelet_index': 2,
        'min_wavelet_peak_height': 0.0,
        'min_wavelet_peak_loc': 15,
        'max_wavelet_peak_loc': 25, 
        }


# quality metrics to be used for noise classifier
get_QMetrics = [ 'Amplitude',
                   'amplitude_cutoff',
                   'ContamPct',
                   'cumulative_drift',
                   'd_prime',
                   'firing_rate',
                   'isi_viol',
                   'isolation_distance',
                   'l_ratio',
                   'presence_ratio',
                   
                   'syncSpike_2', 
                   'syncSpace_2',
                   'farSyncSpike_2', 
                   'nearSyncSpike_2',
                   
                   'syncSpike_4', 
                   'syncSpace_4',
                   'farSyncSpike_4', 
                   'nearSyncSpike_4',
                   
                   'syncSpike_8', 
                   'syncSpace_8',
                   'farSyncSpike_8', 
                   'nearSyncSpike_8',
                   
           
                'nn_hit_rate',
                'nn_miss_rate',
                'max_drift',
                'silhouette_score',
                 
                
                ]

useNoiseMetrics = [ 
                    'syncSpike_2',
                     'syncSpike_4',
                   'firing_rate',
                   'presence_ratio',
                   'nn_hit_rate',
                   'nn_miss_rate',
                   'cumulative_drift'
 ]
    
