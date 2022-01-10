import pandas as pd
import os
from preprocessing import merge_frames, preprocess_frame, get_n_spikes
from run_predictor_sample import run_predictor
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
#%% Give your kilosort output here
test1 = r'D:\repo_data\imec0_ks2'

#%% Add Matlab function call 


#%%
# data preprocessing
kilosort_output_folder =[ test1]
frames = merge_frames(kilosort_output_folder)

frame_metrics=[]
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
      
    run_predictor(frames[0])
    
#%% Categorise into SUA, MUA and Noise

focus_df = frames[0].copy()
focus_mask = (focus_df['isi_viol'] <= 0.5) & (focus_df['amplitude_cutoff'] <= 0.1)
not_noisy_mask = (focus_df['is_noise'] ==0)
focus_df.loc[(not_noisy_mask & focus_mask), 'category'] = "SUA"
focus_df.loc[(not_noisy_mask & ~focus_mask), 'category'] = "MUA"
focus_df.loc[~not_noisy_mask, 'category'] = "noise"

