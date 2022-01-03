import pandas as pd
import os
from preprocessing import merge_frames, preprocess_frame, get_n_spikes
from run_predictor_sample import run_predictor

#%% Give your kilosort output here
test1 = r'D:\repo_data\imec0_ks2'

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
    
   
if __name__ == '__main__':
    #calling_predictor    
    run_predictor(frames[0])