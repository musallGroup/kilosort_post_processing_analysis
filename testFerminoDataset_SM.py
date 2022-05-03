

# targFolder = r'D:\SharedEphysData\FromSyliva'
# targFolder = r'Y:\invivo_ephys\SharedEphys\FromFermino'
# targFolder = r'D:\SharedEphysData\FromGaia'
targFolder = r'D:\SharedEphysData\FerminoData\KilosortOut'

# get modules and params
import os
import numpy as np
import pandas as pd
from preprocessing import get_feature_columns,remove_miss_vals,get_roc_metrics
from run_predictor_sample import test_predictor, identify_best_estimator, run_predictor
import baseParams
import inspect
import pickle

# path to default classifier
defClassifierPath = r'C:\Users\jain\Documents\GitHub\kilosort_post_processing_analysis'


# get recordings and keep the ones that have the cluster_group.tsv file
folderCheck = os.listdir(targFolder)
fPath = []
for i, path in enumerate(folderCheck):
    if os.path.isfile(os.path.join(targFolder,path,'cluster_group.tsv')):
        fPath.append(path)

    
#%% Add Combine all

if os.path.isdir(targFolder + r'\mergeCLF') == False:
    os.mkdir(targFolder + r'\mergeCLF')
    
trans_frames = {}
array_length = len(fPath)

for i in range(array_length):
    print("=====================", i)
    baseParams.useNoiseMetrics = list(set(baseParams.useNoiseMetrics))
    frames = get_feature_columns([os.path.join(targFolder,fPath[i])], baseParams.get_QMetrics) # get metrics from recording
    gTruth = get_feature_columns([os.path.join(targFolder,fPath[i])], ['group']) 
    
    frame = frames[0][0]
    gTruth = gTruth[0][0]
    print("=====================")
    
    trans_frames[i] = frame

    a = pd.get_dummies(gTruth['group'])
    trans_frames[i]['gTruth'] = a['noise']

df_list = [ v for k,v in trans_frames.items()] 
df = pd.concat(df_list ,axis=0)
trans_df = remove_miss_vals(df)
auc_vals_frame = get_roc_metrics(trans_df.drop('cluster_id', axis = 1))

keep_cols =  np.where((auc_vals_frame.roc_auc_val > 0.59) | (auc_vals_frame.roc_auc_val < 0.41))[0].tolist() # get metric with high AUC values 
metrics_names = trans_df.columns[keep_cols].values
metrics_df = pd.DataFrame(trans_df, columns = np.append(metrics_names,'gTruth'))

classifierPath = os.path.join(targFolder,'mergeCLF','mergeClf')
identify_best_estimator(metrics_df, metrics_names, classifierPath) # re-train classifier



#%% apply to test recording
testFolder = r'D:\SharedEphysData\FerminoData\TestData\Kilosort2_2021-10-06_000401'


frames = merge_frames([testFolder], np.append(baseParams.useNoiseMetrics,'cluster_id') ) # get metrics from recording
frames = pd.concat(frames[0],axis = 0)

isNoise = run_predictor(frames, classifierPath); # run classifier






