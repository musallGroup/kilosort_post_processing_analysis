

# targFolder = r'D:\SharedEphysData\FromSyliva'
targFolder = r'Y:\invivo_ephys\SharedEphys\FromFermino'
# targFolder = r'D:\SharedEphysData\FromGaia'

# get modules and params
import os
import numpy as np
import pandas as pd
from preprocessing import get_feature_columns,remove_miss_vals,get_roc_metrics
from run_predictor_sample import test_predictor, identify_best_estimator
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
auc_vals_frame = get_roc_metrics(trans_df)

keep_cols =  np.where((auc_vals_frame.roc_auc_val > 0.59) | (auc_vals_frame.roc_auc_val < 0.41))[0].tolist() # get metric with high AUC values 
metrics_names = trans_df.columns[keep_cols].values # added to baseParams.AUC_Metrics_fermino
baseParams.AUC_Metrics_fermino.append(metrics_names)

#metrics_names = np.append(metrics_names,'gTruth')
metrics_df = pd.DataFrame(trans_df, columns = metrics_names)

classifierPath = r'Y:\invivo_ephys\SharedEphys\FromFermino\new_clf'
identify_best_estimator(metrics_df, baseParams.useNoiseMetrics,classifierPath) # re-train classifier



#%%   
# compute possible logical combinations for classifiers
allCombs = [str(np.zeros(len(fPath)))]*pow(2,len(fPath))
for i in range(pow(2,len(fPath))):
    allCombs[i] = np.zeros(len(fPath))
    temp = bin(i).replace("0b","")
    temp = temp[::-1]
    
    for x in range(len(temp)):
        if temp[x] == '1':
            allCombs[i][x] = '1'
            
            
allCombs = allCombs[1:] #dont use first case
            
#%% train all classifiers from scratch
if os.path.isdir(targFolder + r'\crossRefClassifiers_AJ') == False:
    os.mkdir(targFolder + r'\crossRefClassifiers_AJ')
    
    
for i in allCombs:
    print ('====================', i)
    cFiles = []
    for x in range(len(i)):
        if i[x] == 1:
            cFiles.append(os.path.join(targFolder, fPath[x]))
            
    # get metrics for classifier
    #frames = get_feature_columns(cFiles, baseParams.useNoiseMetrics) 
    frames = get_feature_columns(cFiles,baseParams.AUC_Metrics_fermino)
    frames[0][0].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    frames = pd.concat(frames[0],axis = 0)

    # get human labels from current recordings
    gTruth = get_feature_columns(cFiles, ['group'])
    gTruth = pd.concat(gTruth[0],axis = 0)
    a = pd.get_dummies(gTruth['group'])
    frames['gTruth'] = a['noise']    
    
    # train new classifier and save
    classifierPath = os.path.join(targFolder, 'crossRefClassifiers_AJ', 'Clf_' + str(i))
    #identify_best_estimator(frames, baseParams.useNoiseMetrics, classifierPath) # re-train classifier
    identify_best_estimator(frames,baseParams.AUC_Metrics_fermino, classifierPath)

#%% test all classifiers
decoderPerf = pd.DataFrame(columns=['TruePositive', 'FalsePositive', 'TrueNegative', 'FalseNegative', 'TotalPerf', 'trainedRecs'])

# test default classifier    
for xx in fPath:
    
    xx = os.path.join(targFolder, xx)    
    frames = get_feature_columns([xx], baseParams.useNoiseMetrics)
    frames = pd.concat(frames[0],axis = 0)

    gTruth = get_feature_columns([xx], ['group'])
    gTruth = pd.concat(gTruth[0],axis = 0)
    a = pd.get_dummies(gTruth['group'])
    frames['gTruth'] = a['noise']   
    
    [isNoise, confusionMatrix] = test_predictor(frames, defClassifierPath); # test classifier

    # keep results for currrent recording / decoder
    decoderPerf = decoderPerf.append({'TruePositive': confusionMatrix[0], 
                                      'FalsePositive': confusionMatrix[1],
                                      'TrueNegative': confusionMatrix[2],
                                      'FalseNegative': confusionMatrix[3],
                                      'TotalPerf': confusionMatrix[4],
                                      'trainedRecs': 0},
                                      ignore_index = True)
    
# test adjusted classifiers
for i, cComb in enumerate(allCombs):
    
    cFiles = []
    for x in range(len(cComb)):
        if cComb[x] == 0 or np.sum(cComb) == len(cComb):
            cFiles.append(os.path.join(targFolder, fPath[x]))
            
    classifierPath = os.path.join(targFolder, 'crossRefClassifiers', 'Clf_' + str(cComb))
        
    for xx in cFiles:
        
        frames = get_feature_columns([xx], baseParams.useNoiseMetrics)
        frames = pd.concat(frames[0],axis = 0)
    
        gTruth = get_feature_columns([xx], ['group'])
        gTruth = pd.concat(gTruth[0],axis = 0)
        a = pd.get_dummies(gTruth['group'])
        frames['gTruth'] = a['noise']
        
        [isNoise, confusionMatrix] = test_predictor(frames, classifierPath); # test classifier

        # keep results for currrent recording / decoder
        decoderPerf = decoderPerf.append({'TruePositive': confusionMatrix[0], 
                                          'FalsePositive': confusionMatrix[1],
                                          'TrueNegative': confusionMatrix[2],
                                          'FalseNegative': confusionMatrix[3],
                                          'TotalPerf': confusionMatrix[4],
                                          'trainedRecs': int(round(np.sum(cComb)))},
                                          ignore_index = True)
                         
#show results
print('Decoder performance on unseen datasets for folder: ' + targFolder)
print(decoderPerf.sort_values('trainedRecs'))

#%% combine existing with new labels and see if default classifier becomes better
embeder = pickle.load(open(defClassifierPath + '\embedder.sav', 'rb'))
usedMetrics = embeder[1]
usedMetrics.append('gTruth')

# get default labels with ground truth
defFrames = get_feature_columns([defClassifierPath + r'\orgConfigs'], usedMetrics) 
defFrames = pd.concat(defFrames[0],axis = 0)


for i in allCombs:
    
    cFiles = []
    for x in range(len(i)):
        if i[x] == 1:
            cFiles.append(os.path.join(targFolder, fPath[x]))

    # get metrics for current classifier
    frames = get_feature_columns(cFiles, baseParams.useNoiseMetrics) 
    frames = pd.concat(frames[0],axis = 0)
    
    # get human labels from current recordings
    gTruth = get_feature_columns(cFiles, ['group'])
    gTruth = pd.concat(gTruth[0],axis = 0)
    a = pd.get_dummies(gTruth['group'])
    frames['gTruth'] = a['noise']
    
    # combined with existing labels
    frames = frames.append(defFrames)
    
    # train new classifier and save
    classifierPath = os.path.join(targFolder, 'crossRefClassifiers', 'mergeClf_' + str(i))
    identify_best_estimator(frames, baseParams.useNoiseMetrics, classifierPath) # re-train classifier


#%% test all merged classifiers
mergeDecoderPerf = pd.DataFrame(columns=['TruePositive', 'FalsePositive', 'TrueNegative', 'FalseNegative', 'TotalPerf', 'trainedRecs'])

# test adjusted classifiers
for i, cComb in enumerate(allCombs):
    
    cFiles = []
    for x in range(len(cComb)):
        if cComb[x] == 0 or np.sum(cComb) == len(cComb):
            cFiles.append(os.path.join(targFolder, fPath[x]))

    classifierPath = os.path.join(targFolder, 'crossRefClassifiers', 'mergeClf_' + str(cComb))
        
    for xx in cFiles:
        
        frames = get_feature_columns([xx], baseParams.useNoiseMetrics)
        frames = pd.concat(frames[0],axis = 0)
    
        gTruth = get_feature_columns([xx], ['group'])
        gTruth = pd.concat(gTruth[0],axis = 0)
        a = pd.get_dummies(gTruth['group'])
        frames['gTruth'] = a['noise']   
        
        print(cFiles)
        [isNoise, confusionMatrix] = test_predictor(frames, classifierPath); # test classifier

        # keep results for currrent recording / decoder
        mergeDecoderPerf = mergeDecoderPerf.append({'TruePositive': confusionMatrix[0], 
                                          'FalsePositive': confusionMatrix[1],
                                          'TrueNegative': confusionMatrix[2],
                                          'FalseNegative': confusionMatrix[3],
                                          'TotalPerf': confusionMatrix[4],
                                          'trainedRecs': int(round(np.sum(cComb)))},
                                          ignore_index = True)
                         

print('Decoder performance on unseen datasets for folder: ' + targFolder)
print(decoderPerf.sort_values('trainedRecs'))

#show results
print('Merged decoder performance on unseen datasets for folder: ' + targFolder)
print(mergeDecoderPerf.sort_values('trainedRecs'))

