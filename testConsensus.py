# get modules and params
import os
import numpy as np
import pandas as pd
from preprocessing import merge_frames
from run_predictor_sample import run_predictor, test_predictor
import baseParams
import inspect
import pickle


classifierFolder = r'D:\SharedEphysData\testClassifiers'
folderCheck = os.listdir(classifierFolder)
classPath = []
for i, path in enumerate(folderCheck):
    if os.path.isfile(os.path.join(classifierFolder,path,'classifier.sav')):
        classPath.append(path)
        

#%% run classifiers on test recording with ground truth
testFolder = r'D:\SharedEphysData\FerminoData\TestData\Kilosort2_2021-09-21_120535'
classifierFolder = r'D:\SharedEphysData\FerminoData\testClassifiersRed'

folderCheck = os.listdir(classifierFolder)
classPath = []
for i, path in enumerate(folderCheck):
    if os.path.isfile(os.path.join(classifierFolder,path,'classifier.sav')):
        classPath.append(path)
        
noiseMetrics = baseParams.useNoiseMetrics
noiseMetrics.append('cluster_id')
noiseMetrics = list(set(noiseMetrics))


# test adjusted classifiers
isNoise = [0]*len(classPath)
decoderPerf = pd.DataFrame(columns=['TruePositive', 'FalsePositive', 'TrueNegative', 'FalseNegative', 'TotalPerf', 'classifier'])
for i, classifierPath in enumerate(classPath):
        
        
    frames = merge_frames([testFolder], noiseMetrics) # get metrics from recording
    frames = pd.concat(frames[0],axis = 0)
    gTruth = merge_frames([testFolder], ['group'])
    gTruth = pd.concat(gTruth[0],axis = 0)
    a = pd.get_dummies(gTruth['group'])
    frames['gTruth'] = a['noise']   
    
    # test classifier
    [isNoise[i], confusionMatrix] = test_predictor(frames, os.path.join(classifierFolder, classifierPath)); # test classifier

    # keep results for currrent recording / decoder
    decoderPerf = decoderPerf.append({'TruePositive': confusionMatrix[0], 
                                      'FalsePositive': confusionMatrix[1],
                                      'TrueNegative': confusionMatrix[2],
                                      'FalseNegative': confusionMatrix[3],
                                      'TotalPerf': confusionMatrix[4],
                                      'classifier': classifierPath},
                                      ignore_index = True)
    

# test performance with consensus vote at 50% threshold
frames = merge_frames([testFolder], noiseMetrics) # get metrics from recording
frames = pd.concat(frames[0],axis = 0)
allNoise = pd.concat(isNoise,axis = 1)
noiseCnt = np.sum(allNoise, axis = 1)
consensusNoise = noiseCnt / np.max(noiseCnt) > 0.5

#true positive (correctly recognized noise clusters)    
confusionMatrix[0] = np.sum((consensusNoise == 1) & (a['noise'] == 1)) / len(a['noise'])

#false alarm rate (percent falsely labeled neural clusters)    
confusionMatrix[1]  = np.sum((consensusNoise == 1) & (a['noise'] == 0)) / len(a['noise'])
 
#true negative (correctly recognized neural clusters)    
confusionMatrix[2]  = np.sum((consensusNoise == 0) & (a['noise'] == 0)) / len(a['noise'])

#false negatove (missed noise clusters)    
confusionMatrix[3]  = np.sum((consensusNoise == 0) & (a['noise'] == 1)) / len(a['noise'])
 
#total performance  
confusionMatrix[4] = (np.sum(consensusNoise == a['noise']) / len(consensusNoise))

decoderPerf = decoderPerf.append({'TruePositive': confusionMatrix[0], 
                                  'FalsePositive': confusionMatrix[1],
                                  'TrueNegative': confusionMatrix[2],
                                  'FalseNegative': confusionMatrix[3],
                                  'TotalPerf': confusionMatrix[4],
                                  'classifier': 'consensus'},
                                  ignore_index = True)
     
print('Decoder performance on unseen datasets for folder: ' + testFolder)
print(decoderPerf.sort_values('trainedRecs'))
        
        
 #%% run classifiers on test recording without ground truth
testFolder = r'D:\SharedEphysData\FerminoData\TestData\Kilosort2_2021-10-06_000401'
classifierFolder = r'D:\SharedEphysData\FerminoData\testClassifiersAll'

folderCheck = os.listdir(classifierFolder)
classPath = []
for i, path in enumerate(folderCheck):
    if os.path.isfile(os.path.join(classifierFolder,path,'classifier.sav')):
        classPath.append(path)
        
noiseMetrics = baseParams.defaultNoiseMetrics


# test adjusted classifiers
isNoise = [0]*len(classPath)
for i, classifierPath in enumerate(classPath):
        
        
    frames = merge_frames([testFolder], noiseMetrics) # get metrics from recording
    frames = pd.concat(frames[0],axis = 0)
    
    isNoise[i] = run_predictor(frames, os.path.join(classifierFolder, classifierPath)); # run classifier
    

# test performance with consensus vote at 50% threshold
noiseMetrics.append('cluster_id')
frames = merge_frames([testFolder], noiseMetrics) # get metrics from recording
frames = pd.concat(frames[0],axis = 0)
frames = frames[np.isnan(frames['firing_rate']) == 0]
allNoise = pd.concat(isNoise,axis = 1)
noiseCnt = np.sum(allNoise, axis = 1)
noiseProb = noiseCnt / np.max(noiseCnt) 
consensusNoise = noiseProb > 0.5

mapping = {False: 'neural', True: 'noise'}
labels = [mapping[value] for value in consensusNoise[0]]

df = pd.DataFrame(data={'cluster_id' : frames['cluster_id'], 'consensusNoise': labels, 'noiseProb': noiseProb})
df.to_csv(testFolder + r'\cluster_noiseConsensus.tsv', index=False, sep ='\t')

        