# targFolder = r'D:\SharedEphysData\FromSyliva'
# targFolder = r'D:\SharedEphysData\FerminoData\KilosortOut'
targFolder = r'D:\SharedEphysData\FromGaia'

# get modules and params
import os
import numpy as np
import pandas as pd
from preprocessing import merge_frames
from run_predictor_sample import test_predictor, identify_best_estimator
import baseParams
import inspect
import pickle

# path to default classifier
defClassifierPath = r'C:\Users\musall\Documents\GitHub\kilosort_post_processing_analysis'


# get recordings and keep the ones that have the cluster_group.tsv file
folderCheck = os.listdir(targFolder)
fPath = []
for i, path in enumerate(folderCheck):
    if os.path.isfile(os.path.join(targFolder,path,'cluster_group.tsv')):
        fPath.append(path)
        
        
# compute possible logical combinations for classifiers
allCombs = [str(np.zeros(len(fPath)))]*pow(2,len(fPath))
for i in range(pow(2,len(fPath))):
    allCombs[i] = np.zeros(len(fPath))
    temp = bin(i).replace("0b","")
    temp = temp[::-1]
    
    for x in range(len(temp)):
        if temp[x] == '1':
            allCombs[i][x] = '1'
            
            
allCombs = allCombs[1:] #dont use first and last case
            
#%% train all classifiers from scratch
if os.path.isdir(targFolder + r'\crossRefClassifiers') == False:
    os.mkdir(targFolder + r'\crossRefClassifiers')
    
    
for i in allCombs:
    
    cFiles = []
    for x in range(len(i)):
        if i[x] == 1:
            cFiles.append(os.path.join(targFolder, fPath[x]))
            
    # get metrics for classifier
    frames = merge_frames(cFiles, baseParams.useNoiseMetrics) 
    frames = pd.concat(frames[0],axis = 0)

    # get human labels from current recordings
    gTruth = merge_frames(cFiles, ['group'])
    gTruth = pd.concat(gTruth[0],axis = 0)
    a = pd.get_dummies(gTruth['group'])
    frames['gTruth'] = a['noise']    
    
    # train new classifier and save
    classifierPath = os.path.join(targFolder, 'crossRefClassifiers', 'Clf_' + str(i))
    identify_best_estimator(frames, baseParams.useNoiseMetrics, classifierPath) # re-train classifier


#%% test all classifiers
decoderPerf = pd.DataFrame(columns=['HitRate', 'FalseAlarmRate', 'trainedRecs'])

# test default classifier    
for xx in fPath:
    
    xx = os.path.join(targFolder, xx)    
    frames = merge_frames([xx], baseParams.useNoiseMetrics)
    frames = pd.concat(frames[0],axis = 0)

    gTruth = merge_frames([xx], ['group'])
    gTruth = pd.concat(gTruth[0],axis = 0)
    a = pd.get_dummies(gTruth['group'])
    frames['gTruth'] = a['noise']   
    
    test_predictor(frames, defClassifierPath); # test classifier

    #hit rate (percent recognized noise clusters)    
    cIdx = frames['gTruth'] == 1
    temp1 = np.round(np.sum(frames['is_noise'][cIdx]) / np.sum(cIdx),2)
 
    #false alarm rate (percent falsely labeled neural clusters)    
    cIdx = frames['gTruth'] == 0
    temp2 =  np.round(np.sum(frames['is_noise'][cIdx]) / np.sum(cIdx),2)
 
    # keep results for currrent recording / decoder
    decoderPerf = decoderPerf.append({'HitRate': temp1, 
                                      'FalseAlarmRate': temp2, 
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
        
        frames = merge_frames([xx], baseParams.useNoiseMetrics)
        frames = pd.concat(frames[0],axis = 0)
    
        gTruth = merge_frames([xx], ['group'])
        gTruth = pd.concat(gTruth[0],axis = 0)
        a = pd.get_dummies(gTruth['group'])
        frames['gTruth'] = a['noise']   
        
        print(cFiles)
        test_predictor(frames, classifierPath); # test classifier

        #hit rate (percent recognized noise clusters)    
        cIdx = frames['gTruth'] == 1
        temp1 = np.round(np.sum(frames['is_noise'][cIdx]) / np.sum(cIdx),2)
     
        #false alarm rate (percent falsely labeled neural clusters)    
        cIdx = frames['gTruth'] == 0
        temp2 =  np.round(np.sum(frames['is_noise'][cIdx]) / np.sum(cIdx),2)
     
        # keep results for currrent recording / decoder
        decoderPerf = decoderPerf.append({'HitRate': temp1, 
                                          'FalseAlarmRate': temp2, 
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
defFrames = merge_frames([defClassifierPath + r'\orgConfigs'], usedMetrics) 
defFrames = pd.concat(defFrames[0],axis = 0)


for i in allCombs:
    
    cFiles = []
    for x in range(len(i)):
        if i[x] == 1:
            cFiles.append(os.path.join(targFolder, fPath[x]))

    # get metrics for current classifier
    frames = merge_frames(cFiles, baseParams.useNoiseMetrics) 
    frames = pd.concat(frames[0],axis = 0)
    
    # get human labels from current recordings
    gTruth = merge_frames(cFiles, ['group'])
    gTruth = pd.concat(gTruth[0],axis = 0)
    a = pd.get_dummies(gTruth['group'])
    frames['gTruth'] = a['noise']
    
    # combined with existing labels
    frames = frames.append(defFrames)
    
    # train new classifier and save
    classifierPath = os.path.join(targFolder, 'crossRefClassifiers', 'mergeClf_' + str(i))
    identify_best_estimator(frames, baseParams.useNoiseMetrics, classifierPath) # re-train classifier


#%% test all merged classifiers
mergeDecoderPerf = pd.DataFrame(columns=['HitRate', 'FalseAlarmRate', 'trainedRecs'])

# test adjusted classifiers
for i, cComb in enumerate(allCombs):
    
    cFiles = []
    for x in range(len(cComb)):
        if cComb[x] == 0 or np.sum(cComb) == len(cComb):
            cFiles.append(os.path.join(targFolder, fPath[x]))
            
    classifierPath = os.path.join(targFolder, 'crossRefClassifiers', 'mergeClf_' + str(cComb))
        
    for xx in cFiles:
        
        frames = merge_frames([xx], baseParams.useNoiseMetrics)
        frames = pd.concat(frames[0],axis = 0)
    
        gTruth = merge_frames([xx], ['group'])
        gTruth = pd.concat(gTruth[0],axis = 0)
        a = pd.get_dummies(gTruth['group'])
        frames['gTruth'] = a['noise']   
        
        print(cFiles)
        test_predictor(frames, classifierPath); # test classifier

        #hit rate (percent recognized noise clusters)    
        cIdx = frames['gTruth'] == 1
        temp1 = np.round(np.sum(frames['is_noise'][cIdx]) / np.sum(cIdx),2)
     
        #false alarm rate (percent falsely labeled neural clusters)    
        cIdx = frames['gTruth'] == 0
        temp2 =  np.round(np.sum(frames['is_noise'][cIdx]) / np.sum(cIdx),2)
     
        # keep results for currrent recording / decoder
        mergeDecoderPerf = mergeDecoderPerf.append({'HitRate': temp1, 
                                          'FalseAlarmRate': temp2, 
                                          'trainedRecs': int(round(np.sum(cComb)))},
                                          ignore_index = True)
                         

print('Decoder performance on unseen datasets for folder: ' + targFolder)
print(decoderPerf.sort_values('trainedRecs'))

#show results
print('Merged decoder performance on unseen datasets for folder: ' + targFolder)
print(mergeDecoderPerf.sort_values('trainedRecs'))

