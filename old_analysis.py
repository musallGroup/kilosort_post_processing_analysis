# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 14:15:53 2021

@author: Jain 
"""
#cell 0 
import pyupset as pyu
from pickle import load
import glob
import os.path
import os

from matplotlib.ticker import StrMethodFormatter
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec

from scipy.optimize import curve_fit

import numpy as np

import pandas as pd

import sklearn # for the roc curv
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    adjusted_rand_score,
    adjusted_mutual_info_score
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import sklearn.cluster as cluster


import scikitplot as skplt
import skopt
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objs as go
import plotly.io as pio

import seaborn as sns
import umap
import hdbscan

from mlxtend.plotting import plot_decision_regions

#%% load path cell1
#cell 1
# 1. Load Optogenetics data 
glbdmxCAR_2217_09 = r"G:\final_outputs\OP\2217_20210909_GDMX_imec0_ks2"
locaar_2217_09 = r"G:\final_outputs\OP\2217_20210909_LOC_imec0_ks2"
rescale_glbdmxCAR_2217_09 = r"G:\final_outputs\OP\2217_20210909_RGDMX_imec0_ks2"

glbdmxCAR_2217_11 = r"G:\final_outputs\OP\2217_20210911__GDMX_imec0_ks2"
locaar_2217_11 = r"G:\final_outputs\OP\2217_20210911_LOC_imec0_ks2"
rescale_glbdmxCAR_2217_11 = r"G:\final_outputs\OP\2217_20210911_RGDMX_imec0_ks2"
#=================================================================================================================================
glbdmxCAR_2009_23 = r"G:\final_outputs\WF\2009_20210223_GDMX_imec0_ks2"
locaar_2009_23 = r"G:\final_outputs\WF\2009_20210223_LOC_imec0_ks2"
rescale_glbdmxCAR_2009_23 = r"G:\final_outputs\WF\2009_20210223_RGDMX_imec0_ks2"

glbdmxCAR_2009_26a = r"G:\final_outputs\WF\2009_20210226a_GDMX_imec0_ks2"
locaar_2009_26a = r"G:\final_outputs\WF\2009_20210226a_LOC_imec0_ks2"
rescale_glbdmxCAR_2009_26a = r"G:\final_outputs\WF\2009_20210226a_RGDMX_imec0_ks2"

NP_2241 = r'G:\final_outputs\NP\2241_210602_imec0_ks2'
NP_RD10 = r'G:\final_outputs\NP\RD10_2220_20210722_imec0_ks2'


#unseen =r'G:\final_outputs\RD10_2128_20210119_g0_t0_imec0\imec0_ks2'
#unseen = r'G:\Outputs\catgt_GN09_20210505_g0\GN09_20210505_g0_imec0\imec0_ks2'
test1 = r'Y:\invivo_ephys\Neuropixels\RD10_2129_20210112\RD10_2129_20210112_g0\RD10_2129_20210112_g0_imec0\RD10_2129_20210112_g0_t0_imec0\imec0_ks2'
test2 = r'Y:\invivo_ephys\Neuropixels\RD10_2130_20210119\RD10_2130_20210119_g0\RD10_2130_20210119_g0_imec0\RD10_2130_20210119_g0_t0_imec0\imec0_ks2'
test3 = r'Y:\invivo_ephys\Neuropixels\RD10_2130_20210120\RD10_2130_20210120_g0\RD10_2130_20210120_g0_imec0\RD10_2130_20210120_g0_t0_imec0\imec0_ks2'
test4 = r'Y:\invivo_ephys\Neuropixels\RD10_2130_20210121\RD10_2130_20210121_g0\RD10_2130_20210121_g0_imec0\RD10_2130_20210121_g0_t0_imec0\imec0_ks2'
test5 = r'Y:\invivo_ephys\Neuropixels\RD10_2220_20210721\RD10_2220_20210721_g0\RD10_2220_20210721_g0_imec0\imec0_ks2'
test6 = r'Y:\invivo_ephys\Neuropixels\RD10_2220_20210722\RD10_2220_20210722_g0\RD10_2220_20210722_g0_imec0\imec0_ks2'
test7 = r'Y:\invivo_ephys\Neuropixels\RD10_2220_20210723\RD10_2220_20210723_g0\RD10_2220_20210723_g0_imec0\imec0_ks2'
#===================================================================================================================================
kilosort_output_folder =[ glbdmxCAR_2217_09 ,
           locaar_2217_09,
           rescale_glbdmxCAR_2217_09,
           
           glbdmxCAR_2217_11,
           locaar_2217_11,
           rescale_glbdmxCAR_2217_11,
    
            glbdmxCAR_2009_23 ,
            locaar_2009_23,
            rescale_glbdmxCAR_2009_23,
            
           glbdmxCAR_2009_26a,
           locaar_2009_26a,
           rescale_glbdmxCAR_2009_26a,
           
           NP_2241,
           NP_RD10,
           #unseen
           test1,
           test2,
           test3,
           test4,test5,test6,test7
           ]

#%% #%% 
#nr
#focus_df.loc[focus_df['isi_viol'].between(-np.inf,0.5,inclusive=True)  &
#            focus_df['amplitude_cutoff'].between(-np.inf,0.1,inclusive=True)
#           ,'category']= "SUA"


#%%

data = [ pd.Series([347,625]),
         pd.Series([389,638]),
        pd.Series([305,511]),
        #pd.Series([523,495,295])
    ]

headers = ['global','local','rescale']

df = pd.concat(data, axis=1, keys=headers)

#ax = sns.violinplot(data = df);

ax = sns.boxplot(data = df);
plt.title("Neuropixels with Optogenetics", fontsize=30)
plt.ylabel("Total clusters", fontsize= 20)
plt.xlabel("Different Pipelines", fontsize=20)



#%% cell2
#3. Combine csv and tsv files 
def merge_frames(paths_all):
    frames = {}
    for i, path in enumerate(paths_all): #parse with indexing
        frames[i] = []
        for file in os.listdir(path):
            if ".tsv" in file:
                frame = pd.read_csv(os.path.join(path, file), sep='\t')
                frames[i].append(frame)
            elif ".csv" in file and 'merged_data_frames' not in file:
                frame = pd.read_csv(os.path.join(path, file))
                frames[i].append(frame)
                
    # 4. merge the dataframes
    for key in frames:
        print(key, '->', frames[key])
        frame = frames[key][0]
        for merge_frame in frames[key][1:]:
            frame = frame.merge(merge_frame, on="cluster_id")
        frames[key] = frame
    
    # 5. write into a file
    for i, path in enumerate(paths_all):
        frames[i].to_csv(os.path.join(path, 'merged_data_frames.csv'))
    return frames


# 6. cleaned datasets : all non-useful features are dropped andchange categorical into integer

def preprocess_frame(frame):
    enc = LabelEncoder()
    print(frame.columns)
    #frame['label'] = frame['label'].fillna(-1)
    frame['d_prime']= frame['d_prime'].fillna(frame['d_prime'].median())
    frame['Amplitude']= frame['Amplitude'].fillna(frame['Amplitude'].median())
    frame['nn_hit_rate']= frame['nn_hit_rate'].fillna(frame['nn_hit_rate'].median())
    frame['nn_miss_rate']= frame['nn_miss_rate'].fillna(frame['nn_miss_rate'].median())
    frame['isolation_distance']= frame['isolation_distance'].fillna(frame['isolation_distance'].median())
    frame['ContamPct']=frame['ContamPct'].replace([np.inf, -np.inf], 100)
    #frame['label']=frame['label'].mask(frame['label']<0, 1)
    #frame['certanity'] = frame['certanity'].fillna(1)
    
    
    if 'label' in frame.columns:
        frame['label'] = frame['label'].fillna(-1)
        frame['label']=frame['label'].mask(frame['label']<0, 1)
    if 'certanity' in frame.columns:
        frame['certanity']= frame['certanity'].fillna(1)
    if 'Unnamed: 0' in frame.columns:
        frame.drop(['Unnamed: 0'],axis = 1,inplace=True)
    if 'epoch_name' in frame.columns:
        frame.drop(['epoch_name'],axis = 1,inplace=True)
    if 'silhouette_score' in frame.columns:
        frame.drop(['silhouette_score'],axis = 1,inplace=True)
    if 'l_ratio' in frame.columns:
        frame.drop(['l_ratio'],axis = 1,inplace=True)
    if 'max_drift' in frame.columns:
        frame.drop(['max_drift'],axis = 1,inplace=True)
    if 'Var1' in frame.columns:
        frame.drop(['Var1'],axis = 1,inplace=True)
        
    frame['group'] = enc.fit_transform(frame['group'])
    print(enc.inverse_transform( [0, 1,0]))
    frame['KSLabel']= enc.fit_transform(frame['KSLabel'])
    print(enc.inverse_transform([0, 1]))
    
    
    return(frame)

#%% function for n_spikes cell3
def get_n_spikes(kilosort_output_folder, total_units):
    
    spike_clusters = np.load(os.path.join(kilosort_output_folder, 'spike_clusters.npy'))
    
    unit_n_spikes = np.zeros(total_units)
    for cluster_id in np.unique(spike_clusters):
        unit_n_spikes[cluster_id] = np.sum(spike_clusters == cluster_id)
        
    return unit_n_spikes
#%% check labels cell4

def get_groundTruth(frame):
    gt = 0
    df_i = frame.set_index('cluster_id')
    gTruth=[]
    for index, row in df_i.iterrows():
        label =   row['label'] 
        certanity =  row['certanity']
        
        if label == 0 and certanity == 1:
            gt = 0 
            gTruth.append(gt)
            
        elif label == 0 and certanity == 0:
            gt = 1
            gTruth.append(gt)
            
        elif label == 1 and certanity == 0: 
            gt = 1
            gTruth.append(gt)
            
        elif label == 1 and certanity == 1:
            gt = 1
            gTruth.append(gt)
        else :
            print(index)
            print(key)
            print(label)
            print(certanity)
            
    df_i['gTruth'] =gTruth
    return(df_i)
#%% function call cell5

frames = merge_frames(kilosort_output_folder)
#%% cell6
frame_metrics=[]
for path in kilosort_output_folder:
    frame_metrics.append(pd.read_csv(os.path.join(path, "metrics.csv"), index_col = 0))
#%%cell7
n_spikes_per_clusters=[]
for path,metric in zip(kilosort_output_folder, frame_metrics):
    n_spikes_per_clusters.append( get_n_spikes(path, total_units=len(metric)))
#%%cell8
for (key, frame), spike in zip(frames.items(), n_spikes_per_clusters):
    frame['n_spike'] = spike

#%% function call 2 cell9
for key, frame in frames.items():
    print(frame.columns)
    frames[key]=preprocess_frame(frame)


 #%% cwll10
for key, frame in frames.items():
    
    frames[key]=get_groundTruth(frame)
    # print(frames[key].columns)
#%%
opto =  pd.concat([frames[0],frames[1],frames[2],
                   frames[3],frames[4],frames[5]],ignore_index=True)   
#print("opto", opto.size)  
opto.to_csv(r'F:\Neuropixel_TestData\pipeline_analysis\NeuropixelsAnalysis\metric_neuron_opto.csv')
#%%
wf =  pd.concat([frames[6],frames[7],frames[8],
                   frames[9],frames[10],frames[11]],ignore_index=True)   
#print("wf", wf.size)
wf.to_csv(r'F:\Neuropixel_TestData\pipeline_analysis\NeuropixelsAnalysis\metric_neuron_wf.csv') 
#%%
np_data = pd.concat([frames[12],frames[13]], ignore_index = True)
#print("np_data", np_data.size) 

np_data.to_csv(r'F:\Neuropixel_TestData\pipeline_analysis\NeuropixelsAnalysis\metric_neuron_np.csv')

#%%
# gdmx = pd.concat([frames[0],frames[3],frames[6],frames[9]],ignore_index=True) 

# loccar = pd.concat([frames[1],frames[4],frames[7],frames[10]],ignore_index=True) 

# rescale_gdmx = pd.concat([frames[2],frames[5],frames[8],frames[11]],ignore_index=True) 

# np_data = pd.concat([frames[12],frames[13]], ignore_index = True)

#test = pd.concat([frames[14],frames[15],frames[16]])
#%%
unseen_data = frames[14].head(301)
unseen_data.to_csv(r'F:\Neuropixel_TestData\pipeline_analysis\NeuropixelsAnalysis\unseen_data_labelled.csv')
#%%
def plot_confusion_matrix(frame):
    tp =0; fp =0; fn =0; tn =0
    
    for index, row in frame.iterrows():
        actual =   row['group'] 
        prediction_class =  row['label']
    
        
        if prediction_class == 1 and actual == 1:
            tp = tp + 1
        elif actual == 1 and prediction_class == 0:
            fn = fn + 1
        elif actual == 0 and prediction_class == 1: 
            fp = fp + 1
        elif actual == 0 and prediction_class == 0:
            tn = tn + 1
    
    cf_sum = tn+fp+fn+tp
    print(tn/cf_sum, fp/cf_sum, fn/cf_sum, tp/cf_sum)
    A = np.array([tn/cf_sum, fp/cf_sum, fn/cf_sum, tp/cf_sum])
    B = np.reshape(A, (-1, 2))
    
    with sns.axes_style('white'):
        res = sns.heatmap(B,
                    cbar=False,
                
                    annot=True,
                   
                    cmap=ListedColormap(['white']),
                    linewidths=0.5,annot_kws={"fontsize":40}, linecolor='gray',vmin=0, vmax=2)
    for _, spine in res.spines.items():
        spine.set_visible(True)
    #sns.heatmap(B,annot_kws={"fontsize":8},cbar=False,fmt='g',cmap= white,linewidths=0.5)
    #sns.heatmap(B, annot=True)
    plt.title("label vs noise_module",fontsize=30)
    plt.xlabel("Ground Truth",fontsize=30)
    plt.ylabel("Predicted Label",fontsize=30)
    
    print("Precion is ",tp / (tp + fp))

plot_confusion_matrix()

        
#%% stacking all data frames 
wf_np_bhv = pd.concat([wf,np_data,unseen_data],ignore_index=True)
wf_np_bhv.to_csv(r'F:\Neuropixel_TestData\metric_neuron_wf_np_bhv.csv')

opto_np_data_bhv = pd.concat([opto,np_data,unseen_data],ignore_index=True)
opto_np_data_bhv.to_csv(r'F:\Neuropixel_TestData\metric_neuron_opto_np_data_bhv.csv')

opto_wf_bhv = pd.concat([opto ,wf ,unseen_data],ignore_index=True)
opto_wf_bhv.to_csv(r'F:\Neuropixel_TestData\metric_neuron_opto_wf_bhv.csv')

opto_wf_np_data = pd.concat([opto ,wf ,np_data],ignore_index=True)
opto_wf_np_data.to_csv(r'F:\Neuropixel_TestData\metric_neuron_opto_wf_np_data.csv')
#%%
#testa = frames[14]
#testb= frames[15]
#testc = frames[16]
#dataset.to_csv(r'F:\Neuropixel_TestData\metric_neuron_train.csv')
unseen_data.to_csv(r'F:\Neuropixel_TestData\pipeline_analysis\NeuropixelsAnalysis\unseen_data.csv')
#test.to_csv(r'F:\Neuropixel_TestData\metric_neuron_test.csv')
#testa.to_csv(r'F:\Neuropixel_TestData\metric_neuron_testa.csv')
#testb.to_csv(r'F:\Neuropixel_TestData\metric_neuron_testb.csv')
#testc.to_csv(r'F:\Neuropixel_TestData\metric_neuron_testc.csv')
#%%
dataset_new = pd.concat([opto,wf,np_data], ignore_index = True)
dataset_new.to_csv(r'F:\Neuropixel_TestData\pipeline_analysis\NeuropixelsAnalysis\dataset.csv')
#%%

#===========UMAP=============#

# roc_dataframes = dataset[['syncSpike_2',
#                           'firing_rate',
#                           'presence_ratio',
#                           'cumulative_drift',
#                             'nn_hit_rate',
#                             'nn_miss_rate',
#                             'LABEL']].copy()
#%%
#call preprocessed dataset 
dataset_new = pd.read_csv(r'F:\Neuropixel_TestData\metric_neuron_train.csv')
print(dataset_new.columns)

#%% Import Libraries
#try PCA 
#%% ROC Curves
from sklearn import metrics
plt.figure(figsize=(14,10),dpi=640)
syncSpike_16=[]

for key, frame in frames.items():
    actual1= frames[key]['gTruth']
    
    prediction1= frames[key]['syncSpike_16']
    
    fpr, tpr, thresholds = metrics.roc_curve(actual1,prediction1)
    roc_auc = metrics.auc(fpr, tpr)
    
    syncSpike_16.append(roc_auc)
    # plt.plot(fpr, tpr,label="AUC prediction:{0}".format(roc_auc),color='red', linewidth=2)
    # plt.title("syncSpike_2",fontsize=40)
    # plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    # plt.legend(loc = 'lower right',fontsize=40)
    # plt.plot([0, 1], [0, 1],'r--')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.ylabel('True Positive Rate',fontsize=30)
    # plt.xlabel('False Positive Rate',fontsize=30)
    # plt.show()
#%%%
headers= ('Amplitude', 'ContamPct', 
       'firing_rate', 'presence_ratio', 'isi_viol', 'amplitude_cutoff',
       'isolation_distance', 'd_prime', 'nn_hit_rate', 'nn_miss_rate',
       'cumulative_drift', 'syncSpike_2', 'syncSpike_4', 'syncSpike_8',
       'syncSpike_16', 'n_spike')
daa = pd.concat([pd.Series(Amplitude),
                 pd.Series(ContamPct),
                  pd.Series(firing_rate),
                  pd.Series(presence_ratio),
                 pd.Series(isi_viol),
                 pd.Series(amplitude_cutoff),
                 pd.Series(isolation_distance),
                 pd.Series(d_prime),
                 pd.Series(nn_hit_rate),
                 pd.Series(nn_miss_rate),
                 pd.Series(cumulative_drift),
                 pd.Series(syncSpike_2),
                 pd.Series(syncSpike_4),
                 pd.Series(syncSpike_8),
                  pd.Series(syncSpike_16),
                  pd.Series(n_spike),
                 ],axis=1,keys=headers)
#%%\
daa = pd.read_csv(r'F:\Neuropixel_TestData\AUC_Values.csv')
print(daa.columns)
del daa['Unnamed: 0']
#%%
sorted_index = daa.mean().sort_values().index
df_sorted=daa[sorted_index]
#%%
fig,ax = plt.subplots()

plt.xlabel("AUC Values")
plt.axhline(7.5, color ='r', ls = '--') 
sns.boxplot(data=daa, orient='h') 

sns.stripplot(data=daa ,linewidth=0,orient='h',color='.4',size=4)
#%%
fig,ax = plt.subplots()
sorted_index = daa.mean().sort_values().index
df_sorted=daa[sorted_index]
plt.axhline(7.5, color ='r', ls = '--') 
sns.boxplot(data=df_sorted, orient='h') 
#sns.stripplot(data=sorted_index ,linewidth=0,orient='h',color='.4',size=4)

plt.title("AUC-ROC Curve Values for different Quality Metrics")
plt.xlabel("AUC Values")
plt.title("AUC-ROC Curve Values for different Quality Metrics")
fig=plt.figure(figsize=(9,11))
fig.savefig(r"F:\Neuropixel_TestData\AUC_Values.eps", format='eps')





daa.to_csv(r'F:\Neuropixel_TestData\daa.csv')


#%%%
daa.to_csv(r'F:\Neuropixel_TestData\AUC_Values.csv')
#%%
AUC_vals = daa.copy(deep=True)
#%%
for column in AUC_vals.columns:
    AUC_vals[column] =AUC_vals[column]-0.5
    AUC_vals[column]= AUC_vals[column].abs()
#%%
sorted_index = AUC_vals.mean().sort_values().index
df_sorted=AUC_vals[sorted_index]
#%%
fig,ax = plt.subplots()

plt.xlabel("AUC Values")
plt.axhline(7.5, color ='r', ls = '--') 
sns.boxplot(data=df_sorted, orient='h') 
plt.title("AUC-ROC Curve Values for different Quality Metrics")

#fig.savefig(r"F:\Neuropixel_TestData\AUC_Values.eps", format='eps')

sns.stripplot(data=daa,linewidth=0,orient='h',color='.4',size=4)
#%%
sns.boxplot(data=daa, orient='h') 
plt.xlabel("daa")
#%% Read training dataset
dataset_new = pd.read_csv(r'F:\Neuropixel_TestData\pipeline_analysis\NeuropixelsAnalysis\dataset\dataset.csv')
print(dataset_new.columns)
print(dataset_new.shape)
  
#%% Call Data




#============================Classifier========================================
idxs = list(dataset_new.index.values)
metric_data = dataset_new[[    'n_spike', 
                               'syncSpike_2',
                                'syncSpike_4',
                              'firing_rate',
                              'presence_ratio',
                              'nn_hit_rate',
                              'nn_miss_rate',
                              'cumulative_drift']].values

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(metric_data,
                                                                                 dataset_new['gTruth'].values,
                                                                                 idxs, 
                                                                                 shuffle=True,
                                                                                 random_state=42)


#%% Tceate pipeline using  unsupervised feature preprocessing 

pipe = Pipeline([('impute', SimpleImputer(missing_values=np.nan, strategy='median')),
                 ('scaler', StandardScaler())])

#call the tranformer in data 
X_train_transform = pipe.fit_transform(X_train)

#%% Supervised UMAP 

supervised_embedder = umap.UMAP(
    min_dist=0.0, 
    n_neighbors=10, 
    n_components=2, # dimensions 
    random_state=42
    )
X_train_final = supervised_embedder.fit_transform(X_train_transform, y=y_train)
#X_train_final : fit and transformed with emedder
# fig, ax = plt.subplots()
# #fig.set_size_inches(18.5, 10.5)

# plt.scatter(X_train_final[:, 0], X_train_final[:, 1], s=20, c=y_train, cmap='Spectral', alpha=1.0)
# plt.setp(ax, xticks=[], yticks=[])

# cbar = plt.colorbar(boundaries=np.arange(3)-0.5)
# cbar.set_ticks(np.arange(2))

# plt.title(' Embedded via UMAP using Labels');

# plt.show()

#%% Metric Learning 
X_test_trans = pipe.transform(X_test) #to impute and normalise
X_test_final = supervised_embedder.transform(X_test_trans)

# fig, ax = plt.subplots()
# #fig.set_size_inches(18.5, 10.5)

# plt.scatter(X_test_final[:, 0], X_test_final[:, 1], s=20, c=y_test, cmap='Spectral', alpha=1.0)
# plt.setp(ax, xticks=[], yticks=[])

# cbar = plt.colorbar(boundaries=np.arange(3)-0.5)
# cbar.set_ticks(np.arange(2))

# plt.title('Embedded via UMAP predicting Labels');

# plt.show()

#%% Classifier based bayesian xsearch
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neural_network import MLPClassifier 

from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
import lightgbm 
from lightgbm import LGBMRegressor, LGBMClassifier, Booster
from skopt import BayesSearchCV

# classifiers_ = [
#             "adaboost", "decision_tree", "extra_trees",
#             "gradient_boosting", "k_nearest_neighbors",
#             "libsvm_svc", "mlp", "random_forest",
#             "gaussian_nb",
#         ]

#data_dmatrix = xgb.DMatrix(data=X,label=y)
seed=1
# CASH (Combined Algorithm Selection and Hyperparameter optimisation)
models = [
            'ADB',
            'GBC',
            'RFC',
            'KNC',
            'SVC',
            'MLP',
            'ExtraTreesClassifier',
            'xgboost',
           'lightgbm'
         ]
clfs = [
        AdaBoostClassifier(random_state=seed),
        GradientBoostingClassifier(random_state=seed),
        RandomForestClassifier(random_state=seed,n_jobs=-1),
        KNeighborsClassifier(n_jobs=-1),
        SVC(random_state=seed,probability=True),
        MLPClassifier(random_state=seed, max_iter=300,hidden_layer_sizes= (50, 100)),
        ExtraTreesClassifier(n_estimators=100, random_state=0),
        XGBClassifier(n_estimators=100, random_state=0),
        LGBMClassifier(random_state=0)
        ]
params = {
            models[0]:{'learning_rate':[1,2], 
                       'n_estimators':[50,100],
                       'algorithm':['SAMME','SAMME.R']
                       },#AdaB
    
            models[1]:{'learning_rate':[0.05,0.1],
                       'n_estimators':[100,150], 
                       'max_depth':[2,4],
                       'min_samples_split':[2,4],
                       'min_samples_leaf': [2,4]
                       }, #GBC
    
            models[2]:{'n_estimators':[100,150],
                       'criterion':['gini','entropy'],
                       'min_samples_split':[2,4],
                       'min_samples_leaf': [2,4]
                       }, #RFC
    
            models[3]:{'n_neighbors':[20,50], 
                       'weights':['distance','uniform'],
                       'leaf_size':[30]
                       }, #KNN
    
            models[4]: {'C':[0.5,2.5],
                       'kernel':['sigmoid','linear','poly','rbf']
                       }, #SVC
            
            models[5]: {
                         'activation': ['tanh', 'relu'],
                         'solver': ['sgd', 'adam'],
                         'alpha': [0.0001, 0.05],
                         'learning_rate': ['constant','adaptive']
                         }, #MLP
    
            models[6]:{'criterion':['gini', 'entropy'],  
                       'class_weight':['balanced', 'balanced_subsample']
                       }, #extratrees
    
             models[7]:{'max_depth':[2,4], 
                       'eta': [0.2,0.5], 
                       'sampling_method':['uniform','gradient_based'],
                       'grow_policy':['depthwise', 'lossguide']
                      }, #xgboost
                        
    
            models[8]:{'learning_rate':[0.05,0.15],
                       'n_estimators': [100,150]} #lightgbm
    
         }
test_scores = []
val_scores =[]
search_objects=[]
model_estimator = []  

for name, estimator in zip(models,clfs):
    print(name)
    
   
    clf = BayesSearchCV(estimator, params[name], scoring='accuracy', refit='True', n_jobs=-1, n_iter=20,cv=5)
    clf.fit(X_train_final, y_train)   # X_train_final, y_train X is train samples and y is the corresponding labels
   
    
    print("best params: " + str(clf.best_params_))
    print("best scores: " + str(clf.best_score_))
    print("best estimator " +  str(clf.best_estimator_))
    model_estimator.append(clf.best_estimator_)
    
    fig = plot_decision_regions(X=X_train_final, y=y_train, clf=clf, legend=2)
    plt.title(f'Decison boundary of {name} on clusters');
    plt.show()
    
    val_scores.append(clf.best_score_)
    #acc = accuracy_score(y_test, clf.predict(X_test_trans))
    #print("Accuracy: {:.4%}".format(acc))
    search_objects.append(clf)
    
    
    clfscore=clf.score(X_test_final, y_test)
    test_scores.append(clfscore)
#%%
max_value = max(val_scores)#Return the max value of the list
max_index = val_scores.index(max_value)
best_classifier = model_estimator[max_index]    
print("best_classifier is ", best_classifier)

#%%

#from sklearn.ensemble import GradientBoostingClassifier
#clf = RandomForestClassifier(max_depth=2, random_state=0)
# clf = GradientBoostingClassifier(learning_rate=0.09966945482981998, max_depth=4,
#                            min_samples_leaf=4, min_samples_split=3,
#                            n_estimators=148, random_state=1)
clf =  KNeighborsClassifier(n_jobs=-1, n_neighbors=40, weights='distance')
 
clf.fit(X_train_final,y_train)

score = clf.score(X_test_final, y_test) # report 
fig = plot_decision_regions(X=X_train_final, y=y_train, clf=clf, legend=2)
plt.title('Decison boundary')
plt.show()
print(score)
#%%
from sklearn.metrics import accuracy_score

# y_pred1 = np_data['group'].loc[indices_test]
# y_true1 = np_data['gTruth'].loc[indices_test]
y_pred1 = np_data['group']
y_true1 = np_data['gTruth']
print(accuracy_score(y_true1,y_pred1))
#%%

# preprocess_frame()
X_unseen = test_data[[          'n_spike', 
                               'syncSpike_2',
                               'syncSpike_4',
                              'firing_rate',
                              'presence_ratio',
                              'nn_hit_rate',
                              'nn_miss_rate',
                              'cumulative_drift' ]].values


X_test_trans = pipe.transform(X_unseen) #to impute and normalise
X_test_final = supervised_embedder.transform(X_test_trans)
y_pred = clf.predict(X_test_final) #for every row 
test_data['is_noise']=y_pred

#%%
def run_predictor(best_classifier_config,test_dataframe):
    
    X_unseen = test_dataframe[[  'n_spike', 
                             'syncSpike_2',
                             'syncSpike_4',
                              'firing_rate',
                              'presence_ratio',
                              'nn_hit_rate',
                              'nn_miss_rate',
                              'cumulative_drift' ]].values


    X_test_trans = pipe.transform(X_unseen) #to impute and normalise
    X_test_final = supervised_embedder.transform(X_test_trans)
   
    y_pred = best_classifier.predict(X_test_final) #for every row 
    test_dataframe['is_noise']=y_pred
    
    return test_dataframe['is_noise']
#%%    
run_predictor(best_classifier_config=best_classifier, test_dataframe=frames[14] )
#%%
test_data.to_csv(r'F:\Neuropixel_TestData\Output_2128_2021_0119.csv')

#%% MLP classifier
from sklearn.neural_network import MLPClassifier

clf_mlp = MLPClassifier(random_state=1,activation = 'tanh', 
                        alpha = 0.0001,
                        hidden_layer_sizes = (50, 100), 
                        learning_rate = 'constant',
                        solver = 'adam' , max_iter=300).fit(X_train_final,y_train)

score = clf_mlp.score(X_test_final, y_test)
fig = plot_decision_regions(X=X_train_final, y=y_train, clf=clf_mlp, legend=2)
plt.title('Decison boundary of MLP on 4775 clusters');
print(score
#%%
import optuna # HP optimization package

import sklearn.datasets
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier 
import sklearn.model_selection
import sklearn.svm


class Objective(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __call__(self, trial):

        classifier_name = trial.suggest_categorical("classifier",[ "AdaBoostClassifier",
                                                                   "GradientBoostingClassifier",
                                                                   "RandomForestClassifier",
                                                                    "SVC", 
                                                                    "MLPClassifier"
                                                                  ])

        if classifier_name == "AdaBoostClassifier":
            ada_learning_rate = trial.suggest_float("ada_learning_rate",0.1,1.0, log=True)
            ada_n_eastimators = trial.suggest_int("ada_n_eastimators", 50, 150, log=True)
            
            classifier_obj = AdaBoostClassifier(learning_rate = ada_learning_rate,
                                                n_estimators = ada_n_eastimators)
            
            
        elif classifier_name == "GradientBoostingClassifier":
            gb_learning_rate = trial.suggest_float("gb_learning_rate",0.01, 0.1, log=True)
            gb_n_eastimators = trial.suggest_int("gb_n_eastimators", 50, 100, log=True)
            gb_criteion =  trial.suggest_categorical("criteion",["friedman_mse", 
                                                                 "squared_error",
                                                                 "mse", 
                                                                 "mae"])
            gb_max_depth = trial.suggest_int("gbmax_depth", 2, 5, log=True)
            gbmin_samples_split= trial.suggest_int("gbmin_samples_split", 2, 5, log=True)
            gbmin_samples_leaf = trial.suggest_int("gbmin_samples_leaf", 1, 3, log=True)
            
            classifier_obj = GradientBoostingClassifier( learning_rate= gb_learning_rate,
                                                        n_estimators = gb_n_eastimators,
                                                        criterion = gb_criteion,
                                                        max_depth = gb_max_depth,
                                                        min_samples_split = gbmin_samples_split,
                                                        min_samples_leaf = gbmin_samples_leaf)
            
        elif classifier_name == "SVC":
            svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
            svc_kernel = trial.suggest_categorical("kernel",["linear",
                                                             "poly",
                                                             "rbf",
                                                             "sigmoid",
                                                             "precomputed"])
            
            classifier_obj = sklearn.svm.SVC(C=svc_c, kernel = svc_kernel,gamma="auto")
          
            
          
        elif classifier_name == "MLPClassifier":
            # mlp_learning_rate = trial.suggest_categorical("learning_rate",["constant",
            #                                                               "invscaling",
            #                                                               "adaptive"])
            
            mlp_activation = trial.suggest_categorical("activation",["logistic",
                                                                          "tanh","relu"])
            mlp_solver = trial.suggest_categorical("solver",["sgd",
                                                             "adam"])  
            
            #mlp_hd = trial.suggest_categorical('hidden_layer_sizes',[(100,), (200,), (50,), (50, 100)])
            
            mlp_learning_rate_init = trial.suggest_float("learning_rate_init",0.001, 0.05, log=True)
            classifier_obj =MLPClassifier( alpha = 0.0001, hidden_layer_sizes = (50,100),
                                           max_iter=300,
                                          learning_rate = "adaptive",
                                          activation = mlp_activation,
                                          solver = mlp_solver,
                                          learning_rate_init = mlp_learning_rate_init
                                          )                   
                        
            
        elif classifier_name == "RandomForestClassifier":
            rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
            classifier_obj = sklearn.ensemble.RandomForestClassifier(
                max_depth=rf_max_depth, n_estimators=10
            )
            
        else :
            raise ValueError("classifier not found",classifier_name)

        score = sklearn.model_selection.cross_val_score(classifier_obj, self.X.copy(), self.y.copy(), n_jobs=1, cv=3)
        accuracy = score.mean()
        return accuracy


# Load the dataset in advance for reusing it each trial execution.
objective = Objective(X_train_final,y_train)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
print(study.best_trial)
    

#%%
noiseIdx = y_train == 1
sns.scatterplot(supervised_embedder[noiseIdx == False,0], # 0 is first embedding
                supervised_embedder[noiseIdx == False,1]) # 1 is second embedding 
sns.scatterplot(supervised_embedder[noiseIdx,0], 
                supervised_embedder[noiseIdx,1],
                color=".0", marker="+", label = "noise")
plt.title("Scatterplot for Noise in data")
plt.show()
#%% run hdbscan
labels = hdbscan.HDBSCAN(min_samples=1, min_cluster_size=10).fit_predict(embedding)
clustered = (labels >-1)
labels[clustered]+1
temp = np.unique(labels) # number of clusters geneatred
print("total clusters generated ", len(temp))
nLabels = np.zeros(labels.shape, 'int')
for x in range(0,temp.shape[0]):
    cIdx = labels == temp[x]
    nLabels[cIdx] = x
    
print("percent of points clustered ",np.sum(labels>-1) / labels.shape[0])  #percent of clustered points

#%% Simple cluster plot after HDBCAN
fig2, ax2 = plt.subplots()


cColor = sns.color_palette("Paired", np.max(nLabels[clustered])+1); # to get number of total clusters


clustered_embeddings = embedding[clustered]
clustered_noiseIdx = noiseIdx[clustered]

ax2.scatter(clustered_embeddings[:, 0], 
            clustered_embeddings[:, 1], 
            c=[cColor[x] for x in (nLabels[clustered])],
            s=40, marker='o')

ax2.set_aspect('equal', adjustable = 'datalim')
fig2.canvas.draw

np.sum(labels>-1) / labels.shape[0]  #percent of clustered points
plt.title("UMAP Projection giving several clusters")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")


#%% to distinguish between noise and not noise within clusters 

fig2, ax2 = plt.subplots()
fig2.set_size_inches(18.5, 10.5)

cColor = sns.color_palette("Paired", np.max(nLabels[clustered])+1); # to get number of total clusters


clustered_embeddings = embedding[clustered]
clustered_noiseIdx = noiseIdx[clustered]

ax2.scatter(clustered_embeddings[clustered_noiseIdx == False,0], 
            clustered_embeddings[clustered_noiseIdx == False,1], 
            c=[cColor[x] for x in (nLabels[clustered][clustered_noiseIdx == False])],
            s=40, marker='o', alpha=0.5)
ax2.scatter(clustered_embeddings[clustered_noiseIdx, 0], 
            clustered_embeddings[clustered_noiseIdx,1],
            c='.0', # [cColor[x] for x in (nLabels[clustered][clustered_noiseIdx])],
            s=40, label="noise", marker='x', alpha=0.4)
ax2.set_aspect('equal', adjustable = 'datalim')
fig2.canvas.draw

np.sum(labels>-1) / labels.shape[0]  #percent of clustered points
plt.title("UMAP Projection giving several clusters")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")

# Add cluster quality metric 

#%% (Supervised UMAP)


# supervised_embedder = umap.UMAP()
# umap_embedding = supervised_embedder.fit_transform(X_train, y=y_train)

# fig, ax = plt.subplots()
# fig.set_size_inches(18.5, 10.5)

# plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], s=60, c=y_train, cmap='Spectral', alpha=1.0)
# plt.setp(ax, xticks=[], yticks=[])

# cbar = plt.colorbar(boundaries=np.arange(3)-0.5)
# cbar.set_ticks(np.arange(2))

# plt.title(' Embedded via UMAP using Labels');

# plt.show()
# #%% Metric Learning 


# umap_embedding = supervised_embedder.transform(X_test)

# fig, ax = plt.subplots()
# fig.set_size_inches(18.5, 10.5)

# plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], s=60, c=y_test, cmap='Spectral', alpha=1.0)
# plt.setp(ax, xticks=[], yticks=[])

# cbar = plt.colorbar(boundaries=np.arange(3)-0.5)
# cbar.set_ticks(np.arange(2))

# plt.title('Embedded via UMAP predicting Labels');

# plt.show()



#%%
import json
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeClassifier



def class_feature_importance(X, Y, feature_importances, features):
    N, M = X.shape
    X = scale(X)

    out = {}
    for c in set(Y):
        out[c] = dict(
            zip(features, np.mean(X[Y==c, :], axis=0)*feature_importances)
        )

    return out
#%% decision tree

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
def create_dataset(frame):
     #data loading
    X = frame.drop(['label'],axis = 1,inplace=False)
    X = X.astype(np.float64)
    y = frame['label']
    
    #data pre=processing
    columns = X.columns

    return X, y, columns

def fit_pipeline(X, y, columns):
    
    pipeline = Pipeline(steps=[('scale', StandardScaler()),
                               ('model', GradientBoostingClassifier())
                               #('model', RandomForestClassifier(class_weight='balanced'))
                               ])
                                   
    # Cross-validation
    cv_score = np.mean(cross_val_score(pipeline, X, y, cv=8))
    
   # pipeline.fit(X, y)
  #  result = class_feature_importance(pd.DataFrame(X, columns=columns), 
   #                                   y,
   #                                   pipeline.named_steps['model'].feature_importances_,
    #                                  columns)

    return cv_score, pipeline

#%%
train_frames = []
for path in paths:
    train_frames.append(create_dataframe(path))

train_dataset = pd.concat(train_frames)
test_dataset = create_dataframe(path_test_frame)


train_dataset = preprocess_frame(train_dataset)
train_dataset.drop(['cluster_id'], axis=1)

test_dataset = preprocess_frame(test_dataset, is_test=True)
cluster_ids = test_dataset['cluster_id']
test_dataset.drop(['cluster_id'], axis=1)


#%%

X, y, columns = create_dataset(train_dataset)
cv_score,pipeline = fit_pipeline(X, y, columns)
#results.append(result)
#%%
# for roc_curve_
X_test =test_dataset
pipeline = pipeline.fit(X, y) #both things again
#metrics.plot_roc_curve(pipeline, X_test, y_test)
#%%
preds = pipeline.predict(X_test)
predictions = pd.DataFrame({'cluster_id': X_test['cluster_id'], 'label': preds})
print(predictions.describe())