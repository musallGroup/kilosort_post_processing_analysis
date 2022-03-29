# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 15:29:11 2022

@author: jain
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
#import hdbscan

from mlxtend.plotting import plot_decision_regions

#%% load data

data1 =r'D:\Outputs\catgt_39330_2021-12-20_g0\39330_2021-12-20_g0_imec0\imec0_ks2'
data2 =r'D:\Outputs\catgt_39331_20211220_g0\39331_20211220_g0_imec0\imec0_ks2'
data3 =r'D:\Outputs\catgt_2241_210604_g0\2241_210604_g0_imec0\imec0_ks2'
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
    #print(frame.columns)
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
    if 'snr' in frame.columns:
         frame['snr']= frame['snr'].fillna(frame['snr'].median()) 
    if 'PT_ratio' in frame.columns:
         frame['PT_ratio']= frame['PT_ratio'].fillna(frame['PT_ratio'].median())
         
    if 'recovery_slope' in frame.columns:
         frame['recovery_slope']= frame['recovery_slope'].fillna(frame['recovery_slope'].median())
    if 'repolarization_slope' in frame.columns:
         frame['repolarization_slope']= frame['repolarization_slope'].fillna(frame['PT_ratio'].median())         
    if 'velocity_above' in frame.columns:
         frame['velocity_above']= frame['velocity_above'].fillna(frame['velocity_above'].median())         
    if 'velocity_below' in frame.columns:
         frame['velocity_below']= frame['velocity_below'].fillna(frame['velocity_below'].median())         
        
    #if 'category' in frame.columns:
     #   frame.drop(['category'],axis = 1,inplace=True)
    if 'Unnamed: 0' in frame.columns:
            frame.drop(['Unnamed: 0'],axis = 1,inplace=True)
            
    if 'epoch_name_quality_metrics' in frame.columns:
        frame.drop(['epoch_name_quality_metrics'],axis = 1,inplace=True)
        
    if 'epoch_name_waveform_metrics' in frame.columns:
        frame.drop(['epoch_name_waveform_metrics'],axis = 1,inplace=True)
        
    if 'KSLabel_y' in frame.columns:
        frame.drop(['KSLabel_y'],axis = 1,inplace=True)
    if 'KSLabel_x' in frame.columns:
         frame.drop(['KSLabel_x'],axis = 1,inplace=True)

    if 'KSLabel' in frame.columns:
         frame.drop(['KSLabel'],axis = 1,inplace=True)
    if 'group' in frame.columns:
         frame.drop(['group'],axis = 1,inplace=True)
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
        
    #frame['group'] = enc.fit_transform(frame['group'])
    #print(enc.inverse_transform( [0, 1,0]))
    #frame['KSLabel']= enc.fit_transform(frame['KSLabel'])
    #print(enc.inverse_transform([0, 1])) 
    return(frame)

#%% 
kilosort_output_folder =[data1,data2,data3]
frames = merge_frames(kilosort_output_folder)

frame_metrics=[]
for path in kilosort_output_folder:
    frame_metrics.append(pd.read_csv(os.path.join(path, "metrics.csv"), index_col = 0))
    
#%%
for key, frame in frames.items():
    print(frame.columns)
    frames[key]=preprocess_frame(frame)
#%%
   
spike_clusters = np.load(os.path.join(data3, 'spike_clusters.npy'))
df3 = frames[2].copy()
spikes=[]
for cluster_id in np.unique(spike_clusters):
    spikes.append(np.sum(spike_clusters == cluster_id))

#df = df.set_index('key')
df3['n_spike'] = spikes
#print (df3)
#
frames[2]['n_spike'] = df3['n_spike']
#%%
df3_train = df3.head(600)
df3_train.to_csv(r'D:\Outputs\catgt_39330_2021-12-20_g0\39330_2021-12-20_g0_imec0\df3_train.csv')

df3_unseen_label = df3.iloc[600:800]
df3_unseen_label.to_csv(r'D:\Outputs\catgt_39330_2021-12-20_g0\39330_2021-12-20_g0_imec0\df3_unseen_label.csv')

df3_unseen_nolabel = df3.tail(226)
df3_unseen_nolabel.to_csv(r'D:\Outputs\catgt_39330_2021-12-20_g0\39330_2021-12-20_g0_imec0\df3_unseen_nolabel.csv')
#%%
dataset_merged = pd.read_csv(r'D:\Outputs\catgt_39330_2021-12-20_g0\39330_2021-12-20_g0_imec0\merged_dataset.csv')
dataset_merged['category']= dataset_merged['category ']
dataset_merged.drop(['category '],axis = 1,inplace=True)
dataset_merged.drop(['Unnamed: 0'],axis = 1,inplace=True)
#%%
merged_dataset_new = pd.concat([dataset_merged,df3_train], ignore_index = True)
print(merged_dataset_new.isnull().values.any())
print(merged_dataset_new.isna().any())
#%%
merged_dataset_new.to_csv(r'D:\Outputs\catgt_39330_2021-12-20_g0\39330_2021-12-20_g0_imec0\merged_dataset_new.csv')
#%%
#df3.to_csv(r'D:\Outputs\catgt_39330_2021-12-20_g0\39330_2021-12-20_g0_imec0\df3.csv')
#df2.to_csv(r'D:\Outputs\catgt_39330_2021-12-20_g0\39330_2021-12-20_g0_imec0\df2.csv')
#%%
merged_dataset_new = pd.read_csv(r'D:\Outputs\catgt_39330_2021-12-20_g0\39330_2021-12-20_g0_imec0\merged_dataset_new.csv')
data = merged_dataset_new.copy(deep=True)

#data['category'] = data['category'].map({'sua':1,'p-sua':2,'mua':0,'noise':0}) 
#%%
#%%
if 'Unnamed: 0' in data.columns:
    data.drop(['Unnamed: 0'],axis = 1,inplace=True)
    
if 'category' in data.columns:
    data.drop(['category'],axis = 1,inplace=True)


if 'cluster_id' in data.columns:
    data.drop(['cluster_id'],axis = 1,inplace=True)
print(data.columns)
#%% ROC plots
from sklearn import metrics
plt.figure(figsize=(14,10),dpi=640)

for column in data.columns:
    actual1= data['label']
    prediction1= data['Amplitude']
    fpr, tpr, thresholds = metrics.roc_curve(actual1,prediction1,pos_label=0)
    #fpr, tpr, thresholds = metrics.roc_curve(actual1,prediction1, pos_label=0) for firing_rate
    #auc1 = auc(fpr,tpr)
    roc_auc = metrics.auc(fpr, tpr)
    
    #plt.plot(fpr, tpr,label="AUC prediction:{0}".format(roc_auc),color='red', linewidth=2)
    plt.title(column,fontsize=40)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right',fontsize=40)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate',fontsize=30)
    plt.xlabel('False Positive Rate',fontsize=30)
    plt.show()


#%% ROC plot figure
from sklearn import metrics
plt.figure(figsize=(14,10),dpi=640)
amplitude_cutoff=[]


for key, frame in frames.items():
    actual1= frames[key]['label']
    
    prediction1= frames[key]['amplitude_cutoff']
    
    fpr, tpr, thresholds = metrics.roc_curve(actual1,prediction1)
    roc_auc = metrics.auc(fpr, tpr)
    
    amplitude_cutoff.append(roc_auc)
#%%
#%%
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

#%%
print(daa.columns)
sns.boxplot(data=daa['Amplitude'])
plt.show()
#%%
sorted_index = daa.mean().sort_values().index
df_sorted=daa[sorted_index]
#%%
#fig,ax = plt.subplots()
plt.figure(figsize=(11,11))
plt.xlabel("AUC Values")
plt.axhline(6.5, color ='r', ls = '--') 
plt.axhline(0.5, color ='r', ls = '--') 
sns.boxplot(data=df_sorted, orient='h') 
#sns.stripplot(data=daa ,linewidth=0,orient='h',color='.4',size=4)
plt.title("AUC-ROC Curve Values for different Quality Metrics")
plt.xlabel("AUC Values")
plt.title("AUC-ROC Curve Values for different Quality Metrics for SUA")
plt.savefig('C:\imgs\AUC.pdf') 
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
fig,ax = plt.subplots()
sorted_index = daa.mean().sort_values().index
df_sorted=daa[sorted_index]
plt.axhline(7.5, color ='r', ls = '--') 
sns.boxplot(data=df_sorted, orient='h') 
sns.stripplot(data=sorted_index ,linewidth=0,orient='h',color='.4',size=4)

plt.title("AUC-ROC Curve Values for different Quality Metrics")
plt.xlabel("AUC Values")
plt.title("AUC-ROC Curve Values for different Quality Metrics")
fig=plt.figure(figsize=(9,11))
fig.savefig(r"C:\imgs\AUC_Values.eps", format='eps')


#%%
metric_data = data[[ 'n_spike', 'syncSpike_2', 'syncSpike_4',
                    'Amplitude', 'ContamPct', 'firing_rate', 'presence_ratio',
                    'isi_viol', 'amplitude_cutoff', 'isolation_distance', 'd_prime',
                    'nn_hit_rate', 'nn_miss_rate', 'cumulative_drift',
                 
                              ]].values
  


pipe = Pipeline([('impute', SimpleImputer(missing_values=np.nan, strategy='median')),
                 ('scaler', StandardScaler())])

#call the tranformer in data 
metric_data_new = pipe.fit_transform(metric_data)
supervised_embedder = umap.UMAP( 
    min_dist=0.1, 
    n_neighbors=28,#21
    n_components=2, # dimensions 
    random_state=42
    )
X = supervised_embedder.fit_transform(metric_data_new, y= data['category'])
#X_train_final : fit and transformed with emedder
fig, ax = plt.subplots()
#fig.set_size_inches(18.5, 10.5)

plt.scatter(X[:, 0], X[:, 1], s=10, c= data['category'], cmap='Set1', alpha=1.0)

plt.setp(ax, xticks=[], yticks=[])
cbar = plt.colorbar(boundaries=np.arange(3)-0.2)
cbar.set_ticks(np.arange(3))

plt.title(' Embedded via UMAP using Labels with all quality metrics');

plt.show()
#%%
#ML
#%%
#=============================== ML RF classifier================
merged_dataset_new = pd.read_csv(r'D:\Outputs\catgt_39330_2021-12-20_g0\39330_2021-12-20_g0_imec0\merged_dataset_new.csv')
data = merged_dataset_new.copy(deep=True)

data['category'] = data['category'].map({'sua':1,'p-sua':2,'mua':0,'noise':0}) 

from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestClassifier
rkf = RepeatedKFold(n_splits=6, n_repeats=10, random_state=42)

pipe = Pipeline([('impute', SimpleImputer(missing_values=np.nan, strategy='median')),
                 ('scaler', StandardScaler())])

#%% Supervised UMAP 
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

metric_data_new = pipe.fit_transform(metric_data)
supervised_embedder = umap.UMAP( 
    min_dist=0.1, 
    n_neighbors=28, #21
    n_components=2, # dimensions 
    random_state=42
    )
clf =  RandomForestClassifier(max_depth=2, random_state=0, class_weight= 'balanced_subsample') #balanced_subsample

from sklearn.model_selection import RepeatedKFold
skf = StratifiedKFold(n_splits=6, n_repeats=2, random_state=42)
scores=[]
for train_index, test_index in rkf.split(metric_data):
  X_train, X_test = metric_data[train_index], metric_data[test_index]
  y_train, y_test = data['category'][train_index], data['category'][test_index]
  #===========================================================================
  X_train_transform = pipe.fit_transform(X_train)
  mapper = supervised_embedder.fit(X_train_transform, y=y_train)
  X_train_final = mapper.transform(X_train_transform)
  #============================================================================
  X_test_trans = pipe.transform(X_test)
  test_embedding = mapper.transform(X_test_trans) #umap projection learned

  #===========================================================================

  clf.fit(X_train_final,y_train)
  score = clf.score(test_embedding, y_test)
  scores.append(score)
#%%
# preprocess_frame()
X_unseen = df3_unseen_label[[         'n_spike', 'syncSpike_2', 'syncSpike_4',
                    'Amplitude', 'ContamPct', 'firing_rate', 'presence_ratio',
                    'isi_viol', 'amplitude_cutoff', 'isolation_distance', 'd_prime',
                    'nn_hit_rate', 'nn_miss_rate', 'cumulative_drift', ]].values


X_test_trans = pipe.transform(X_unseen) #to impute and normalise
X_test_final = mapper.transform(X_test_trans)
y_pred = clf.predict(X_test_final) #for every row 
df3_unseen_label['category_Clf']=y_pred
#%%
df3_unseen_label['category'] = df3_unseen_label['category'].map({'sua':1,'p-sua':2,'mua':0,'noise':0}) 
df3_unseen_label = df3_unseen_label.dropna(axis=0)

from sklearn.metrics import accuracy_score

# y_pred1 = np_data['group'].loc[indices_test]
# y_true1 = np_data['gTruth'].loc[indices_test]
y_pred1 =df3_unseen_label['category_Clf']
y_true1 = df3_unseen_label['category']
print(accuracy_score(y_true1,y_pred1))
#%%
val = []
tp = 0 
for index, row in df3_unseen_label.iterrows():
        actual =   row['category'].astype(int)
        prediction_class =  row['category_Clf'].astype(int)
    
        
        if (prediction_class == 1) & (actual == 1):
            tp = tp + 1
            val.append(tp)
    

#%%
if 'Unnamed: 0' in df1.columns:
    df1.drop(['Unnamed: 0'],axis = 1,inplace=True)
    
if 'category ' in df1.columns:
    df1.drop(['category '],axis = 1,inplace=True)


if 'cluster_id' in df1.columns:
    df1.drop(['cluster_id'],axis = 1,inplace=True)
print(dataset_new1.columns)
#%%
from sklearn import metrics
plt.figure(figsize=(14,10),dpi=640)

for column in df1.columns:
    actual1= df1['label']
    prediction1= df1[column]
    fpr, tpr, thresholds = metrics.roc_curve(actual1,prediction1,pos_label=0)
    #fpr, tpr, thresholds = metrics.roc_curve(actual1,prediction1, pos_label=0) for firing_rate
    #auc1 = auc(fpr,tpr)
    roc_auc = metrics.auc(fpr, tpr)
    
    #plt.plot(fpr, tpr,label="AUC prediction:{0}".format(roc_auc),color='red', linewidth=2)
    plt.title(column,fontsize=40)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right',fontsize=40)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate',fontsize=30)
    plt.xlabel('False Positive Rate',fontsize=30)
    plt.show()
#%% Supervised UMAP 
metric_data = dataset_new1[[    'n_spike',
                               'syncSpike_2',
                              'Amplitude', 
                              'ContamPct', 
                              'firing_rate', 
                              'isi_viol',
                              'amplitude_cutoff', 
                              'isolation_distance', 
                              'd_prime', 
                              'nn_hit_rate',
                              'cumulative_drift']].values
  


pipe = Pipeline([('impute', SimpleImputer(missing_values=np.nan, strategy='median')),
                 ('scaler', StandardScaler())])

#call the tranformer in data 
metric_data_new = pipe.fit_transform(metric_data)
supervised_embedder = umap.UMAP( 
    min_dist=0.0, 
    n_neighbors=10, 
    n_components=2, # dimensions 
    random_state=42
    )
X = supervised_embedder.fit_transform(metric_data_new, y=dataset_new1['label'])
#X_train_final : fit and transformed with emedder
fig, ax = plt.subplots()
#fig.set_size_inches(18.5, 10.5)

plt.scatter(X[:, 0], X[:, 1], s=10, c= frame['label'], cmap='Spectral', alpha=1.0)
plt.setp(ax, xticks=[], yticks=[])
cbar = plt.colorbar(boundaries=np.arange(3)-0.5)
cbar.set_ticks(np.arange(2))

plt.title(' Embedded via UMAP using Labels with quality metrics');

plt.show()

#%%
suaIdx = frame['label'] == 1
sns.scatterplot(supervised_embedder[suaIdx == False,0], # 0 is first embedding
                supervised_embedder[suaIdx == False,1]) # 1 is second embedding 
sns.scatterplot(supervised_embedder[suaIdx,0], 
                supervised_embedder[suaIdx,1],
                color=".0", marker="+", label = "noise")
plt.title("Scatterplot for sua in data")
plt.show()
#%% run hdbscan
import hdbscan
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score


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


cColor = sns.color_palette("Paired", np.max(nLabels[clustered]) # to get number of total clusters


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
