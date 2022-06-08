# -*- coding: utf-8 -*-

"""
@author: Jain 
"""
#cell 0 calling all the libraries 
import pyupset as pyu
from pickle import load
import glob
import os.path
import os
import pickle
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


from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import umap



def get_feature_columns(paths_all, useMetrics):
    frames = {}
    missedMetrics = {}
    for i, path in enumerate(paths_all): #parse with indexing
        
        recFrames = []
        searchMetrics = useMetrics
        
        for file in os.listdir(path):
            
            frame = []
            if ".tsv" in file:
                frame = pd.read_csv(os.path.join(path, file), sep='\t')
                
            elif ".csv" in file:
                frame = pd.read_csv(os.path.join(path, file))

            if len(frame) > 0:
                foundMetrics = set(searchMetrics) & set(list(frame.columns))
                
                if len(foundMetrics) > 0:
                    print('================')
                    print('Found target metrics:')
                    print(foundMetrics)
                    print('in file:')
                    print(file)
                    
                    
                if len(foundMetrics) > 0:
                    searchMetrics = set(searchMetrics).symmetric_difference(foundMetrics)
                    foundMetrics.add('cluster_id') # keep cluster_ID
                    frame = frame[frame.columns.intersection(foundMetrics)]
                    
                else:
                    frame = []
                
            if len(frame) > 0 :  
                if len(recFrames) == 0:
                    recFrames = frame
                else:
                    clusterIDs = pd.concat([frame['cluster_id'], recFrames['cluster_id']],axis = 0) #merge all cluster_ids
                    clusterIDs = pd.DataFrame(np.sort(np.unique(clusterIDs)), columns = ['cluster_id'])
                    
                    recFrames = clusterIDs.merge(recFrames, how = 'left', on = 'cluster_id')        
                    recFrames = recFrames.merge(frame, how = 'left', on = 'cluster_id')        

                 
        frames[i] = recFrames
        missedMetrics[i] = searchMetrics
        
    return frames, missedMetrics

def remove_miss_vals(frame):
    # removing columns with high precentage of nans 
    if 'epoch_name' in frame.columns:
        frame.drop(['epoch_name'],axis = 1,inplace=True)
    if 'Var1' in frame.columns:
        frame.drop(['Var1'],axis = 1,inplace=True)
    if 'Unnamed: 0' in frame.columns:
        frame.drop(['Unnamed: 0'],axis = 1,inplace=True)
        
    print('columns with missing vals : ', frame.columns[frame.isnull().any()].tolist())
    percent_missing = frame.isnull().sum() * 100 / len(frame)
    drop_cols = np.where(percent_missing > 79.0)[0].tolist()
    
    print("columns dropped : " , frame.columns[drop_cols].values)
    
    frame = frame.drop(columns= frame.columns[drop_cols], axis=1)
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    #interpolation to handle missing values
    trans_frame = frame.interpolate(limit_direction ='both')
    percent_missing = trans_frame.isnull().sum() * 100 / len(trans_frame)
    print('columns with missing vals after interpolation : ', trans_frame.columns[trans_frame.isnull().any()].tolist())
    
    return trans_frame


def get_roc_metrics(frame):
    

    """
    Use : Find the metrics that can help identify noise. 

    Inputs:
    -------
    fPath : Output to Kilosort Directory 
    frame : Dataframe with all the quality metrics
    gTruth : Manual label 
    

    Outputs:
    -------
    df_auc : data frame with AUC values for each metric 
    
    """
     
    fig, ax = plt.subplots(figsize=(6, 8))
    
   # a = pd.get_dummies(gTruth['group'])
   # frame['gTruth'] = a['noise']  

    roc_aucs = []
    for column in frame.columns:
        actual = frame['gTruth']
        prediction = frame[column]
        fpr, tpr, thresholds = metrics.roc_curve(actual,prediction)

        roc_auc = metrics.auc(fpr, tpr)
        roc_aucs.append(roc_auc)
     
    metric_col = list(frame.columns)
    L = [list(row) for row in zip(metric_col, roc_aucs)]
    df_auc = pd.DataFrame(L, columns=['q_metric', 'roc_auc_val'])
    
    values = ['gTruth']
    df_auc = df_auc[df_auc.q_metric.isin(values) == False]
    sns.barplot(x='roc_auc_val',
            y="q_metric", 
            data=df_auc, 
            order=df_auc.sort_values('roc_auc_val').q_metric, orient = 'h')
    
    return df_auc



# 4. cleaned datasets : all non-useful features are dropped andchange categorical into integer
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
    #print(enc.inverse_transform( [0, 1,0]))
    frame['KSLabel']= enc.fit_transform(frame['KSLabel'])
    #print(enc.inverse_transform([0, 1]))
    
    
    return(frame)

# function for n_spikes by Thomas Rueland
def get_n_spikes(kilosort_output_folder, total_units):
    
    spike_clusters = np.load(os.path.join(kilosort_output_folder, 'spike_clusters.npy'))
    
    unit_n_spikes = np.zeros(total_units)
    for cluster_id in np.unique(spike_clusters):
        unit_n_spikes[cluster_id] = np.sum(spike_clusters == cluster_id)
        
    return unit_n_spikes


def get_groundTruth(frame):
    gt = 0
    df_i = frame.set_index('cluster_id')
    gTruth=[]
    for index, row in df_i.iterrows():
        label =   row['LABEL'] 
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
            # print(key)
            print(label)
            print(certanity)
            
    df_i['gTruth'] =gTruth
    return(df_i)


def create_preprocessing_pipeline(seed):
    preprocessing_pipeline = Pipeline([
        ('impute', SimpleImputer(missing_values=np.nan, strategy='median')),
        ('scaler', StandardScaler())
        ])
    return preprocessing_pipeline



