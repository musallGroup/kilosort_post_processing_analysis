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
import hdbscan

from mlxtend.plotting import plot_decision_regions


from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import umap

# load path cell1
# test1 = r'Y:\invivo_ephys\Neuropixels\RD10_2129_20210112\RD10_2129_20210112_g0\RD10_2129_20210112_g0_imec0\RD10_2129_20210112_g0_t0_imec0\imec0_ks2'
# test2 = r'Y:\invivo_ephys\Neuropixels\RD10_2130_20210119\RD10_2130_20210119_g0\RD10_2130_20210119_g0_imec0\RD10_2130_20210119_g0_t0_imec0\imec0_ks2'

# kilosort_output_folder =[ 
#            #unseen
#            test1,test2
#            ]


#1. Combine csv and tsv files 
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
                
    # 2. merge the dataframes
    for key in frames:
        print(key, '->', frames[key])
        frame = frames[key][0]
        for merge_frame in frames[key][1:]:
            frame = frame.merge(merge_frame, on="cluster_id")
        frames[key] = frame
    
    # 3. write into a file
    for i, path in enumerate(paths_all):
        frames[i].to_csv(os.path.join(path, 'merged_data_frames.csv'))
    return frames


# 4. cleaned datasets : all non-useful features are dropped andchange categorical into integer

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

# function for n_spikes cell3
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


def create_preprocessing_pipeline():
    preprocessing_pipeline = Pipeline([
        ('impute', SimpleImputer(missing_values=np.nan, strategy='median')),
        ('scaler', StandardScaler())
        #('umap-embedder', umap.UMAP(min_dist=0.0, n_neighbors=30, n_components=2,random_state=4 ))
        ])
    return preprocessing_pipeline



