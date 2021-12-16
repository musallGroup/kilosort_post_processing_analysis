import os
import pickle
import numpy as np
import pandas as pd

import os.path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import metrics

import seaborn as sns
from BayesianSearch import clfs

def plot_confusion_matrix(frame):
    tp =0; fp =0; fn =0; tn =0
    
    for index, row in frame.iterrows():
        actual =   row['gTruth'] 
        prediction_class =  row['group']
    
        
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
    plt.title("rescale",fontsize=30)
    plt.xlabel("Ground Truth",fontsize=30)
    plt.ylabel("Predicted Label",fontsize=30)
    
    print("Precion is ",tp / (tp + fp))

def get_n_spikes(kilosort_output_folder):
    
    spike_times = np.load(os.path.join(kilosort_output_folder, 'spike_times.npy'))
    spike_clusters = np.load(os.path.join(kilosort_output_folder, 'spike_clusters.npy'))
    spike_templates = np.load(os.path.join(kilosort_output_folder, 'spike_templates.npy'))
    templates = np.load(os.path.join(kilosort_output_folder, 'templates.npy'))
    
    (cluster_best_channel, 
     spike_times, 
     spike_templates, 
     spike_clusters) = compute_best_cluster_channel(templates, 
                                                   spike_clusters, 
                                                   spike_times, 
                                                   spike_templates)    
                                                    
                                                    
    unit_spikes, unit_ids = compute_unit_spikes(spike_times, spike_clusters)    

    unit_n_spikes = [len(x) for x in unit_spikes] 

    return unit_n_spikes, unit_ids    
#   by thomas
def compute_best_cluster_channel(templates, spike_clusters, spike_times, spike_templates):
    
    cluster_best_channel = []
    faulty_inds = []
    for t, temp in enumerate(templates[np.unique(spike_clusters)]):
        if np.sum(np.isnan(temp)) == (temp.shape[0]*temp.shape[1]):
            spike_ind = np.unique(spike_clusters)[t]
            faulty_inds.append(spike_ind)
        else:
            y, x = np.where(np.max(temp) == temp)
            cluster_best_channel.append(x[0])
            
    # exclude all spikes with faulty inds
    for ind in faulty_inds:
        spike_times = spike_times[spike_clusters != ind]
        spike_templates = spike_templates[spike_clusters != ind]
        spike_clusters = spike_clusters[spike_clusters != ind]
        
        
    return cluster_best_channel, spike_times, spike_templates, spike_clusters

def compute_unit_spikes(spike_times, spike_clusters):
    unit_spikes = []
    unit_ids = []
    for cluster_id in np.unique(spike_clusters):
        unit_spikes.append(spike_times[np.where(spike_clusters == cluster_id)])
        unit_ids.append(cluster_id)   
        
    return unit_spikes, unit_ids     
 
def merge_frames(paths_all):
    #3. Combine csv and tsv files 

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

def plot_roc_curve(frame):
    plt.figure(figsize=(14,10),dpi=640)
    actual1= frame['LABEL']
    prediction1=frame['syncSpike_2']
    fpr, tpr, thresholds = metrics.roc_curve(actual1,prediction1,pos_label=0)
    #fpr, tpr, thresholds = metrics.roc_curve(actual1,prediction1, pos_label=0) for firing_rate
    #auc1 = auc(fpr,tpr)
    roc_auc = metrics.auc(fpr, tpr)

    #plt.plot(fpr, tpr,label="AUC prediction:{0}".format(roc_auc),color='red', linewidth=2)
    plt.title('syncSpike_4',fontsize=40)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right',fontsize=40)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate',fontsize=30)
    plt.xlabel('False Positive Rate',fontsize=30)
    plt.show()

def plot_umap_embedding(X, y, title):
    fig, ax = plt.subplots()
    #fig.set_size_inches(18.5, 10.5)

    plt.scatter(X[:, 0], X[:, 1], s=20, c=y, cmap='GnBu', alpha=1.0)
    plt.setp(ax, xticks=[], yticks=[])

    cbar = plt.colorbar(boundaries=np.arange(3)-0.5)
    cbar.set_ticks(np.arange(2))

    plt.title(title)

    plt.show()

