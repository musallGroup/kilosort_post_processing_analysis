# targFolder = r'D:\SharedEphysData\FromSyliva'
targFolder = r'D:\SharedEcephys\Ferimos_data\FromFermino'
# targFolder = r'D:\SharedEphysData\FromGaia'

# get modules and params
import os
import random
import copy

import numpy as np
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

from funcs_to_use import (
    get_preprocessing_umap_pipeline,
    metrics_to_use,
    get_leave_one_out_frame,
    get_clf_best_estimators_list,
    create_models_for_voting_clfs,
    predict_using_voting_clf,
    plot_cm,
    split_data_to_X_y
)


# path to default classifier (repository link)
defClassifierPath = r"C:\Users\jain\Documents\GitHub\kilosort_post_processing_analysis"
seed=0
CV_SPLITS = 2
np.random.seed(seed)
random.seed(seed)

# get recordings and keep the ones that have the cluster_group.tsv file
folderCheck = os.listdir(targFolder)
file_paths = []
for i, path in enumerate(folderCheck):
    if os.path.isfile(os.path.join(targFolder,path,'cluster_group.tsv')):
        file_paths.append(os.path.join(targFolder,path))
     
# output : we get 7 folders with Kilosort outputs. 

strict_metrics = metrics_to_use(file_paths) # quality metrics with auc-roc greater than threshold

keep_data_frames, leave_out_dfs = get_leave_one_out_frame(file_paths,strict_metrics)

X_train, y_train = split_data_to_X_y(keep_data_frames[0])
X_test, y_test = split_data_to_X_y(leave_out_dfs[0])

preprocess_umap_pipeline = get_preprocessing_umap_pipeline(seed)
kfold = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=seed)

clf_best_estimators = get_clf_best_estimators_list(
    X_train,
    y_train,
    preprocess_umap_pipeline,
    copy.deepcopy(kfold),
    seed
)

classifiers = create_models_for_voting_clfs(X_train, y_train, preprocess_umap_pipeline,clf_best_estimators,copy.deepcopy(kfold), seed)

print("number of classifier models generated : " , len(classifiers))

y_preds, y_probs = predict_using_voting_clf(classifiers, X_test, y_test)

plot_cm(y_test, y_preds, 'Voting classifier')  

totalProbs = np.abs(y_probs[:,0] - y_probs[:,1])

