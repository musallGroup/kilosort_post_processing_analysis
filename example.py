# get modules and params
import random
import argparse

import numpy as np

import os
import random
import argparse

import numpy as np
from sklearn.model_selection import StratifiedKFold

from funcs_to_use import (
    get_preprocessing_umap_pipeline,
    get_smote_resampled,
    metrics_to_use,
    get_leave_one_out_frame,
    plot_cm,
    split_data_to_X_y
)
from preprocessing import create_preprocessing_pipeline
from without_voting_classifier import run_best_estimator
from voting_classifier import run_voting_classifier

# path to default classifier (repository link)
defClassifierPath = r"C:\Users\jain\Documents\GitHub\kilosort_post_processing_analysis"
#np.random.choice(range(1000),3)

def main(seed=764,
    umap=True,
    cv_splits=5,
    target_folder=r'D:\SharedEcephys\Ferimos_data\FromFermino',
    exp_dir=r'D:\kilosort_post_analysis_outputs',
    use_voting_classifier=True
    ):
    exp_dir = os.path.join(exp_dir, f"{seed}")
    os.makedirs(exp_dir, exist_ok=True)  # change to false after timestamps added

    # get recordings and keep the ones that have the cluster_group.tsv file
    folder_check = os.listdir(target_folder)
    file_paths = []
    for i, path in enumerate(folder_check):
        if os.path.isfile(os.path.join(target_folder,path,'cluster_group.tsv')):
            file_paths.append(os.path.join(target_folder,path))
        
    # output : we get 7 folders with Kilosort outputs. 

    strict_metrics = metrics_to_use(file_paths) # quality metrics with auc-roc greater than threshold
    keep_data_frames, leave_out_dfs = get_leave_one_out_frame(file_paths,strict_metrics)

    required_output = {}
    required_output[seed] = {}
    required_output[seed]['preds'] = []
    required_output[seed]['probs'] = []
    required_output[seed]['y_test'] = []

    for i in range(len(keep_data_frames)):
        print(f"Starting to create predictions for leave_out_frame:{i}")
        predictions_exp_dir = os.path.join(exp_dir, f'prediction_outputs_{i}')
        os.makedirs(predictions_exp_dir, exist_ok=True)  # change to false after timestamps added

        X_train, y_train = split_data_to_X_y(keep_data_frames[i])
        X_test, y_test = split_data_to_X_y(leave_out_dfs[i])
        X_train, y_train = get_smote_resampled(X_train, y_train, seed)

        preprocess_pipeline =  get_preprocessing_umap_pipeline(seed) if umap else create_preprocessing_pipeline(seed)
        kfold = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)

        run_func = run_voting_classifier if use_voting_classifier else run_best_estimator
        y_pred, y_prob = run_func(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            preprocess_pipeline=preprocess_pipeline,
            kfold=kfold,
            seed=seed,
            exp_dir=predictions_exp_dir,
            strict_metrics=strict_metrics
        )

        required_output[seed]['preds'].append(y_pred)
        np.save(os.path.join(predictions_exp_dir, 'predictions.npy'), np.array(y_pred))
        required_output[seed]['probs'].append(y_prob)
        np.save(os.path.join(predictions_exp_dir, 'probabilities.npy'), np.array(y_pred))
        required_output[seed]['y_test'].append(y_test)
        np.save(os.path.join(predictions_exp_dir, 'y_test.npy'), np.array(y_pred))

        plot_cm(y_test, y_pred, f'Classifier[{i}]', exp_dir)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--seed',
    type=int,
    default=764)
parser.add_argument(
    '--umap',
    type=bool,
    default=True)
parser.add_argument(
    '--cv_splits',
    type=int,
    default=5)
parser.add_argument(
    '--target_folder',
    type=str,
    default=r'D:\SharedEcephys\Ferimos_data\FromFermino')
parser.add_argument(
    '--exp_dir',
    type=str,
    default=r'D:\kilosort_post_analysis_outputs'
)
parser.add_argument(
    '--use_voting_classifier',
    type=bool,
    default=False)
args = parser.parse_args()
options = vars(args) #option is a dictionary of args
print(options)

if __name__ == '__main__':

    np.random.seed(args.seed)
    random.seed(args.seed)
    options['exp_dir'] = os.path.join(args.exp_dir, f"use_voting_classifier_{args.use_voting_classifier}")
    main(**options)
    
