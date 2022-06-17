# import packages
import random
import argparse

import numpy as np

import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from funcs_to_use import (
    get_preprocessing_umap_pipeline,
    get_smote_resampled,
    split_data_to_X_y
)
from preprocessing import remove_miss_vals,get_roc_metrics, create_preprocessing_pipeline
from preprocessing import create_preprocessing_pipeline
from without_voting_classifier import run_best_estimator

# path to default classifier (repository link)
repository_path = r"C:\Users\jain\Documents\GitHub\kilosort_post_processing_analysis"


def main(
    seed=764,
    umap=True,
    cv_splits=5,
    lab_dataset =  pd.read_csv(r'D:\cosyne_analysis\dataset.csv',index_col=[0]),
    target_dataset = pd.read_csv(r'D:\SharedEcephys\From_Mortiz\2143_20210324_g0_t0_imec0\imec0_ks2\metrics.csv'),
    exp_dir=r'D:\kilosort_post_analysis_outputs\for_mortiz',
    rec_name = 2143
    ):

    exp_dir = os.path.join(exp_dir, f"{seed}")
    os.makedirs(exp_dir, exist_ok=True)  # change to false after timestamps added
    
    frame = remove_miss_vals(lab_dataset)
    target_frame = remove_miss_vals(target_dataset)

    #calculating roc_metrics
    auc_vals_frame = get_roc_metrics(frame)
    keep_cols =  np.where((auc_vals_frame.roc_auc_val > 0.69) | (auc_vals_frame.roc_auc_val < 0.31))[0].tolist() # get metric with high AUC values 
    strict_metrics = frame.columns[keep_cols].values

    required_output = {}
    required_output[seed] = {}
    required_output[seed]['preds'] = []
    required_output[seed]['probs'] = []
    required_output[seed]['y_test'] = []
    print(f"Starting to create predictions for leave_out_frame:{rec_name}")
    predictions_exp_dir = os.path.join(exp_dir, f'prediction_outputs_{rec_name}')
    os.makedirs(predictions_exp_dir, exist_ok=True)

    X_train, y_train = split_data_to_X_y(frame)
    X_train, y_train = get_smote_resampled(X_train, y_train, seed)

    preprocess_pipeline =  get_preprocessing_umap_pipeline(seed) if umap else create_preprocessing_pipeline(seed)
    kfold = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    X_train_final=X_train[strict_metrics]
    X_test_final=target_frame[strict_metrics]

    run_func = run_best_estimator
    y_pred, y_prob, X_test = run_func(
        X_train=X_train_final,
        y_train=y_train,
        X_test=X_test_final,
        preprocess_pipeline=preprocess_pipeline,
        kfold=kfold,
        seed=seed,
        exp_dir=predictions_exp_dir,
        strict_metrics=strict_metrics)

    required_output[seed]['preds'].append(y_pred)
    np.save(os.path.join(predictions_exp_dir, 'predictions.npy'), np.array(y_pred))
    required_output[seed]['probs'].append(y_prob)
    np.save(os.path.join(predictions_exp_dir, 'probabilities.npy'), np.array(y_pred))
    X_test['noise_predictions']=y_pred
    X_test['noise_probabilities']=y_prob[:,1]
  # X_test.to_csv(os.path.join(predictions_exp_dir, 'output_with_metrics.csv'))


    mapping = {False: 'neural', True: 'noise'}
    labels = [mapping[value] for value in X_test['noise_predictions']]
    target_frame['classification'] = labels
    target_frame['noise_probabilities'] = y_prob[:,1]
    target_frame.to_csv(os.path.join(predictions_exp_dir, 'output_with_whole_metrics.csv'))

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

# parser.add_argument(
#     '-- target_dataset',
#     type= pd.DataFrame,
#     target_dataset=pd.read_csv(r'D:\SharedEcephys\From_Mortiz\2143_20210324_g0_t0_imec0\imec0_ks2\metrics.csv'),
    
# )

parser.add_argument(
    '--exp_dir',
    type=str,
    default=r'D:\kilosort_post_analysis_outputs'
)

# parser.add_argument(
#     '--lab_dataset',
#     type = pd.DataFrame,
#     lab_dataset=pd.read_csv(r'D:\cosyne_analysis\dataset.csv',index_col=[0])

# )

parser.add_argument(
    '--rec_name',
    type = int,
    default = 2143
)
args = parser.parse_args()
options = vars(args) #option is a dictionary of args
print(options)

if __name__ == '__main__':

    np.random.seed(args.seed)
    random.seed(args.seed)
    main(**options)
    




