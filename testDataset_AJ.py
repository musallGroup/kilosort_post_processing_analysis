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

#%%

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
#%%
print(len(classifiers))

y_preds, y_probs = predict_using_voting_clf(classifiers, X_test, y_test)

plot_cm(y_test, y_preds, 'Voting classifier')  

totalProbs = np.abs(y_probs[:,0] - y_probs[:,1])
plt.hist(totalProbs,bins =100 )
plt.title("totalProbs in voting clf")
plt.ylabel('counts')
plt.xlabel('probs')
plt.show()

g = totalProbs > 0.80
h = totalProbs > 0.90
# sum(g)
# sum(h)
#h = len(totalProbs[g])


#def plot_confusion_matrix(y_true,y_pred)



from itertools import compress
list(compress(y_test, g))
list(compress(y_preds, g))

plot_cm(list(compress(y_test, g)), list(compress(y_preds, g)), 'total Probs > 0.80 from voting clf') 
plot_cm(list(compress(y_test, h)), list(compress(y_preds, h)), 'total Probs > 0.90 from voting clf')           
#%% 






































# #step 2: plot ROC curves and get AUC values 

# # baseParams.get_QMetrics = list(set(baseParams.get_QMetrics))
# # frames = get_feature_columns(fPath, baseParams.get_QMetrics) 
# # gTruths = get_feature_columns(fPath, ['group']) 

# # #create a single large dataframe
# # frame = pd.concat(frames[0],axis = 0)
# # gTruth = pd.concat(gTruths[0],axis = 0)

# # a = pd.get_dummies(gTruth['group'])
# # frame['gTruth'] = a['noise']

# # frame = remove_miss_vals(frame)
# # auc_vals_frame = get_roc_metrics(frame)


# # keep_cols =  np.where((auc_vals_frame.roc_auc_val > 0.79) | (auc_vals_frame.roc_auc_val < 0.21))[0].tolist() # get metric with high AUC values 
# # metrics_names_strict = frame.columns[keep_cols].values # added to baseParams.AUC_Metrics_fermino

# metrics_params = list(set(baseParams.get_QMetrics))
# strict_metrics = metrics_to_use(fPath,metrics_params)
# # output : good metrics that will be a part of the classifier 
# # metrics_names_strict = ['firing_rate', 'presence_ratio', 'cumulative_drift', 'syncSpike_2',
# #        'nearSyncSpike_2', 'syncSpike_4', 'nearSyncSpike_4']

# #%%

# # step 3 : Generate Dataset with metrics_names_strict 
            


    
# #%%
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import accuracy_score, log_loss
# from sklearn.neural_network import MLPClassifier 
# from sklearn.ensemble import ExtraTreesClassifier
# from xgboost import XGBClassifier
# import lightgbm 
# from lightgbm import LGBMRegressor, LGBMClassifier, Booster
# from skopt import BayesSearchCV
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import StratifiedKFold
# from sklearn.ensemble import RandomForestClassifier
# from mlxtend.plotting import plot_decision_regions
# from sklearn.model_selection import cross_validate
# from sklearn.model_selection import cross_val_score
# seed = 1

# clfs = {
#         'AdaBoostClassifier' : AdaBoostClassifier(random_state=seed),
#         'GradientBoostingClassifier' :GradientBoostingClassifier(random_state=seed),
#         'RandomForestClassifier' :RandomForestClassifier(random_state=seed,n_jobs=-1),
#         'KNeighborsClassifier': KNeighborsClassifier(n_jobs=-1),
#         'SVC': SVC(random_state=seed,probability=True),
#         'MLPClassifier' :MLPClassifier(random_state=seed, max_iter=300,hidden_layer_sizes= (50, 100)),
#         'ExtraTreesClassifier' : ExtraTreesClassifier(n_estimators=100, random_state=0),
#         'XGBClassifier' : XGBClassifier(n_estimators=100, random_state=0),
#         'LGBMClassifier' : LGBMClassifier(random_state=0)
# }
# models =  list(clfs.keys())

          
# params = {
#             'AdaBoostClassifier':{'learning_rate':[1,2], 
#                        'n_estimators':[50,100],
#                        'algorithm':['SAMME','SAMME.R']
#                        },#AdaB
    
#             'GradientBoostingClassifier':{'learning_rate':[0.05,0.1],
#                        'n_estimators':[100,150], 
#                        'max_depth':[2,4],
#                        'min_samples_split':[2,4],
#                        'min_samples_leaf': [2,4]
#                        }, #GBC
    
#             'RandomForestClassifier':{'n_estimators':[100,150],
#                        'criterion':['gini','entropy'],
#                        'min_samples_split':[2,4],
#                        'min_samples_leaf': [2,4]
#                        }, #RFC
    
#             'KNeighborsClassifier':{'n_neighbors':[20,50], 
#                        'weights':['distance','uniform'],
#                        'leaf_size':[30]
#                        }, #KNN
    
#             'SVC': {'C':[0.5,2.5],
#                        'kernel':['sigmoid','linear','poly','rbf']
#                        }, #SVC
            
#             'MLPClassifier': {
#                          'activation': ['tanh', 'relu'],
#                          'solver': ['sgd', 'adam'],
#                          'alpha': [0.0001, 0.05],
#                          'learning_rate': ['constant','adaptive']
#                          }, #MLP
    
#             'ExtraTreesClassifier':{'criterion':['gini', 'entropy'],  
#                        'class_weight':['balanced', 'balanced_subsample']
#                        }, #extratrees
    
#              'XGBClassifier':{'max_depth':[2,4], 
#                        'eta': [0.2,0.5], 
#                        'sampling_method':['uniform','gradient_based'],
#                        'grow_policy':['depthwise', 'lossguide']
#                       }, #xgboost
                        
    
#             'LGBMClassifier':{'learning_rate':[0.05,0.15],
#                        'n_estimators': [100,150]} #lightgbm
    
#          }
# # run search with given dataset        

# preprocessing_pipeline = create_preprocessing_pipeline()

# #%%
# from sklearn.utils import shuffle
# from imblearn import under_sampling, over_sampling
# from imblearn.over_sampling import SMOTE, ADASYN

# folds = 8
# from sklearn.pipeline import Pipeline
# pipeline = Pipeline([
#     ('pre', preprocessing_pipeline),
#     ('umap-embedder', umap.UMAP(min_dist=0.0, n_neighbors=10, n_components=2,random_state=4))
#     ])

# X = good_auc_frame_not0.drop(['gTruth'], axis=1)
# y = good_auc_frame_not0['gTruth']
# X, y = shuffle(X, y, random_state=0)

# kfold = StratifiedKFold(n_splits=8, shuffle = True,random_state = 42)
# sm = SMOTE(random_state=42)
# #ad list to test sizes
# X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# # to get equal distribution for labels in y_ytain
# X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# # implementing preprocessing and umap embedder on train and test data



# # X_train_transform = preprocessing_pipeline.fit_transform(X_train_res)
# # X_train_final = supervised_embedder.fit_transform(X_train_transform, y=y_train_res)
# # X_test_trans = preprocessing_pipeline.transform(X_test) #to impute and normalise
# # X_test_final = supervised_embedder.transform(X_test_trans)

# # time passes

# X_train_transform= pipeline.fit_transform(X_train_res)
# X_test_transform= pipeline.fit_transform(X_test)

# #del clf_best_estimators
# clf_best_estimators = []
  
# # run search for each model 
# for name in models:
#     print(name)
    
#     estimator = clfs[name]
#     clf = BayesSearchCV(estimator, params[name], scoring='accuracy', refit='True', n_jobs=-1, n_iter=20,cv=kfold) 
#     clf.fit(X_train_transform, y_train_res)   # X_train_final, y_train X is train samples and y is the corresponding labels
    
#     print("best estimator " +  str(clf.best_estimator_))
#     print("best params: " + str(clf.best_params_))
#     clf_best_estimators.append(clf.best_estimator_)

# #%%
# #del scores

# cv = StratifiedKFold(n_splits=8, shuffle = True,random_state = 42)

# classifiers = []

# for classifier in clf_best_estimators:
#     clf_pipeline = Pipeline([
#         ('pre-umap', pipeline),
#         ('estimator', classifier)])
    
#     for i, (train_index, test_index) in enumerate(cv.split(X_train_res, y_train_res)):
#       X_train_in, X_val = X_train_res.iloc[train_index], X_train_res.iloc[test_index]
#       y_train_in, y_val = y_train_res.iloc[train_index], y_train_res.iloc[test_index]
#       #===========================================================================
#       # X_train_trans = preprocessing_pipeline.fit_transform(X_train_in)
#       # mapper = supervised_embedder.fit(X_train_trans, y=y_train_in
#       #                                  )
#       # #save mapper
#       # mappers.apppend(mapper)
#       # #fig print after every run 
#       # X_train_final = mapper.transform(X_train_trans)
#       # #============================================================================
#       # X_val_trans = preprocessing_pipeline.transform(X_val)
#       # X_val_final = mapper.transform(X_val_trans)
      
      

#       print("running estimator : " , classifier )
#       print("running split : ", i )
#       fitted_clf  = clf_pipeline.fit(X_train_in,y_train_in)
#       classifiers.append(fitted_clf)
      
#       # score = clf_pipeline.predict(X_val)
#       # predictions.append(score)
        


#  #%%
# # eclf = VotingClassifier(estimators= None, voting='soft',n_jobs=-1)
# # eclf.estimators_ = classifiers
# # eclf.le_ = LabelEncoder().fit(y_train_res) #https://stackoverflow.com/questions/42920148/using-sklearn-voting-ensemble-with-partial-fit/54610569#54610569
# # eclf.classes_ = eclf.le_.classes_
# # y_pred_soft = eclf.predict(X_test) 
# # y_probs = eclf.predict_proba(X_test)


# #%% # testing on frame not included 

# X_frame = frames[0][0]
# X_frame = remove_miss_vals(X_frame)
# X_frame = X_frame[metrics_names_strict]

# test_frame = gTruths[0][0].copy(deep = True)
# b = pd.get_dummies(test_frame['group'])
# test_frame['gTruth'] = b['noise']
# y_gtruth = test_frame['gTruth']


# eclf = VotingClassifier(estimators= None, voting='soft',n_jobs=-1)
# eclf.estimators_ = classifiers
# eclf.le_ = LabelEncoder().fit(y_train_res) #https://stackoverflow.com/questions/42920148/using-sklearn-voting-ensemble-with-partial-fit/54610569#54610569
# eclf.classes_ = eclf.le_.classes_
# y_pred_soft_test = eclf.predict(X_frame) 
# y_probs = eclf.predict_proba(X_frame)

# # unique, counts = np.unique(y_pred_soft_test, return_counts=True)
# # np.asarray((unique, counts)).T

# # #%%
# # import matplotlib.pyplot as plt

# # noise_probs = y_probs[:,1]

# # _ = plt.hist(noise_probs, bins='auto')
# # plt.show()
# # #%%
# # from sklearn.metrics import classification_report
# # from sklearn.metrics import accuracy_score
# # from sklearn.metrics import balanced_accuracy_score

# # old_output = pd.read_csv(r'D:\SharedEcephys\Ferimos_data\FromFermino\clfs_folder\clf_forKilosort2_2021-03-13_180605\cluster_noiseOutput.tsv',delimiter="\t")
# # c = pd.get_dummies(old_output['group'])
# # old_output['gTruth'] = c['noise']

# # accuracy_score(y_gtruth,y_pred_soft_test)
# # print(classification_report(y_gtruth,y_pred_soft_test))
# # print(balanced_accuracy_score(y_gtruth,y_pred_soft_test))
# # print(accuracy_score(y_gtruth,old_output['gTruth']))

# #%%
# totalProbs = np.abs(y_probs[:,0] - y_probs[:,1])
# plt.hist(totalProbs)
# g = np.where( totalProbs > 0.70 )
# h = len(totalProbs[g])
# plt.title("totalProbs > 0.70 is 294 in voting clf")
# plt.ylabel('counts')
# plt.xlabel('probs')
# plt.show()
# #%%
# # confusionMatrix = pd.DataFrame(columns=['TruePositive', 'FalsePositive', 'TrueNegative', 'FalseNegative', 'TotalPerf'])
# # def get_confusion_matrix_results(y_pred,y_true) :
# #     confusionMatrix = np.zeros(5)
# #     #true positive (correctly recognized noise clusters)    
# #     true_positive = np.sum((y_pred == 1) & (y_true == 1)) / len(y_true)
# #     #false alarm rate (percent falsely labeled neural clusters)    
# #     false_positive = np.sum((y_pred == 1) & (y_true == 0)) / len(y_true)
# #     #true negative (correctly recognized neural clusters)    
# #     true_negative = np.sum((y_pred == 0) & (y_true == 0)) / len(y_true)
# #     #false negatove (missed noise clusters)    
# #     false_negative  = np.sum((y_pred == 0) & (y_true == 1)) / len(y_true)
# #     #total performance    
# #     performace_overall =  np.round(np.sum(y_pred == y_true) / len(y_true), 2) # permuataion score

# #     return(true_positive,false_positive,true_negative,false_negative,performace_overall)

# # get_confusion_matrix_results(y_pred_soft_test,y_gtruth)
# # get_confusion_matrix_results(old_output['gTruth'],y_gtruth)

# #%%
# #plot confusion matrix 
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# import seaborn as sns
# #def plot_confusion_matrix(y_true,y_pred)

# cm = confusion_matrix(y_gtruth, y_pred_soft_test, labels=clf.classes_)
# ax= plt.subplot()
# sns.heatmap(cm, annot=True, fmt='g', ax=ax,cmap='Blues');  #annot=True to annotate cells, ftm='g' to disable scientific notation
# # labels, title and ticks
# ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
# ax.set_title('Confusion Matrix from Voting classifier'); 
# ax.xaxis.set_ticklabels(['Neural', 'Noise']); ax.yaxis.set_ticklabels(['Neural', 'Noise']);
# #plt.savefig('Confusion Matrix from classifier (SUA).pdf') 
# plt.show()

# # cm = confusion_matrix(y_gtruth,old_output['gTruth'], labels=clf.classes_)
# # ax= plt.subplot()
# # sns.heatmap(cm, annot=True, fmt='g', ax=ax,cmap='Blues');  #annot=True to annotate cells, ftm='g' to disable scientific notation
# # # labels, title and ticks
# # ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
# # ax.set_title('Confusion Matrix from old classifier'); 
# # ax.xaxis.set_ticklabels(['Neural', 'Noise']); ax.yaxis.set_ticklabels(['Neural', 'Noise']);
# # #plt.savefig('Confusion Matrix from classifier (SUA).pdf') 
# # plt.show()
# #%%
# from statistics import mode
# a = scores[0:4]
# outputs = []
# precent_noise = []

# for i in range(a[0].size):
#     item_list =  [item[i] for item in a]
#     output = mode(item_list)
#     outputs.append(output)
#     res_noise = (len([ele for ele in lst3 if ele > 0]) / len(lst3)) * 100
#     precent_noise.append(res_noise)     
# #%%   
          
          
# for clf_estimator in clf_best_estimators:
#     cv_results = cross_validate(clf_estimator, X_train_final, y_train, cv=kfold, return_estimator=True)
#     clf_model = cv_results['estimator']
#     clf_models.append(clf_model)
#     # select the first model
# rfc_fit = rfc_fit[0]

# # save it
# from sklearn.externals import joblib
# filename = os.path.join(savedir, 'final_model.joblib')
# joblib.dump(rfc_fit,filename)

# # load it
# rfc_model2 = joblib.load(filename)
# #%%
# voting_clf = VotingClassifier(estimators=clf_best_estimators, voting='hard')
# clf_best_estimators.append(voting_clf)
# for clf in clf_best_estimators:
#     scores = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=5)

# #%%
# voting_clf.fit(X_train_final, y_train)
# preds_vote = voting_clf.predict(X_test_final)
# probs =  voting_clf.predict_proba(X_test_final)    
# #%% 
# from sklearn.ensemble import StackingRegressor
# from sklearn.linear_model import RidgeCV

# stacking_regressor = StackingRegressor(estimators=clf_best_estimators, final_estimator=RidgeCV())
# stacking_regressor

# #%%
# from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import cross_validate
# X = good_auc_frame_not0.drop(['gTruth'], axis=1)
# y = good_auc_frame_not0['gTruth']

# X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                     test_size=0.10, # 338 left for testing
#                                                     stratify = y , random_state=42)

# preprocessing_pipeline = create_preprocessing_pipeline()

# classifier = SVC(kernel="linear",random_state=42,probability=True, class_weight = 'balanced',
#          decision_function_shape = 'ovo')

# kfold = StratifiedKFold(n_splits=8, shuffle = True,random_state = 42)

# #call the tranformer in data 
# supervised_embedder = umap.UMAP( 
#     min_dist=0.0, 
#     n_neighbors=33, 
#     n_components=2, # dimensions 
#     random_state=42,
#    metric = 'manhattan'
#     )
# #classifier = byesian_seatch se config 
# def get_models(classifier, X_train, y_train, preprocessing_pipeline, supervised_embedder, kfold  ):
 
#     X_train_trans = preprocessing_pipeline.fit_transform(X_train)
#     mapper = supervised_embedder.fit(X_train_trans, y=y_train )
#     X_train_final = mapper.transform(X_train_trans)
#     #===========================================================================
     
#     cv_results = cross_validate(classifier, X_train_final, y_train, cv=kfold, return_estimator=True)
#     clf_fit = cv_results['estimator']
#     return clf_fit
    
# clf_fit = get_models(classifier, X_train, y_train, preprocessing_pipeline, supervised_embedder, kfold  )  
# voting_clf = VotingClassifier(estimators= clf_fit, voting='soft')
   
   
      
      
# #%%     
 
        
# X = good_auc_frame_not0.drop(['gTruth'], axis=1)
# y = good_auc_frame_not0['gTruth']

# X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                     test_size=0.10, # 338 left for testing
#                                                     stratify = y ,
#                                                    random_state=42)
# X_trainextra = X_train.copy(deep = True) 

# def get_nulls(training, testing):
#     print("Training Data:")
#     print(pd.isnull(training).sum())
#     print("Testing Data:")
#     print(pd.isnull(testing).sum())
    
# get_nulls(X_train, X_test)
 

# preprocessing_pipeline = create_preprocessing_pipeline()

# #call the tranformer in data 
# supervised_embedder = umap.UMAP( 
#     min_dist=0.0, 
#     n_neighbors=33, 
#     n_components=2, # dimensions 
#     random_state=42,
#    metric = 'manhattan'
#     )

# clf1 = SVC(random_state=42,probability=True, class_weight = 'balanced',
#          decision_function_shape = 'ovo')
# clf2 =  RandomForestClassifier(random_state=0, class_weight= 'balanced_subsample', 
#                              min_samples_leaf= 4, n_estimators=50) #balanced_subsample
# clf3 =  KNeighborsClassifier(n_neighbors=10)
# clf4 = XGBClassifier()

# rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=2,random_state=42) #8
# clfs = [clf1,clf2,clf3,clf4]
# scores=[]
# voting_scores = []
# l_losses =[]
# f1s = []
# for train_index, test_index in rskf.split(X_train, y_train):
#   X_train_in, X_val = X_train.iloc[train_index], X_train.iloc[test_index]
#   y_train_in, y_val = y_train.iloc[train_index], y_train.iloc[test_index]
#   #===========================================================================
#   X_train_trans = preprocessing_pipeline.fit_transform(X_train_in)
#   mapper = supervised_embedder.fit(X_train_trans, y=y_train_in
#                                    )
#   X_train_final = mapper.transform(X_train_trans)
#   #============================================================================
#   X_val_trans = preprocessing_pipeline.transform(X_val)
#   X_val_final = mapper.transform(X_val_trans) #umap projection learned

#   #===========================================================================

#   clf1.fit(X_train_final,y_train_in)
#   clf2.fit(X_train_final,y_train_in)
#   clf3.fit(X_train_final,y_train_in)
#   clf4.fit(X_train_final,y_train_in)
  
#   svc_pred = clf1.predict(X_val_final)
#   rfc_pred = clf2.predict(X_val_final)
#   knn_pred = clf3.predict(X_val_final)
#   xgb_pred = clf4.predict(X_val_final)
  
  
#   averaged_preds = (svc_pred + rfc_pred + knn_pred + xgb_pred)//4
#   acc = accuracy_score(y_val, averaged_preds)
#   scores.append(acc)
  
#   voting_clf = VotingClassifier(estimators=[('SVC', clf1), ('rfc', clf2), ('knn', clf3),('xgb', clf4) ], voting='soft')
#   voting_clf.fit(X_train_final, y_train_in)
#   preds_vote = voting_clf.predict(X_val_final)
#   probs =  voting_clf.predict_proba(X_val_final)
  
#   d = voting_clf.get_params(deep = True)
#   acc_vote = accuracy_score(y_val, preds_vote)
#   voting_scores.append(acc_vote)
  
#   l_loss = log_loss(y_val, preds_vote)
#   l_losses.append(l_loss)
  
#   f1 = f1_score(y_val, preds_vote)
#   f1s.append(f1)
# #%%
# from matplotlib import pyplot as plt

# eclf = VotingClassifier(estimators=[('SVC', clf1), ('rfc', clf2), ('knn', clf3),('xgb', clf4) ], voting='soft')
# # predict class probabilities for all classifiers
# probas = [c.fit(X_train_final, y_train_in).predict_proba(X_val_final) for c in (clf1, clf2, clf3, clf4, eclf)]

# # get class probabilities for the first sample in the dataset
# class1_1 = [pr[0, 0] for pr in probas]
# class2_1 = [pr[0, 1] for pr in probas]






# #%%  
  
#   y_np = y_train_in.values
#   fig = plot_decision_regions(X=X_train_final, y=y_np, clf=clfs, legend=2)
#   plt.title(f'SUA Decison boundary on clusters');
#   plt.savefig('SUA Decision Boundary from classifier.pdf') 
#   plt.show()
  
#   preds = clf.predict(test_embedding) #y_val
#   score = balanced_accuracy_score(y_val,preds)
#   scores.append(score)

# #%% without umap using X_trainextra

# for train_index, test_index in rskf.split(X_train, y_train):
#   X_train_in, X_val = X_train.iloc[train_index], X_train.iloc[test_index]
#   y_train_in, y_val = y_train.iloc[train_index], y_train.iloc[test_index]
#   #===========================================================================
#   X_train_transform = preprocessing_pipeline.fit_transform(X_train_in)
#   mapper = supervised_embedder.fit(X_train_transform, y=y_train_in
#                                    )
#   X_train_final = mapper.transform(X_train_transform)
#   #============================================================================
#   X_val_trans = pipe.transform(X_val)
#   test_embedding = mapper.transform(X_val_trans) #umap projection learned

#   #===========================================================================

#   clf.fit(X_train_final,y_train_in)
#   y_np = y_train_in.values
#   fig = plot_decision_regions(X=X_train_final, y=y_np, clf=clf, legend=2)
#   plt.title(f'SUA Decison boundary on clusters');
#   plt.savefig('SUA Decision Boundary from classifier.pdf') 
#   plt.show()
  
#   preds = clf.predict(test_embedding) #y_val
#   score = balanced_accuracy_score(y_val,preds)
#   scores.append(score)

# #%%
# from itertools import filterfalse

# resultArrs = {}

# for i in range(len(fPath)):
#     resultArrs[i] =[] 
#     leave_out = [i]
#     filter_it = lambda x: x in leave_out
#     for f in filterfalse(filter_it, range(len(fPath))):
#         resultArrs[i].append(fPath[f])

# #%%
# baseParams.get_QMetrics = list(set(baseParams.get_QMetrics))
# frames = get_feature_columns(fPath, baseParams.get_QMetrics) # change
# gTruth = get_feature_columns(fPath, ['group']) #change


# frame = pd.concat(frames[0],axis = 0)
# gTruth = pd.concat(gTruth[0],axis = 0)
# a = pd.get_dummies(gTruth['group'])
# frame['gTruth'] = a['noise']
# frame = remove_miss_vals(frame)
# auc_vals_frame = get_roc_metrics(frame)


# keep_cols =  np.where((auc_vals_frame.roc_auc_val > 0.79) | (auc_vals_frame.roc_auc_val < 0.21))[0].tolist() # get metric with high AUC values 
# metrics_names_strict = frame.columns[keep_cols].values # added to baseParams.AUC_Metrics_fermino


# classifierPath = r'D:\SharedEcephys\FromFerimos\TestData\clf_testData_strictMetrics'
# identify_best_estimator(frame, metrics_names_strict,classifierPath)

# #%%
# fPath = r'D:\SharedEcephys\FromFerimos\TestData\Kilosort2_2021-10-06_000401'
# filename = os.path.basename(fPath)

# frames = get_feature_columns([fPath], baseParams.get_QMetrics) # get metrics from recording

# run_predictor(frames[0][0],classifierPath) # test classifier

# mapping = {False: 'neural', True: 'noise'}
# labels = [mapping[value] for value in frames[0][0]['is_noise']]

# df = pd.DataFrame(data={'cluster_id' : frames[0][0]['cluster_id'], 'group': labels})
# df.to_csv(classifierPath + filename + r'\cluster_noiseOutput.csv')

# #%% Add Combine all
# for i in range(7):
#     baseParams.get_QMetrics = list(set(baseParams.get_QMetrics))
#     frames = get_feature_columns(resultArrs[4], baseParams.get_QMetrics) # change
#     gTruth = get_feature_columns(resultArrs[4], ['group']) #change
#     filename = os.path.basename(fPath[4]) #change
    
#     frame = pd.concat(frames[0],axis = 0)
#     gTruth = pd.concat(gTruth[0],axis = 0)
#     a = pd.get_dummies(gTruth['group'])
#     frame['gTruth'] = a['noise']
#     frame = remove_miss_vals(frame)
#     auc_vals_frame = get_roc_metrics(frame)
    
    
#     keep_cols =  np.where((auc_vals_frame.roc_auc_val > 0.59) | (auc_vals_frame.roc_auc_val < 0.41))[0].tolist() # get metric with high AUC values 
#     metrics_names = frame.columns[keep_cols].values # added to baseParams.AUC_Metrics_fermino
    
#     metrics_df = pd.DataFrame(frame, columns = np.append(metrics_names,'gTruth'))
    
    
#     classifierPath = r'Y:\invivo_ephys\SharedEphys\FromFermino\clfs_folder' + '\clf_for' + filename
#     identify_best_estimator(metrics_df, metrics_names,classifierPath) # re-train classifier
    
    
#     metrics_names =  np.append(metrics_names,'cluster_id')
#     C_frame = get_feature_columns([fPath[4]], metrics_names) # change
    
#     C_frame = remove_miss_vals(C_frame[0][0])
#     C_frame['cluster_id'] = C_frame['cluster_id'].astype(int)
#     run_predictor(C_frame,classifierPath) # test classifier
    
#     mapping = {False: 'neural', True: 'noise'}
#     labels = [mapping[value] for value in C_frame['is_noise']]
    
#     df = pd.DataFrame(data={'cluster_id' : C_frame['cluster_id'], 'group': labels})
#     df.to_csv(classifierPath +  '\cluster_noiseOutput.tsv', index=False, sep ='\t') 

# #%%
# data_path = r'D:\SharedEcephys\FromDimos\20210803_DK_252MEA6010_le_sp_dorsal'
# baseParams.get_QMetrics.append('cluster_id')
# frames = get_feature_columns(data_path, baseParams.get_QMetrics) # change


# gTruth = get_feature_columns(fPath[0], ['group'])


# test_predictor(test_dataframe, classifierPath)

