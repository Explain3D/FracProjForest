#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:13:08 2022

@author: tan
"""

import numpy as np
import torch
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, MDS, TSNE

import time
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import joblib

#Todo rotation vote

rotate = 3
n_feat = 228  
sample_num = 30
postfix = 'seqgaus2diidv'    #Identical to the postfix in preprocessing_HD.py
rotate_aug = 'Feature' #Rotation augmentation: 'None', 'Number', 'Feature'
rotate_vote = False #Ony for 'None' nad 'Number' in rotate_aug
save_model = False

dataset = 'ModelNet40'

print("Dataset: ", dataset)

if dataset == 'ScanObjectNN':
    suffix = '_hardest'
    train_x = np.load('data/' + dataset + '/data_HD_' + str(n_feat) + '/train_data_' + str(sample_num) + '_rotate_' + str(rotate) + postfix + suffix + '.npy',allow_pickle=True)
    train_y = np.load('data/' + dataset + '/data_HD_' + str(n_feat) + '/train_label_' + str(sample_num) + '_rotate_' + str(rotate) + postfix + suffix + '.npy',allow_pickle=True)
    test_x = np.load('data/' + dataset + '/data_HD_' + str(n_feat) + '/test_data_' + str(sample_num) + '_rotate_' + str(rotate) + postfix + suffix + '.npy',allow_pickle=True)
    test_y = np.load('data/' + dataset + '/data_HD_' + str(n_feat) + '/test_label_' + str(sample_num) + '_rotate_' + str(rotate) + postfix + suffix + '.npy',allow_pickle=True)

else:
    train_x = np.load('data/' + dataset + '/data_HD_' + str(n_feat) + '/train_data_' + str(sample_num) + '_rotate_' + str(rotate) + postfix + '.npy',allow_pickle=True)
    train_y = np.load('data/' + dataset + '/data_HD_' + str(n_feat) + '/train_label_' + str(sample_num) + '_rotate_' + str(rotate) + postfix + '.npy',allow_pickle=True)
    test_x = np.load('data/' + dataset + '/data_HD_' + str(n_feat) + '/test_data_' + str(sample_num) + '_rotate_' + str(rotate) + postfix + '.npy',allow_pickle=True)
    test_y = np.load('data/' + dataset + '/data_HD_' + str(n_feat) + '/test_label_' + str(sample_num) + '_rotate_' + str(rotate) + postfix + '.npy',allow_pickle=True)


train_y = np.squeeze(train_y)
test_y = np.squeeze(test_y)


num_train = train_x.shape[0]
num_test = test_x.shape[0]

print("Shapes: ")
print(train_x.shape)
print(test_x.shape)

train_x, train_y = shuffle(train_x, train_y,random_state=0)
test_x, test_y = shuffle(test_x, test_y, random_state=0)

#Processing rotation data
cur_num_f = train_x.shape[2]
if rotate_aug == 'None':
    test_rotate_backup = test_x.copy()
    train_x = train_x[:,:,:cur_num_f//rotate]
    test_x = test_x[:,:,:cur_num_f//rotate]
    train_x = train_x.reshape(num_train,-1)
    test_x = test_x.reshape(num_test,-1)
elif rotate_aug == 'Number':
    test_rotate_backup = test_x.copy()
    num_single_r = n_feat // rotate
    for i in range(rotate):
        if i == 0:
            new_train_x = train_x[:,:,i*num_single_r:(i+1)*num_single_r]
        else:
            new_train_x = np.concatenate([new_train_x,train_x[:,:,i*num_single_r:(i+1)*num_single_r]], axis = 0)
        
    new_test_x = test_x[:,:,:cur_num_f//rotate]
    train_x = new_train_x
    test_x = new_test_x
    train_x = new_train_x.reshape(num_train*rotate,-1)
    test_x = new_test_x.reshape(num_test,-1)
    train_y = np.tile(train_y, rotate)
    
elif rotate_aug == 'Feature':
    print(train_x.shape)
    print(test_x.shape)
    train_x = train_x.reshape(num_train,-1)
    test_x = test_x.reshape(num_test,-1)


rdm_seed = 0
n_estimators = 100
min_samples_leaf = 1
min_samples_split = 3
max_depth = None

clf = RandomForestClassifier(random_state=rdm_seed, n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, max_depth=max_depth)
start_train_time = time.time()
clf = clf.fit(train_x, train_y)
end_train_time = time.time()
train_pred = clf.predict(train_x)
train_acc = accuracy_score(train_y,train_pred)
total_train_time = end_train_time - start_train_time
print("Train acc: ", train_acc)
print("Train time: ", total_train_time)

if rotate_vote == True:
    assert rotate_aug != "Feature"
    num_single_r = n_feat // rotate
    pred_mtx = []
    total_time = 0
    for i in range(rotate):
        rotate_test_x = test_rotate_backup[:,:,i*num_single_r:(i+1)*num_single_r]
        rotate_test_x = rotate_test_x.reshape(num_test,-1)
        start_time = time.time()
        test_pred = clf.predict(rotate_test_x)
        end_time = time.time()
        total_time = total_time + (end_time - start_time)
        pred_mtx.append(test_pred)
    pred_mtx = np.asarray(pred_mtx)
    final_pred = []
    for pd in range(pred_mtx.shape[1]):
        cur_p = np.bincount(pred_mtx[:,pd])
        final_pred.append(np.argmax(cur_p))
    final_pred = np.asarray(final_pred)
        
    test_acc = accuracy_score(test_y,final_pred)
    conf_matrix = confusion_matrix(test_y, final_pred)
    s_f1 = f1_score(test_y, final_pred, average='macro')
    

elif rotate_vote == False:
    start_time = time.time()
    test_pred = clf.predict(test_x)
    end_time = time.time()
    total_time = end_time - start_time
    test_acc = accuracy_score(test_y,test_pred)
    conf_matrix = confusion_matrix(test_y, test_pred)
    s_f1 = f1_score(test_y, test_pred, average='macro')

print("\n")
print("F1 score: ", s_f1 )
print("Mean acc per class: ", np.mean(conf_matrix.diagonal()/conf_matrix.sum(axis=1)))
print("Test acc: ", test_acc)
print("Total Test time: ", end_time - start_time)
print("Average Test time per ins: ", total_time / test_x.shape[0])


best_test_score = -float('inf')
best_impt_filter = 0

#Feature selection
impt_filter = 0.00007    #0.00007, 86.3% for ModelNet40
print("\nfilter: ", impt_filter)
start_time = time.time()
importances = clf.feature_importances_
end_time = time.time()
print("EXP time: ", end_time - start_time)
plt.hist(importances)
cri_f = np.argwhere(importances > impt_filter).squeeze()
print(cri_f.shape)

print("Retraining...")
train_x_cleaned = train_x[:, cri_f]
test_x_cleaned = test_x[:, cri_f]

print(train_x_cleaned.shape)


clf_new = RandomForestClassifier(random_state=rdm_seed, n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, max_depth=max_depth)
start_train_time = time.time()
clf_new = clf_new.fit(train_x_cleaned, train_y)
end_train_time = time.time()
train_pred_new = clf_new.predict(train_x_cleaned)
train_acc_new = accuracy_score(train_y,train_pred_new)
total_train_time = end_train_time - start_train_time
print("Train acc: ", train_acc_new)
print("Train time: ", total_train_time)

start_time = time.time()
test_pred_new = clf_new.predict(test_x_cleaned)
end_time = time.time()
total_time = end_time - start_time
test_acc_new = accuracy_score(test_y,test_pred_new)
conf_matrix_new = confusion_matrix(test_y, test_pred_new)
s_f1_new = f1_score(test_y, test_pred_new, average='macro')

print("\n")
print("F1 score: ", s_f1_new )
print("Mean acc per class: ", np.mean(conf_matrix_new.diagonal()/conf_matrix_new.sum(axis=1)))
print("Test acc: ", test_acc_new)
print("Total Test time: ", end_time - start_time)
print("Average Test time per ins: ", total_time / test_x.shape[0])

if test_acc_new > best_test_score:
    best_test_score = test_acc_new
    best_impt_filter = impt_filter

if save_model == True:
    np.save("data/" + dataset + "/data_processed/feat_dic.npy", cri_f)
    joblib.dump(clf_new, "log/" + dataset + "/rf/"+str(rotate)+"_"+str(n_feat)+"_"+str(postfix)+"_"+str(rotate_aug)+".joblib")
    print("Model saved!")
    np.save("data/" + dataset + "/data_processed/"+str(rotate)+"_"+str(n_feat)+"_"+str(postfix)+"_"+str(rotate_aug)+"_train_data"+".npy", train_x_cleaned)
    np.save("data/" + dataset + "/data_processed/"+str(rotate)+"_"+str(n_feat)+"_"+str(postfix)+"_"+str(rotate_aug)+"_train_label"+".npy", train_y)
    np.save("data/" + dataset + "/data_processed/"+str(rotate)+"_"+str(n_feat)+"_"+str(postfix)+"_"+str(rotate_aug)+"_test_data"+".npy", test_x_cleaned)
    np.save("data/" + dataset + "/data_processed/"+str(rotate)+"_"+str(n_feat)+"_"+str(postfix)+"_"+str(rotate_aug)+"_test_label"+".npy", test_y)

print("\nBest test score: ", best_test_score)
print("Best filter parameter: ", best_impt_filter)


