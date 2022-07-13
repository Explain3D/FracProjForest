#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 16:06:25 2022

@author: tan
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import joblib
import time
import matplotlib.pyplot as plt

def scale_one(exp):
    sum_exp = np.sum(exp)
    return exp / sum_exp

rotate = 3
n_feat = 228
sample_num = 30
postfix = 'seqgaus2diidv'
rotate_aug = 'Feature' #Rotation augmentation: 'None', 'Number', 'Feature'
rotate_vote = False

#228 dim intotal, which reaches 86.3% acc:
    #3 times rotation, 78 dim for each
    #0-3 dim for num of fractions
    #4-27st dims for sequenced voxels information (max, argmax, min, argmin, mean, var)
    #28-61st dims for distribution inside voxels
    #62-77 dims for sum of sampled voxels (9 samples)

dataset = 'ModelNet40'

train_data = np.load("data/" + dataset + "/data_processed/"+str(rotate)+"_"+str(n_feat)+"_"+str(postfix)+"_"+str(rotate_aug)+"_train_data"+".npy")
train_label = np.load("data/" + dataset + "/data_processed/"+str(rotate)+"_"+str(n_feat)+"_"+str(postfix)+"_"+str(rotate_aug)+"_train_label"+".npy")
test_data = np.load("data/" + dataset + "/data_processed/"+str(rotate)+"_"+str(n_feat)+"_"+str(postfix)+"_"+str(rotate_aug)+"_test_data"+".npy")
test_label = np.load("data/" + dataset + "/data_processed/"+str(rotate)+"_"+str(n_feat)+"_"+str(postfix)+"_"+str(rotate_aug)+"_test_label"+".npy")

rdm_seed = 0
shuffle_p = 100

#clf = joblib.load("log/" + dataset + "/rf/"+str(rotate)+"_"+str(n_feat)+"_"+str(postfix)+"_"+str(rotate_aug)+".joblib")
clf = joblib.load("sanity/model/noised_shuffle_" + str(shuffle_p) + ".joblib")
print("\nProcessing " + str(shuffle_p) + " percent shuffled model...\n")

tic = time.time()
test_pred_new = clf.predict(test_data)
test_acc_new = accuracy_score(test_label,test_pred_new)
toc = time.time()
conf_matrix_new = confusion_matrix(test_label, test_pred_new)
s_f1_new = f1_score(test_label, test_pred_new, average='macro')

print("\n")
print("F1 score: ", s_f1_new )
print("Mean acc per class: ", np.mean(conf_matrix_new.diagonal()/conf_matrix_new.sum(axis=1)))
print("Test acc: ", test_acc_new)
print("Processing time: ", toc - tic)

#Interpretablity: inherent explanations

ori_acc = test_acc_new

feat_dic = np.load("data/" + dataset + "/data_processed/feat_dic.npy")

fractual_mtx = (feat_dic // n_feat) + 1        #Clustering by fractual

rest = feat_dic % n_feat
rotate_mtx = rest // (n_feat//rotate)                   #Clustering by rotation

rest = rest % (n_feat//rotate) 
feat_cat_mtx = np.zeros_like(rest)

feat_cat_mtx[np.argwhere(rest <= 3)] = 0    #Clustering by feature category
feat_cat_mtx[np.intersect1d(np.argwhere(rest>3), np.argwhere(rest<=28))] = 1
feat_cat_mtx[np.intersect1d(np.argwhere(rest>28), np.argwhere(rest<=61))] = 2
feat_cat_mtx[np.intersect1d(np.argwhere(rest>61), np.argwhere(rest<=67))] = 3
feat_cat_mtx[np.intersect1d(np.argwhere(rest>67), np.argwhere(rest<=76))] = 4
exp_idx = np.vstack([fractual_mtx, rotate_mtx, feat_cat_mtx])

#perturb fractual
acc_frac_pool = []
for fr in range(sample_num):
    print("\nPerturbing Fractual number ", fr + 1)
    cur_frac = fr + 1
    cur_frac_idx = np.argwhere(fractual_mtx == cur_frac) 
    train_data_perturb = np.delete(train_data.copy(),cur_frac_idx, axis=1)
    test_data_perturb = np.delete(test_data.copy(),cur_frac_idx, axis=1)
    
    #Retrain a new model
    clf_new = RandomForestClassifier(random_state=rdm_seed)
    clf_new = clf_new.fit(train_data_perturb, train_label)
    train_pred_new = clf_new.predict(train_data_perturb)
    train_acc_new = accuracy_score(train_label, train_pred_new)
    print("Train acc: ", train_acc_new)
    
    test_pred_new = clf_new.predict(test_data_perturb)
    test_acc_new = accuracy_score(test_label, test_pred_new)
    conf_matrix_new = confusion_matrix(test_label, test_pred_new)
    s_f1_new = f1_score(test_label, test_pred_new, average='macro')
    
    print("F1 score: ", s_f1_new )
    print("Mean acc per class: ", np.mean(conf_matrix_new.diagonal()/conf_matrix_new.sum(axis=1)))
    print("Test acc: ", test_acc_new)
    acc_frac_pool.append(test_acc_new)
fractual_attribution = scale_one(-(acc_frac_pool - ori_acc))
np.save("sanity/pert_m/frac_" + str(shuffle_p) + ".npy", fractual_attribution)

#perturb rotation
acc_rotation_pool = []
for rt in range(rotate):
    print("\nPerturbing Fractual number ", rt + 1)
    cur_rotation = rt 
    cur_rot_idx = np.argwhere(rotate_mtx == cur_rotation) 
    train_data_perturb = np.delete(train_data.copy(),cur_rot_idx, axis=1)
    test_data_perturb = np.delete(test_data.copy(),cur_rot_idx, axis=1)
    print(train_data_perturb.shape)
    
    #Retrain a new model
    clf_new = RandomForestClassifier(random_state=rdm_seed)
    clf_new = clf_new.fit(train_data_perturb, train_label)
    train_pred_new = clf_new.predict(train_data_perturb)
    train_acc_new = accuracy_score(train_label, train_pred_new)
    print("Train acc: ", train_acc_new)
    
    test_pred_new = clf_new.predict(test_data_perturb)
    test_acc_new = accuracy_score(test_label, test_pred_new)
    conf_matrix_new = confusion_matrix(test_label, test_pred_new)
    s_f1_new = f1_score(test_label, test_pred_new, average='macro')
    
    print("F1 score: ", s_f1_new )
    print("Mean acc per class: ", np.mean(conf_matrix_new.diagonal()/conf_matrix_new.sum(axis=1)))
    print("Test acc: ", test_acc_new)
    acc_rotation_pool.append(test_acc_new)
rotation_attribution = scale_one(-(acc_rotation_pool - ori_acc))
np.save("sanity/pert_m/rotate_" + str(shuffle_p) + ".npy", rotation_attribution)

#perturb feature
num_feat_cat = 5
acc_feat_pool = []
for ft in range(num_feat_cat):
    print("\nPerturbing Fractual number ", ft + 1)
    cur_feature = ft 
    cur_feat_idx = np.argwhere(feat_cat_mtx == cur_feature) 
    train_data_perturb = np.delete(train_data.copy(),cur_feat_idx, axis=1)
    test_data_perturb = np.delete(test_data.copy(),cur_feat_idx, axis=1)
    print(train_data_perturb.shape)
    
    #Retrain a new model
    clf_new = RandomForestClassifier(random_state=rdm_seed)
    clf_new = clf_new.fit(train_data_perturb, train_label)
    train_pred_new = clf_new.predict(train_data_perturb)
    train_acc_new = accuracy_score(train_label, train_pred_new)
    print("Train acc: ", train_acc_new)
    
    test_pred_new = clf_new.predict(test_data_perturb)
    test_acc_new = accuracy_score(test_label, test_pred_new)
    conf_matrix_new = confusion_matrix(test_label, test_pred_new)
    s_f1_new = f1_score(test_label, test_pred_new, average='macro')
    
    print("F1 score: ", s_f1_new )
    print("Mean acc per class: ", np.mean(conf_matrix_new.diagonal()/conf_matrix_new.sum(axis=1)))
    print("Test acc: ", test_acc_new)
    acc_feat_pool.append(test_acc_new)
feature_attribution = scale_one(-(acc_feat_pool - ori_acc))
np.save("sanity/pert_m/feat_" + str(shuffle_p) + ".npy", feature_attribution)



# Example data
alpha = 0.135     #Voxel rescaling factor

#Fractual exp
exp_frac = np.zeros([sample_num])
frac_list = []
num_list = []
for i in range(sample_num):
    frac_list.append(np.exp(sample_num - i * alpha))
    num_list.append(i)
frac_list = frac_list / frac_list[0]
frac_pos = list(map("{:10.3f}".format, frac_list))

#Rotation exp
exp_rotate = np.zeros([rotate])
rotate_angle = []
for i in range(rotate):
    rotate_angle.append(np.round(np.degrees(np.pi * 2 * i / rotate)))
rotate_pos = list(map("{:10.0f}".format, rotate_angle))

#Feature exp
num_feat_cat = 5
exp_feat = np.zeros([num_feat_cat])
feat_pos = ['Num_V','1D_proj','2D_proj','In_db', 'Samp_db']

fig = plt.figure(tight_layout=True)
fig.set_size_inches(10,4.5)
ax1 = fig.add_subplot(131)
ax1.barh(frac_pos, fractual_attribution, align='center')
ax1.set_yticks(frac_pos)
ax1.invert_yaxis()  # labels read top-to-bottom
ax1.set_xlabel('Attribution')
ax1.set_ylabel('Voxel size')
ax1.yaxis.set_label_position("right")
ax1.set_title('Fractual')

ax2 = fig.add_subplot(132)
ax2.barh(rotate_pos, rotation_attribution, align='center')
ax2.set_yticks(rotate_pos)
ax2.invert_yaxis()  # labels read top-to-bottom
ax2.set_xlabel('Attribution')
ax2.set_ylabel('Rotate Angle')
ax2.yaxis.set_label_position("right")
ax2.set_title('Rotation')

ax3 = fig.add_subplot(133)
ax3.barh(feat_pos, feature_attribution, align='center')
ax3.set_yticks(feat_pos)
ax3.invert_yaxis()  # labels read top-to-bottom
ax3.set_xlabel('Attribution')
ax3.set_ylabel('Feature')
ax3.yaxis.set_label_position("right")
ax3.set_title('Feature')

plt.savefig('exp/' + dataset + '/perturb_attribution.png')