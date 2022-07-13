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


rotate = 3
n_feat = 228
sample_num = 30
postfix = 'seqgaus2diidv'
rotate_aug = 'Feature' #Rotation augmentation: 'None', 'Number', 'Feature'
rotate_vote = False

dataset = 'ModelNet40'

#228 dim intotal, which reaches 86.3% acc:
    #3 times rotation, 78 dim for each
    #0-3 dim for num of fractions
    #4-27st dims for sequenced voxels information (max, argmax, min, argmin, mean, var)
    #28-61st dims for distribution inside voxels
    #62-77 dims for sum of sampled voxels (9 samples)

test_data = np.load("data/" + dataset + "/data_processed/"+str(rotate)+"_"+str(n_feat)+"_"+str(postfix)+"_"+str(rotate_aug)+"_test_data"+".npy")
test_label = np.load("data/" + dataset + "/data_processed/"+str(rotate)+"_"+str(n_feat)+"_"+str(postfix)+"_"+str(rotate_aug)+"_test_label"+".npy")

print(test_data.shape)

clf = joblib.load("log/" + dataset + "/rf/"+str(rotate)+"_"+str(n_feat)+"_"+str(postfix)+"_"+str(rotate_aug)+".joblib")


max_depth = list()
max_leaf_nodes = list()
print(len(clf.estimators_))
for tree in clf.estimators_:
    max_depth.append(tree.tree_.max_depth)
    print(tree.tree_.max_depth)

print("avg max depth %0.1f" % (sum(max_depth) / len(max_depth)))

rdm_seed = 0

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
exp = clf.feature_importances_

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



# Example data
alpha = 0.135     #Voxel rescaling factor

#Fractual exp
exp_frac = np.zeros([sample_num])
for i in range(sample_num):
    exp_frac[i] = np.sum(exp[fractual_mtx == i+1])
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
    exp_rotate[i] = np.sum(exp[rotate_mtx == i])
    rotate_angle.append(np.round(np.degrees(np.pi * 2 * i / rotate)))
rotate_pos = list(map("{:10.0f}".format, rotate_angle))

#Feature exp
num_feat_cat = 5
exp_feat = np.zeros([num_feat_cat])
for i in range(num_feat_cat):
    exp_feat[i] = np.sum(exp[feat_cat_mtx == i])
feat_pos = ['Num_V','1D_proj','2D_proj','In_db', 'Samp_db']

fig = plt.figure(tight_layout=True)
fig.set_size_inches(10,4.5)
ax1 = fig.add_subplot(131)
ax1.barh(frac_pos, exp_frac, align='center')
ax1.set_yticks(frac_pos)
ax1.invert_yaxis()  # labels read top-to-bottom
ax1.set_xlabel('Attribution')
ax1.set_ylabel('Voxel size')
ax1.yaxis.set_label_position("right")
ax1.set_title('Fractual')

ax2 = fig.add_subplot(132)
ax2.barh(rotate_pos, exp_rotate, align='center')
ax2.set_yticks(rotate_pos)
ax2.invert_yaxis()  # labels read top-to-bottom
ax2.set_xlabel('Attribution')
ax2.set_ylabel('Rotate Angle')
ax2.yaxis.set_label_position("right")
ax2.set_title('Rotation')

ax3 = fig.add_subplot(133)
ax3.barh(feat_pos, exp_feat, align='center')
ax3.set_yticks(feat_pos)
ax3.invert_yaxis()  # labels read top-to-bottom
ax3.set_xlabel('Attribution')
ax3.set_ylabel('Feature')
ax3.yaxis.set_label_position("right")
ax3.set_title('Feature')

plt.savefig('exp/' + dataset + '_' + 'feat_importance.png')
