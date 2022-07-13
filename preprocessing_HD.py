#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:38:32 2022

@author: tan
"""

import torch
import os
import importlib
import open3d as o3d
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
from Hausdorff_dim_tool import Hausdorff_dim as HD_RF
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from get_shapenet import Dataset

np.set_printoptions(threshold=sys.maxsize)

n_points = 1024

dataset = 'ModelNet40'

suffix = ''

if dataset == 'ModelNet40':
    DATA_PATH = 'data/modelnet40_normal_resampled/'
    TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=n_points, split='train',
                                                         normal_channel=False)
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=n_points, split='test',
                                                        normal_channel=False)
    num_train = len(TRAIN_DATASET)
    num_test = len(TEST_DATASET)
    
    
elif dataset == 'ShapeNet':
    DATA_PATH = 'data/'
    dataset_name = 'shapenetcorev2'
    TRAIN_DATASET = Dataset(root=DATA_PATH, dataset_name=dataset_name, num_points=1024, split='train')
    TEST_DATASET = Dataset(root=DATA_PATH, dataset_name=dataset_name, num_points=1024, split='test')
    num_train = len(TRAIN_DATASET)
    num_test = len(TEST_DATASET)
    print(num_train)
    print(num_test)

elif dataset == 'ScanObjectNN':
    suffix = '_obj'
    DATA_PATH = 'data/ScanObjectNN/'
    TRAIN_DATASET = []
    TEST_DATASET = []
    train_x = np.load(DATA_PATH + 'train_x' + suffix + '.npy')
    test_x = np.load(DATA_PATH + 'test_x' + suffix + '.npy')
    train_y = np.load(DATA_PATH + 'train_y' + suffix + '.npy')
    test_y = np.load(DATA_PATH + 'test_y' + suffix + '.npy')
    num_train = train_x.shape[0]
    num_test = test_x.shape[0]

new_train_data = []
new_train_label = []
new_test_data = []
new_test_label = []

rotate = 3
rotate_form = 'h'
alpha = 0.135
sample_num = 10 
additional_par = 'seqgaus2diidv'  #postfix, could be any string, but should be identical when calling the model

print("Dataset: ", dataset)
print("Number of samples: ", sample_num)

for n_tr in range(num_train):
    print("Processing training data: ", n_tr)
    if dataset != 'ScanObjectNN':
        current_data = TRAIN_DATASET[n_tr][0]
        current_label = TRAIN_DATASET[n_tr][1]
    elif dataset == 'ScanObjectNN':
        current_data = train_x[n_tr]
        current_label = train_y[n_tr]
    current_HD = HD_RF(current_data, sample_num, alpha, rotate)
    if rotate_form == 'v':
        for rep in range(rotate):
            new_train_data.append(current_HD[rep])
            new_train_label.append(current_label)
    
    elif rotate_form == 'h':
        concated_data = np.hstack(current_HD)
        new_train_data.append(concated_data)
        new_train_label.append(current_label)
    
    print(concated_data.shape)

for n_te in range(num_test):
    print("Processing test data: ", n_te)
    if dataset != 'ScanObjectNN':
        current_data = TEST_DATASET[n_te][0]
        current_label = TEST_DATASET[n_te][1]
    elif dataset == 'ScanObjectNN':
        current_data = test_x[n_te]
        current_label = test_y[n_te]
    current_HD = HD_RF(current_data, sample_num, alpha, rotate)
    
    if rotate_form == 'v':
        new_test_data.append(current_HD[0])
        new_test_label.append(current_label)
    
    elif rotate_form == 'h':
        concated_data = np.hstack(current_HD)
        new_test_data.append(concated_data)
        new_test_label.append(current_label)

    print(concated_data.shape)

new_train_data = np.array(new_train_data)
new_train_label = np.array(new_train_label)


new_test_data = np.array(new_test_data)
new_test_label = np.array(new_test_label)


num_feat = int(new_train_data.shape[-1]) #/ rotate)

print(new_train_data.shape)
print(new_train_label.shape)
print(new_test_data.shape)
print(new_test_label.shape)


#Make direction in data/ before preprocessing
np.save('data/' + dataset + '/data_HD_' + str(num_feat) + '/train_data_' + str(sample_num) + '_rotate_' + str(rotate) + additional_par + suffix + '.npy', new_train_data)
np.save('data/' + dataset + '/data_HD_' + str(num_feat) + '/train_label_' + str(sample_num) + '_rotate_' + str(rotate) + additional_par + suffix + '.npy', new_train_label)
np.save('data/' + dataset + '/data_HD_' + str(num_feat) + '/test_data_' + str(sample_num) + '_rotate_' + str(rotate) + additional_par + suffix + '.npy', new_test_data)
np.save('data/' + dataset + '/data_HD_' + str(num_feat) + '/test_label_' + str(sample_num) + '_rotate_' + str(rotate) + additional_par + suffix + '.npy', new_test_label)



