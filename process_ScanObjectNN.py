#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 11:54:32 2022

@author: tan
"""

import numpy as np
import h5py       

def sampling(points, sample_size):
    num_p = points.shape[0]
    index = range(num_p)
    sampled_index = np.random.choice(index,size=sample_size)
    sampled = points[sampled_index]
    return sampled

def nor_data(dataset):
    x_max = np.max(dataset[:,:,0])
    x_min = np.min(dataset[:,:,0])
    y_max = np.max(dataset[:,:,1])
    y_min = np.min(dataset[:,:,1])
    z_max = np.max(dataset[:,:,2])
    z_min = np.min(dataset[:,:,2])
    x_nor = (dataset[:,:,0] - x_min) / ((x_max - x_min) / 2) - 1
    y_nor = (dataset[:,:,1] - y_min) / ((y_max - y_min) / 2) - 1
    z_nor = (dataset[:,:,2] - z_min) / ((z_max - z_min) / 2) - 1
    
    x_nor = np.expand_dims(x_nor,-1)
    y_nor = np.expand_dims(y_nor,-1)
    z_nor = np.expand_dims(z_nor,-1)
    
    dataset_new = np.concatenate((x_nor,y_nor,z_nor), -1)
    return dataset_new

train_file = 'data/ScanObjectNN/training_objectdataset.h5'
test_file = 'data/ScanObjectNN/test_objectdataset.h5'

num_points = 1024

f_train = h5py.File(train_file,'r+')
f_test = h5py.File(test_file, 'r+')

train_x = f_train['data']
train_y = f_train['label']
test_x = f_test['data']
test_y = f_test['label']

np.save('data/ScanObjectNN/train_y_obj.npy', train_y)
np.save('data/ScanObjectNN/test_y_obj.npy', test_y)

print(train_x.shape)
#sample train data
train_x_processed = []
for i in range(train_x.shape[0]):
    print(i)
    cur_p = train_x[i]
    sampled_p = sampling(cur_p, num_points)
    train_x_processed.append(sampled_p)
train_x_processed = np.asarray(train_x_processed)

    
print(test_x.shape)
#sample train data
test_x_processed = []
for i in range(test_x.shape[0]):
    print(i)
    cur_p = test_x[i]
    sampled_p = sampling(cur_p, num_points)
    test_x_processed.append(sampled_p)
test_x_processed = np.asarray(test_x_processed)

train_x_processed = nor_data(train_x_processed)
test_x_processed = nor_data(test_x_processed)

np.save('data/ScanObjectNN/train_x_obj.npy', train_x_processed)
np.save('data/ScanObjectNN/test_x_obj.npy', test_x_processed)
