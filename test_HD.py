#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 11:25:55 2021

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
from Hausdorff_dim_tool import Hausdorff_dim

def sampling(points, sample_size):
    num_p = points.shape[0]
    index = range(num_p)
    np.random.seed(1)
    sampled_index = np.random.choice(index,size=sample_size)
    sampled = points[sampled_index]
    return sampled

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


class_name = 'person'

num_class = 40
SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(BASE_DIR, 'data/shape_names.txt'))] 
file_name = 'data/modelnet40_normal_resampled/' + class_name + '/' + class_name + '_0001.txt'
obj =  sampling(np.loadtxt(file_name,delimiter=',')[:,0:3], 1024)

coef, HD = Hausdorff_dim(obj, 0.001)
print(HD)