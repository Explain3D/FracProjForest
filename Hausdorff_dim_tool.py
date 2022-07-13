#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 15:30:04 2021

@author: tan
"""

#How spread inside a voxel

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os
import copy
from scipy import optimize

def sparse_argsort(arr):
    indices = np.nonzero(arr)[0]
    return indices[np.argsort(arr[indices])]

def sample_voxel_seq(vox_values, cur_voxel_size):
    total_voxel = int(np.ceil(2 / cur_voxel_size))
    counter_mtx = np.zeros([total_voxel])
    for v in vox_values:
        cur_v = int(v)
        counter_mtx[cur_v] += 1
    return counter_mtx

def sample_voxel_seq_2d(vox_values, cur_voxel_size):
    num_voxels_axis = int(np.ceil(2 / cur_voxel_size))
    voxel_bins = np.zeros([num_voxels_axis,num_voxels_axis])
    for i in range(vox_values.shape[0]):
        vox_idx1 = int(vox_values[i,0])
        vox_idx2 = int(vox_values[i,1])
        voxel_bins[vox_idx1, vox_idx2] += 1 
    voxel_bins = np.reshape(voxel_bins,(-1,1)).squeeze()
    return voxel_bins

    
def get_top_k(counter_mtx, k, order=1):
    num_seq = sparse_argsort(counter_mtx)
    if order == 1:
        return num_seq[-k:][::-1], counter_mtx[num_seq[-k:][::-1]]
    
    elif order == -1:
        return num_seq[:k], counter_mtx[num_seq[:k]]

    else:
        select_v = int(len(num_seq) * order)
        return num_seq[select_v], counter_mtx[num_seq[select_v]]
        

def statisic_seq(counter_mtx, k=2):
    
    arg_max_seq_k, max_seq_k = get_top_k(counter_mtx, k, 1)

    arg_min_seq_k, min_seq_k = get_top_k(counter_mtx, k, -1)
    
    
    
    if arg_max_seq_k.shape[0] < k:
        padding = np.zeros(k-arg_max_seq_k.shape[0])
        arg_max_seq_k = np.hstack([arg_max_seq_k, padding])
        max_seq_k = np.hstack([max_seq_k, padding])
        arg_min_seq_k = np.hstack([arg_min_seq_k, padding])
        min_seq_k = np.hstack([min_seq_k, padding])
    
    
    mean_vox = np.mean(counter_mtx)
    var_vox = np.var(counter_mtx)
    return max_seq_k, arg_max_seq_k, min_seq_k, arg_min_seq_k, mean_vox, var_vox  #max_vox, arg_max_vox, min_vox, arg_min_vox, med_vox, arg_med_vox,
    

def sample_indv_vox(points, counter_x, counter_y, counter_z, xvoxel, yvoxel, zvoxel, xprop=0.5, yprop=0.5, zprop=0.5):
    #Number
    x_tar_idx = int(len(counter_x) * xprop)
    x_p_idx = np.where(xvoxel == x_tar_idx)
    y_tar_idx = int(len(counter_y) * yprop) 
    y_p_idx = np.where(yvoxel == y_tar_idx)
    z_tar_idx = int(len(counter_z) * zprop)
    z_p_idx = np.where(zvoxel == z_tar_idx)
    points_in_vox = np.intersect1d(np.intersect1d(x_p_idx, y_p_idx), z_p_idx)
    num_points = points_in_vox.shape[0]
    
    #Distribution
    cur_points = points[points_in_vox]
    
    if num_points == 0:
        mean_x_p, mean_y_p, mean_z_p, var_x_p, var_y_p, var_z_p = 0,0,0,0,0,0
    else:
        mean_x_p = np.mean(cur_points[:,0])
        mean_y_p = np.mean(cur_points[:,1])
        mean_z_p = np.mean(cur_points[:,2])
        var_x_p = np.var(cur_points[:,0])
        var_y_p = np.var(cur_points[:,1])
        var_z_p = np.var(cur_points[:,2])
        
    return num_points, mean_x_p, mean_y_p, mean_z_p, var_x_p, var_y_p, var_z_p

def voxel_statistical(points, voxel):
    min_voxel = np.int(np.min(voxel))
    max_voxel = np.int(np.max(voxel))
    
    total_m = []
    total_v = []
    total_i = []
    total_a = []
    for indx in range(min_voxel, max_voxel+1):
        cur_idx = np.where(voxel == indx)
        cur_points = points[cur_idx]
        cur_mean = np.mean(cur_points)
        cur_var = np.var(cur_points)
        if len(cur_points) == 0:
            cur_min = np.nan
            cur_max = np.nan
        else:
            cur_min = np.min(cur_points)
            cur_max = np.max(cur_points)
        total_m.append(cur_mean)
        total_v.append(cur_var)
        total_i.append(cur_min)
        total_a.append(cur_max)
    total_m = np.array(total_m)
    total_m = total_m[~np.isnan(total_m)]    
    total_v = np.array(total_v)
    total_v = total_v[~np.isnan(total_v)] 
    total_i = np.array(total_i)
    total_i = total_i[~np.isnan(total_i)] 
    total_a = np.array(total_a)
    total_a = total_a[~np.isnan(total_a)]
    
    mean_final = np.mean(total_m)
    var_final = np.mean(total_v)
    min_final = np.min(total_i)
    max_final = np.max(total_a)
    return mean_final, var_final, min_final, max_final
    
    

def rotate_points(points, dup):
    points_list = []
    for i in range(dup):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points[:,0:3]) 
        R = pc.get_rotation_matrix_from_xyz((0, 2 * np.pi * (i) / dup, 0))
        pc.rotate(R, center=(0, 0, 0))
        pc_ar = np.asarray(pc.points)
        points_list.append(pc_ar)
    return points_list

    
def count_points_in_voxel(points,voxel_size):
    
    def gaussfit_1d(voxel):
        X = np.arange(voxel.size)
        x = np.sum(X*voxel)/np.sum(voxel)
        width = np.sqrt(np.abs(np.sum((X-x)**2*voxel)/np.sum(voxel)))
        max = voxel.max()
        fit = lambda t : max*np.exp(-(t-x)**2/(2*width**2))
        return x, width
    
    def gaussfit_2D(voxel):
        def gaussian(height, center_x, center_y, width_x, width_y):
            """Returns a gaussian function with the given parameters"""
            width_x = float(width_x)
            width_y = float(width_y)
            pars =  lambda x,y: height*np.exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)
            return pars
                
        def moments(data):
            """Returns (height, x, y, width_x, width_y)
            the gaussian parameters of a 2D distribution by calculating its
            moments """
            total = data.sum()
            X, Y = np.indices(data.shape)
            x = (X*data).sum()/total
            y = (Y*data).sum()/total
            col = data[:, int(y)]
            width_x = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
            row = data[int(x), :]
            width_y = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
            height = data.max()
            return height, x, y, width_x, width_y
        
        def fitgaussian(data):
            """Returns (height, x, y, width_x, width_y)
            the gaussian parameters of a 2D distribution found by a fit"""
            params = moments(data)
            errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                         data)
            p, success = optimize.leastsq(errorfunction, params)
            return p
    
        len_1D = voxel.shape[0]
        len_2D = int(np.sqrt(len_1D))
        if len_2D < 3:
            return np.zeros([5])
        voxel = np.reshape(voxel,[len_2D,len_2D])
        params = fitgaussian(voxel)
        (height, x, y, width_x, width_y) = params
        params[np.isnan(params)] = 0
        return params
    
    #Tiny shift the axis those have points on the boundry
    for c in range(3):
        bdr = np.max(points[:,c])
        if bdr == 1.0:
            points[:,c] -= 0.01
    #Calculating index of voxels
    points = points + 1
    voxels = points // voxel_size
    
    x_values = voxels[:,0]
    y_values = voxels[:,1]
    z_values = voxels[:,2]

    #Gathering information of voxels in the space sequence
    #E.g. if there is no points in (0,0,0), then count=0 in 0-voxel
    counter_x = sample_voxel_seq(x_values, voxel_size)
    counter_y = sample_voxel_seq(y_values, voxel_size)
    counter_z = sample_voxel_seq(z_values, voxel_size)
    #Gaussfit_1d return shape of (2,)
    gaus_x = gaussfit_1d(counter_x)
    gaus_y = gaussfit_1d(counter_y)
    gaus_z = gaussfit_1d(counter_z)

    
    #Calculate the voxel with maximum/minimum number of points, and their indexes
    x_max_seq, x_argmax_seq, x_min_seq, x_argmin_seq, x_mean_seq, x_var_seq = statisic_seq(counter_x, 1) #x_max_seq, x_argmax_seq, x_min_seq, x_argmin_seq, x_mean_seq, x_var_seq
    y_max_seq, y_argmax_seq, y_min_seq, y_argmin_seq, y_mean_seq, y_var_seq = statisic_seq(counter_y, 1)
    z_max_seq, z_argmax_seq, z_min_seq, z_argmin_seq, z_mean_seq, z_var_seq = statisic_seq(counter_z, 1)
    x_final = np.hstack([x_max_seq, x_argmax_seq, x_min_seq, x_argmin_seq, x_mean_seq, x_var_seq, gaus_x])
    y_final = np.hstack([y_max_seq, y_argmax_seq, y_min_seq, y_argmin_seq, y_mean_seq, y_var_seq, gaus_y])
    z_final = np.hstack([z_max_seq, z_argmax_seq, z_min_seq, z_argmin_seq, z_mean_seq, z_var_seq, gaus_z])
    seq_info = np.hstack([x_final, y_final, z_final])


    #Projecting to 3 planes
    voxels_xy = points[:,[0,1]] // voxel_size
    voxels_xz = points[:,[0,2]] // voxel_size
    voxels_yz = points[:,[1,2]] // voxel_size
    counter_xy = sample_voxel_seq_2d(voxels_xy, voxel_size)
    counter_yz = sample_voxel_seq_2d(voxels_yz, voxel_size)
    counter_xz = sample_voxel_seq_2d(voxels_xz, voxel_size)

    #gaussfit_2D return shape of (5,)
    gauss2d_xy = gaussfit_2D(counter_xy)
    gauss2d_yz = gaussfit_2D(counter_yz)
    gauss2d_xz = gaussfit_2D(counter_xz)


    xy_max_seq, xy_argmax_seq, xy_min_seq, xy_argmin_seq, xy_mean_seq, xy_var_seq = statisic_seq(counter_xy, 1) #x_max_seq, x_argmax_seq, x_min_seq, x_argmin_seq, x_mean_seq, x_var_seq
    yz_max_seq, yz_argmax_seq, yz_min_seq, yz_argmin_seq, yz_mean_seq, yz_var_seq = statisic_seq(counter_yz, 1)
    xz_max_seq, xz_argmax_seq, xz_min_seq, xz_argmin_seq, xz_mean_seq, xz_var_seq = statisic_seq(counter_xz, 1)
    
    xy_final = np.hstack([xy_max_seq, xy_argmax_seq, xy_min_seq, xy_argmin_seq, xy_mean_seq, xy_var_seq, gauss2d_xy])
    yz_final = np.hstack([yz_max_seq, yz_argmax_seq, yz_min_seq, yz_argmin_seq, yz_mean_seq, yz_var_seq, gauss2d_yz])
    xz_final = np.hstack([xz_max_seq, xz_argmax_seq, xz_min_seq, xz_argmin_seq, xz_mean_seq, xz_var_seq, gauss2d_xz])
    seq_planes_info = np.hstack([xy_final, yz_final, xz_final])
    

    #Take a look at the distribution of each pixel, and get the mean/max/min of them
    x_st_m, x_st_v, x_st_i, x_st_a = voxel_statistical(points[:,0], x_values)
    y_st_m, y_st_v, y_st_i, y_st_a = voxel_statistical(points[:,1], y_values)
    z_st_m, z_st_v, z_st_i, z_st_a = voxel_statistical(points[:,2], z_values)
    
    inside_voxel_info = np.array([x_st_m, y_st_m, z_st_m, x_st_v, y_st_v, z_st_v])

    #Get the distribution of specific voxels, only counting the total amount
    #4 points seems better
    num_samples = 9
    idv_num = np.zeros([num_samples])
    idv_mean_x = np.zeros([num_samples])
    idv_mean_y = np.zeros([num_samples])
    idv_mean_z = np.zeros([num_samples])
    idv_var_x = np.zeros([num_samples])
    idv_var_y = np.zeros([num_samples])
    idv_var_z = np.zeros([num_samples])
    x_sample_rates = [0.5, 0.2, 0.2, 0.8, 0.8, 0.2, 0.2, 0.8, 0.8]
    y_sample_rates = [0.5, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8]
    z_sample_rates = [0.5, 0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8]
    
    for i in range(num_samples):
        idv_num[i], idv_mean_x[i], idv_mean_y[i], idv_mean_z[i], idv_var_x[i], idv_var_y[i], idv_var_z[i] = sample_indv_vox(points, counter_x, counter_y, counter_z, x_values, y_values, z_values, 
                                                                                                                   x_sample_rates[i], y_sample_rates[i], z_sample_rates[i])
    idv_info = np.array([idv_num])
    idv_info = np.squeeze(idv_info.reshape([-1,1]))
    
    #Counting how many voxels in total
    voxels_unique,counts = np.unique(voxels,axis=0, return_counts=True)
    voxels_unique_xy, counts_xy = np.unique(voxels_xy, axis=0, return_counts=True)
    voxels_unique_yz, counts_yz = np.unique(voxels_yz, axis=0, return_counts=True)
    voxels_unique_xz, counts_xz = np.unique(voxels_xz, axis=0, return_counts=True)
    dim_info = np.array([voxels_unique.shape[0], voxels_unique_xy.shape[0], 
                         voxels_unique_yz.shape[0], voxels_unique_xz.shape[0]])

    final_info = np.hstack([dim_info, seq_info, seq_planes_info, inside_voxel_info, idv_info])
    
    return final_info
    
def Hausdorff_dim(points, num_sample, alpha=0.2, rotate=3): 
    rotate_l = rotate_points(points,rotate)
    #Initializing scale list
    max_step = num_sample
    cur_step = 0
    rec_scale = []
    
    while cur_step < max_step:
                scale = np.exp(max_step - cur_step * alpha)
                cur_step += 1
                rec_scale.append(scale)
        
    rec_scale = np.asarray(rec_scale)
    rec_scale = rec_scale / rec_scale[0]
    cur_step = 0
    for l in range(len(rotate_l)):
        cur_pc = rotate_l[l]
        for dim in range(3):
            if np.max(cur_pc[:,dim] > 1) or np.min(cur_pc[:,dim] < -1):
                cur_min = np.min(cur_pc[:,dim])
                cur_max = np.max(cur_pc[:,dim])
                cur_pc[:,dim] = (cur_pc[:,dim] - cur_min) / ((cur_max - cur_min) / 2) - 1
        cur_step = 0
        rec_num_v = []
        tmp = []
        while cur_step < max_step:
            scale = rec_scale[cur_step]
            dim_info = count_points_in_voxel(cur_pc,scale)
            tmp.append(dim_info[0])
            rec_num_v.append(dim_info)
            cur_step += 1
        rec_num_v = np.asarray(rec_num_v, dtype=np.float64)
        print(rec_num_v[0])
        if rec_num_v.shape[0] < max_step:
            length_diff = int(max_step - rec_num_v.shape[0])
            supplement = np.repeat(np.expand_dims(rec_num_v[-1],0),length_diff,axis=0)
            rec_num_v = np.concatenate([rec_num_v, supplement],axis=0)
        if l == 0:
            store_mat = np.expand_dims(rec_num_v,0)
        else:
            store_mat = np.vstack((store_mat, np.expand_dims(rec_num_v,0)))
    return store_mat

            

