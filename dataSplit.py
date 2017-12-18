#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 11:24:47 2017

@author: htic
"""

import numpy as np
#from torch.utils.data import DataLoader,TensorDataset
#import matplotlib.pyplot as plt

def split_data():
    
    path = '/media/htic/NewVolume1/murali/ecg/codes/datasets/data_with_labels.npy'
    train_path = '/media/htic/NewVolume1/murali/ecg/codes/datasets/train_data.npy'
    test_path = '/media/htic/NewVolume1/murali/ecg/codes/datasets/test_data.npy'
    
    dataset_npy = np.load(path)
    dataset_x  = dataset_npy[:,:360].astype(np.float32)
    dataset_y  = dataset_npy[:,360:].astype(np.int)
    
    label_N_indices = np.where(dataset_y == 0)
    label_N_data = dataset_x[label_N_indices[0],:]
    
    label_V_indices = np.where(dataset_y == 1)
    label_V_data = dataset_x[label_V_indices[0],:]
    
    label_A_indices = np.where(dataset_y == 2)
    label_A_data = dataset_x[label_A_indices[0],:]
    
    label_C_indices = np.where(dataset_y == 4)
    label_C_data = dataset_x[label_C_indices[0],:]
    
#    print label_N_data.shape,label_V_data.shape,label_A_data.shape,label_C_data.shape
    
    
    total_size = 2500
    # Restricing dataset to 2500
    total_N_data  = label_N_data[:total_size,:]
    total_V_data  = label_V_data[:total_size,:]
    total_A_data  = label_A_data[:total_size,:]
    total_C_data  = label_C_data[:total_size,:]
    
    
    train_size = 2000
    
    # Train data
    train_N_data  = total_N_data[:train_size,:]
    train_V_data  = total_V_data[:train_size,:]
    train_A_data  = total_A_data[:train_size,:]
    train_C_data  = total_C_data[:train_size,:]
    
    #Train labels
    train_N_labels = np.zeros((train_size,1))
    train_V_labels = np.ones((train_size,1))
    train_A_labels = 2 * np.ones((train_size,1))
    train_C_labels = 3 * np.ones((train_size,1))
    
    
    # Test data
    test_N_data   = total_N_data[train_size:,:]
    test_V_data   = total_V_data[train_size:,:]
    test_A_data   = total_A_data[train_size:,:]
    test_C_data   = total_C_data[train_size:,:]
    
    # Test labels
    test_N_labels = np.zeros((total_size - train_size,1))
    test_V_labels = np.ones((total_size - train_size,1))
    test_A_labels = 2 * np.ones((total_size - train_size,1))
    test_C_labels = 3 * np.ones((total_size - train_size,1))
    
    
    
    train_data = np.vstack((train_N_data,train_V_data,train_A_data,train_C_data))
    train_labels = np.vstack((train_N_labels,train_V_labels,train_A_labels,train_C_labels))
    test_data  = np.vstack((test_N_data,test_V_data,test_A_data,test_C_data))
    test_labels  = np.vstack((test_N_labels,test_V_labels,test_A_labels,test_C_labels))
    
    train_data_with_label = np.hstack((train_data,train_labels))
    test_data_with_label  = np.hstack((test_data,test_labels))
    
    np.random.shuffle(train_data_with_label)
    np.random.shuffle(test_data_with_label)
    
    np.save(train_path,train_data_with_label)
    np.save(test_path,test_data_with_label)    

    return

if __name__ == "__main__":
    split_data()