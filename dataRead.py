#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 16:01:02 2017

@author: htic
"""

import torch
import numpy as np
from torch.utils.data import DataLoader,TensorDataset


def get_data():
    
    
    train_path = '/media/htic/NewVolume1/murali/ecg/codes/datasets/train_data.npy'
    test_path  = '/media/htic/NewVolume1/murali/ecg/codes/datasets/test_data.npy'
    feature_length = 360
    
    
    train_npy  = np.load(train_path)
    test_npy   = np.load(test_path)

    train_x    = train_npy[:,:feature_length].astype(np.float32)
    train_y    = train_npy[:,feature_length:].astype(np.int)    

    test_x     = test_npy[:,:feature_length].astype(np.float32)
    test_y     = test_npy[:,feature_length:].astype(np.int)

    train_x_torch = torch.FloatTensor(train_x)
    train_y_torch = torch.LongTensor(train_y)

    test_x_torch = torch.FloatTensor(test_x)
    test_y_torch = torch.LongTensor(test_y)
    
    
    train_dataset = TensorDataset(train_x_torch,train_y_torch)
    train_loader  = DataLoader(train_dataset)


    return train_loader,test_x_torch,test_y_torch

#
#if __name__ == "__main__":
#    train_loader,test_x,test_y  = get_data()
#    for (x,y) in train_loader:
#        print x.size(),y.size()
#        break
#    print test_x.size(),test_y.size()
    