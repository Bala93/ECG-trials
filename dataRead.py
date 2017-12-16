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
    
    path = '/media/htic/NewVolume1/murali/ecg/codes/datasets/data_with_labels.npy'
    dataset_npy = np.load(path)
    dataset_x  = dataset_npy[:,:360].astype(np.float32)
    dataset_y  = dataset_npy[:,360:].astype(np.int)
    
    in_data     = torch.FloatTensor(dataset_x)
    out_labels  = torch.IntTensor(dataset_y)
    
    
    dset    = TensorDataset(in_data,out_labels)
    dloader = DataLoader(dset)

    return dloader
#for x,y in dloader:
#    print x.size(),y.size()
#    print x
#    break
    
