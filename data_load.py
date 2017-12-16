#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 12:02:15 2017

@author: htic
"""

import numpy as np
from sklearn.preprocessing import normalize

required_labels = ['N','V','A','J','C']
data_path = '/media/htic/NewVolume1/murali/ecg/codes/datasets/whole_data.npy'
label_path = '/media/htic/NewVolume1/murali/ecg/codes/datasets/whole_data_label.npy'

input_data = np.load(data_path)
label_data = np.load(label_path)
input_data = normalize(input_data,norm='l2',axis=1)
#print np.unique(input_d ata)
#
for i in required_labels:
    label_data[label_data == i] = required_labels.index(i)

label_data = np.reshape(label_data,(86853,1))
label_data = label_data.astype(np.float32)

whole_dataset = np.hstack((input_data,label_data))
np.random.shuffle(whole_dataset)
print whole_dataset[0,:]

np.save('data_with_labels.npy',whole_dataset)

#print np.unique(label_data)
#print whole_dataset.shape

    