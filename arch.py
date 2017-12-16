#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 16:14:41 2017

@author: htic
"""

import torch.nn as nn

batch_size = 1

class ECG(nn.Module):
    
    def __init__(self):
        
        super(ECG,self).__init__()
        self.conv1  = nn.Conv1d(1,64,kernel_size=3,padding=1,stride=1)
        self.bn1    = nn.BatchNorm1d(64*batch_size)
        self.conv2  = nn.Conv1d(64,64,kernel_size=3,padding=1,stride=1)
        self.bn2    = nn.BatchNorm1d(64*batch_size)
        self.conv3  = nn.Conv1d(64,64,kernel_size=3,padding=1,stride=1)
        self.mp1    = nn.MaxPool1d(kernel_size=2)
        
        self.bn3 = nn.BatchNorm1d(64*batch_size)
        self.conv4 = nn.Conv1d(64,64,kernel_size=3,padding=1,stride=1)
        self.bn4 = nn.BatchNorm1d(64*batch_size)
        self.conv5 = nn.Conv1d(64,128,kernel_size=3,padding=1,stride=1)
        self.mp2  = nn.MaxPool1d(kernel_size=2)
        
        
        self.bn5 = nn.BatchNorm1d(128*batch_size)
        self.conv6 = nn.Conv1d(128,128,kernel_size=3,padding=1,stride=1)
        self.bn6 = nn.BatchNorm1d(128*batch_size)
        self.conv7 = nn.Conv1d(128,256,kernel_size=3,padding=1,stride=1)
        self.mp3  = nn.MaxPool1d(kernel_size=2)
        
        self.bn7 = nn.BatchNorm1d(256*batch_size)
        self.fc1 = nn.Linear(11520,5000)
        self.fc2 = nn.Linear(5000,1000)
        self.fc3 = nn.Linear(1000,100)
        self.fc4 = nn.Linear(100,2)
        

    def forward(self,x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)    
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = nn.functional.dropout(x)
        x = self.conv3(x)
        x = self.mp1(x)

        x = self.bn3(x)
        x = nn.functional.relu(x)

        x = nn.functional.dropout(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.functional.relu(x)
        x = nn.functional.dropout(x)
        x = self.conv5(x)
        x = self.mp2(x)
        
        
        x = self.bn5(x)
        x = nn.functional.relu(x)
        x = nn.functional.dropout(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = nn.functional.relu(x)
        x = nn.functional.dropout(x)
        x = self.conv7(x)
        x = self.mp3(x)
        
        
        x = self.bn7(x)
        x = nn.functional.relu(x)
        x = x.view(-1,11520)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        x = nn.functional.relu(x)
        x = self.fc4(x)
        x = nn.functional.softmax(x)
        
        return x

        