#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 16:19:17 2017

@author: htic
"""
from torch.autograd import Variable
import torch
from torch.nn import nn

from arch import ECG
from dataRead import get_data


#cuda_available = 
EPOCH = 10
LR = 0.01
dloader = get_data()
ecg = ECG()



optimizer = torch.optim.Adam(ecg.parameters(),lr = LR)
loss_func = nn.CrossEntropyLoss()

for epoch in EPOCH:

    for step,(x,y) in enumerate(dloader):
        x = torch.unsqueeze(x,1)
        x = Variable(x)
        y_predicted = ecg(x)
        loss = loss_func(y_predicted,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 1000 == 0:
            
        
    
        break
    break
