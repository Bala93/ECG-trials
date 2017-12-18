#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 16:19:17 2017

@author: htic
"""
from torch.autograd import Variable
import torch
import torch.nn as nn

from arch import ECG
from dataRead import get_data


#cuda_available = 
EPOCH = 10
LR = 0.01
dloader,test_x,test_y = get_data()

# Get test data to cuda
test_x = Variable(test_x)
test_x = test_x.cuda()
test_y = Variable(test_y)
test_y = test_y.cuda()


ecg = ECG()
ecg = ecg.cuda()


optimizer = torch.optim.Adam(ecg.parameters(),lr = LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):

    for step,(x,y) in enumerate(dloader):
        x = torch.unsqueeze(x,1)
        
        x = x.cuda()
        y = y.cuda()
        
        x = Variable(x)
        y = Variable(y)
        
        y_predicted = ecg(x)
        
        print y_predicted.size()
        print y.size()
        
        # Training phase
        loss = loss_func(y_predicted,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Test the accuracy
        
        if step % 1000 == 0:
            
            test_predict = ecg(test_x)
            print test_predict.size()
#            pred_y = torch.max(test_predict,1)[1]
            
            
        break
    break
