#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 16:19:17 2017

@author: htic
"""

from arch import ECG
import torch
from dataRead import get_data
from torch.autograd import Variable

dloader = get_data()
ecg = ECG()

for x,y in dloader:
    x = torch.unsqueeze(x,1)
    print x.size()
    x = Variable(x)
    y = ecg(x)
    print y
    print y.size()
    break