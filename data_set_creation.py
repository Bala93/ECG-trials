import wfdb as wf
import numpy as np
from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch.nn as nn




path = '/media/htic/NewVolume1/murali/ecg/codes/ECG/datasets/mitdb/'
dat_path = path + '*.dat'
paths = glob(dat_path)
p2 = [path[:-4] for path in paths]
half_qrs = 180
input_data = np.array([])
input_labels = []

#label
all_labels = []
required_labels = ['N','V','A','J','C']


for path in tqdm(p2):
    
    ann    = wf.rdann(path,'atr')
    record = wf.rdsamp(path)
    labels = ann.symbol
    beats  = ann.sample
    len_beats = len(beats)
    
    data = record.p_signals[:,0]
    
    for i in range(0,len_beats,1):
        x_min = beats[i] - half_qrs
        x_max = beats[i] + half_qrs
        
        if ((beats[-1] < x_max) | (x_min < beats[0])):
            continue
    
        if (record.signame[0] == 'MLII'):
            
            label=labels[i]
            if ((label =='[') | (label ==']') | (label =='!')):
                label= 'V'    
            if ((label =='a')):
                label= 'A' 
            if ((label =='/')):
                label= 'C'    
                
            if label in required_labels:
                
                all_labels.append(label)
                label_map = required_labels.index(label)
                inp = data[x_min:x_max]
                
                if (input_data.shape[0] == 0):
                    input_data = inp
                else:
#                    pass
                    input_data = np.vstack((input_data,inp))
                input_labels.append(label_map)

all_labels = np.array(all_labels)
np.save('/media/htic/NewVolume1/murali/ecg/codes/datasets/whole_data_label.npy',all_labels)    
np.save('/media/htic/NewVolume1/murali/ecg/codes/datasets/whole_data.npy',input_data)    