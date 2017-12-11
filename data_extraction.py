
import wfdb as wf 
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt


def plot_graph(y1,count,label):

	plt.figure()
	plt.plot(y1)
	plt.title(label)
	plt.savefig('/media/htic/NewVolume1/murali/ecg/results/plots/' + str(count) + str(label) + '.jpg')
	#y2 = np.zeros(x2.shape)
	#plt.plot(x2,y2,'r*')
	#plt.show()


class RNN(nn.Module):

        def __init__(self):

                super(RNN,self).__init__()

                self.rnn = nn.LSTM(
                        input_size = INPUT_SIZE,
                        hidden_size = 128,
                        num_layers = 1,
                        batch_first = True, )


                self.out = nn.Linear(128,10)


        def forward(self,x):

                rout,(h_n,h_c) = self.rnn(x,None)
                out = self.out(rout[:,-1,:])

                return out



rnn = RNN()





path = '/media/htic/NewVolume1/murali/ecg/codes/ECG/datasets/mitdb/'
dat_path = path + '*.dat'
paths = glob(dat_path)
p2 = [path[:-4] for path in paths]

beat_annotations = ['N', 'L', 'R', 'B',
                    'A', 'a', 'J', 'S', 'V',
                    'r', 'F', 'e', 'j', 'n',
                    'E', '/', 'f', 'Q', '?'] 
# 0.5 of how much signal we want per beat
half_qrs = 120
base = pd.DataFrame()

tot_labels = []

count = 0

for path in tqdm(p2):
	
	ann = wf.rdann(path, 'atr')
	record = wf.rdsamp(path)
	labels = ann.symbol   

	beats = ann.sample
	len_beats = len(beats)
	
	#print len(beats),len(labels)
	# We want just the signals
	data = record.p_signals[:,0]
	print data.shape

	data_ranges = []

	for i in range(0,len_beats - 3 ,4):
		min_ = i
		max_ = i+3
		min_x = beats[min_]
		max_x = beats[max_]
		
	
		xran_min = min_x - half_qrs
		xran_max = max_x + half_qrs		

		#print xran_min,xran_max

		if (beats[-1] < xran_max) | (xran_min < beats[0]):
			continue	
		
		#print y_data.shape
		#print y_data
		label = '{}{}{}{}'.format(labels[i],labels[i+1],labels[i+2],labels[i+3])
		
			
		temp = xran_max - xran_min	
	
		data_ranges.append(temp)	
		y_data = data[xran_min: xran_max]
		y_data = np.concatenate([data[xran_min:xran_max],np.zeros(1300-temp)])
		
		y_data_tensor = torch.FloatTensor(

		print y_data.shape	
		#plot_graph(y_data,count,label)			
		count += 1

		print count
		#input("Press enter to continue")
		#if count == 5:
		#	break

	break

print min(data_ranges),max(data_ranges)

#x2=beats[648733:]
#beat_class = '{}_{}'.format(record.signame[0], ann)
#print (data.shape)
#print(beat_class.shape)

# Data display
#temp_data = data[:,0]
#print (temp_data.shape)
#plt.plot(temp_data[:500])
#plt.show()
#print(data)



#print set(tot_labels)
#print(y_data .shape, count)
#plot_graph(a[np.random.ra
