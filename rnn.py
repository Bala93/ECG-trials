import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms 
#import matplotlib.pyplot as plt


torch.manual_seed(1)

EPOCH = 10
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.01
DOWNLOAD_MNIST = True

train_data = dsets.MNIST(root='./mnist/',train=True,transform = transforms.ToTensor(),download = DOWNLOAD_MNIST)

print (train_data.train_data.size())
print (train_data.train_labels.size())


train_loader = torch.utils.data.DataLoader(dataset = train_data,batch_size = BATCH_SIZE,shuffle = True)

test_data = dsets.MNIST(root='./mnist/',train = False,transform = transforms.ToTensor())
test_x = Variable(test_data.test_data,volatile = True).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels.squeeze()[:2000]

test_x = test_x.cuda()
test_y = test_y.cuda()

class RNN(nn.Module):

	def __init__(self):
		
		super(RNN,self).__init__()

		self.rnn = nn.LSTM(
			input_size = INPUT_SIZE,
			hidden_size = 64,
			num_layers = 1,
			batch_first = True, )


		self.out = nn.Linear(64,10)


	def forward(self,x):

		rout,(h_n,h_c) = self.rnn(x,None)
		out = self.out(rout[:,-1,:])

		return out


rnn = RNN().cuda()
print (rnn)

optimizer = torch.optim.Adam(rnn.parameters(),lr = LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):

	for step,(x,y) in enumerate(train_loader):
		b_x = Variable(x.view(-1,28,28).cuda())	
		b_y = Variable(y.cuda())

		output = rnn(b_x)
		loss   = loss_func(output,b_y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if step % 50 == 0:
			
			test_output = rnn(test_x)
			pred_y = torch.max(test_output,1)[1].data
			compare = (pred_y == test_y)
	 
			accuracy = torch.sum(pred_y == test_y)/float(test_y.size()[0])

			print ( 'Epoch: ', epoch, ' | train loss : %.6f' % loss.data[0], '| test accuracy: %.4f' % accuracy)


		test_output = rnn(test_x[:10].view(-1,28,28))
		pred_y = torch.max(test_output,1)[1].data.squeeze()



		#print (pred_y,'prediction number')
		#print (test_y[:10],'real number')









