#coding:utf-8-
import torch
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F

x1 = np.array([1,1,0,
               1,1,0,
               0,0,0])

x2 = np.array([1,1,1,
               1,0,1,
               1,1,1])

x3 = np.array([0,1,0,
               1,1,1,
               0,1,0])

x4 = np.array([0,0,0,
               0,1,1,
               0,1,1])

x5 = np.array([1,0,0,
               1,0,0,
               1,0,0])

x6 = np.array([0,0,0,
               1,1,1,
               0,0,0])

x7 = np.array([0,0,1,
               0,0,1,
               0,0,1])

x8 = np.array([1,0,1,
               0,1,0,
               1,0,1])

x = np.vstack([x1,x2,x3,x4,x5,x6,x7,x8])

same_class=[]
diff_class=[]
for i in range(len(x)):
	for j in range(len(x)):
		if i==j:
			same_class.append(np.hstack([x[i],x[j]]))
		else:
			diff_class.append(np.hstack([x[i],x[j]]))

same_class = np.array(same_class)
diff_class = np.array(diff_class)
print same_class.shape
print diff_class.shape

data=np.vstack([same_class,diff_class])
label=np.vstack([np.ones((len(same_class),1)),np.zeros((len(diff_class),1))])
print data.shape
print label.shape


y=Variable(torch.FloatTensor(label))


class relation(nn.Module):
	def __init__(self,input_size,hidden_size,output_size):
		super(relation, self).__init__()
		self.linear0 = nn.Linear(9, 1)
		self.linear1 = nn.Linear(input_size,hidden_size)
		self.linear2 = nn.Linear(hidden_size,output_size)
	def forward(self,x1,x2):
		out1=F.relu(self.linear0(x1))
		out2=F.relu(self.linear0(x2))
		h=torch.cat([out1,out2],1)
		out1=F.relu(self.linear1(h))
		out=F.sigmoid(self.linear2(out1))
		return h,out

net=relation(2,3,1)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

for epoch in range(1000):
	optimizer.zero_grad()
	x1 = Variable(torch.FloatTensor(data[:, 0:9]+np.random.randn(*np.shape(data[:,0:9]))*0.05))
	x2 = Variable(torch.FloatTensor(data[:, 9:]+np.random.randn(*np.shape(data[:,9:]))*0.05))
	
	_,outputs =net(x1,x2)  # forward
	loss = criterion(outputs, Variable(torch.FloatTensor(label)))  # loss
	loss.backward()  # backward
	if epoch % 100 == 0:
		print(loss.data[0])
	optimizer.step()

h_value,o= net(Variable(torch.FloatTensor(data[:, 0:9])),Variable(torch.FloatTensor(data[:, 9:])))
print h_value.size()
print o
print h_value.data
plt.scatter(h_value[:,0], h_value[:,1], c=label.flatten())
plt.colorbar()
plt.show()