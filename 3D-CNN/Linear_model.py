import torch
import torch.nn as nn

""" Construct model """
class Model_Linear(nn.Module):
	def __init__(self, use_cuda=True):
		super(Model_Linear, self).__init__()     
		self.use_cuda = use_cuda
		
		self.fc1 = nn.Linear(2048, 100)
		torch.nn.init.normal_(self.fc1.weight, 0, 1)
		self.dropout1 = nn.Dropout(0.0)
		self.fc1_bn = nn.BatchNorm1d(num_features=100, affine=True, momentum=0.3).train()
		self.fc2 = nn.Linear(100, 1)
		torch.nn.init.normal_(self.fc2.weight, 0, 1)
		self.relu = nn.ReLU()

	def forward(self, x):
		fc1_z = self.fc1(x)
		fc1_y = self.relu(fc1_z)
		fc1_d = self.dropout1(fc1_y)
		fc1 = self.fc1_bn(fc1_d) if fc1_d.shape[0]>1 else fc1_d  
		fc2_z = self.fc2(fc1)
		return fc2_z, fc1_z
