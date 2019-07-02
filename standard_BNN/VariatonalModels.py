import torch
if torch.__version__ != '0.3.1':
	raise RuntimeError('PyTorch version must be 0.3.1')


import torch.nn as nn
from torch.autograd import Variable

from utils import return_activation

class VAE_variational(nn.Module):
	def __init__(self,parameters):
		super(VAE_variational, self).__init__()
		self.mean=nn.Parameter(torch.zeros((parameters,)).cuda().normal_(0,1))
		self.logvar=nn.Parameter(torch.zeros((parameters,)).cuda().normal_(0,1))
	def forward(self):		
		return self.mean,self.logvar


class VAE_reconstruction(nn.Module):
	def __init__(self):
		super(VAE_reconstruction, self).__init__()
		
		self.softmax=return_activation('softmax')
		self.relu = return_activation('relu')
		self.ce_cost=torch.nn.functional.cross_entropy
		self.net=0

	def forward(self,x,parameters):
		n=self.net
		w1=parameters[0:n[0]*n[1]].contiguous().view(n[0],n[1])
		b1=parameters[n[0]*n[1]:n[0]*n[1]+n[1]].contiguous().view(n[1],)

		w2=parameters[n[0]*n[1]+n[1]:n[0]*n[1]+n[1]+n[1]*n[2]].contiguous().view(n[1],n[2])
		b2=parameters[n[0]*n[1]+n[1]+n[1]*n[2]:n[0]*n[1]+n[1]+n[1]*n[2]+n[2]].contiguous().view(n[2],)

		w3=parameters[n[0]*n[1]+n[1]+n[1]*n[2]+n[2]:n[0]*n[1]+n[1]+n[1]*n[2]+n[2]+n[2]*n[3]].view(n[2],n[3])
		b3=parameters[n[0]*n[1]+n[1]+n[1]*n[2]+n[2]+n[2]*n[3]:n[0]*n[1]+n[1]+n[1]*n[2]+n[2]+n[2]*n[3]+n[3]].view(n[3],)

		l1=self.relu(Variable.mm(x,w1)+b1)
		l2=self.relu(Variable.mm(l1,w2)+b2)
		prediction=Variable.mm(l2,w3)+b3
		return prediction

	def cost(self,t,prediction):
		return self.ce_cost(prediction,t,reduce=False)

#optimized versions
class VAE_variational_optimized(nn.Module):
	def __init__(self,parameters):
		super(VAE_variational_optimized, self).__init__()
	def reshape(self):
		#for optimizing performance we preallocate the members on where we will fill with samples
		mean_list=nn.ParameterList()
		var_list=nn.ParameterList()
		for i in range(len(self.net)-1):
			l1=self.net[i]
			l2=self.net[i+1]
			mean_list.extend([nn.Parameter(torch.zeros((l1,l2)).cuda().normal_(0,1)),nn.Parameter(torch.zeros((l2,)).cuda().normal_(0,1))])
			var_list.extend([nn.Parameter(torch.zeros((l1,l2)).cuda().normal_(0,1)),nn.Parameter(torch.zeros((l2,)).cuda().normal_(0,1))])

		self.mean_list=mean_list
		self.logvar_list=var_list
	def forward(self):
		return self.mean_list,self.logvar_list


class VAE_reconstruction_optimized(nn.Module):
	def __init__(self):
		super(VAE_reconstruction_optimized, self).__init__()
		self.softmax=return_activation('softmax')
		self.relu = return_activation('relu')
		self.ce_cost=torch.nn.functional.cross_entropy

	def forward(self,x,parameters):
		n=self.net
		for i in range(0,len(parameters)-2,2):
			w,b=parameters[i:i+2]
			x=self.relu(Variable.mm(x,w)+b)
		w,b=parameters[-2],parameters[-1]
		prediction=Variable.mm(x,w)+b
		return prediction
	def cost(self,t,prediction):
		return self.ce_cost(prediction,t,reduce=False)
