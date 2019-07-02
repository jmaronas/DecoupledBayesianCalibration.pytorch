#torch
import torch

if torch.__version__ != '0.3.1':
	raise RuntimeError('PyTorch version must be 0.3.1')
from torch.autograd import Variable
import torch.nn as nn

#python
import math

epsilon=Variable(torch.ones(1,).cuda())*1e-11
pi=Variable(torch.ones(1,).cuda())*float(math.pi)

def move_gpu(gpu_i):
        global epsilon
        global pi
        epsilon=epsilon.cuda(gpu_i)
        pi=pi.cuda(gpu_i)

def anneal_lr(lr_init,epochs_N,e):
        lr_new=-(lr_init/epochs_N)*e+lr_init
        return lr_new

def select_optimizer(parameters,lr=0.0,mmu=0.0,optim='SGD'):
	if optim=='SGD':
		optimizer = torch.optim.SGD(parameters,lr=lr,momentum=mmu)
	elif optim=='ADAM':
		optimizer = torch.optim.Adam(parameters,lr=lr)

	return optimizer

def return_activation(act):
	if act=='relu':
		return nn.ReLU()
	elif act=='softmax':
		return nn.Softmax()
	else:
		raise NotImplemented
