import torch

if torch.__version__ != '0.3.1':
	raise RuntimeError('PyTorch version must be 0.3.1')
from torch.autograd import Variable

def DKL_gaussian(mean_q,logvar_q,mean_p,logvar_p):
	#computes the DKL(q(x)//p(x)) per gaussian of each sample in a batch. Returns (batch,DKL)
	var_p = torch.exp(logvar_p)
	var_q = torch.exp(logvar_q)
	DKL=0.5 * (-1 + logvar_p - logvar_q + (var_q/var_p) + torch.pow(mean_p-mean_q,2)/var_p )
	return DKL

def DKL_gaussian_optimized(mean_q,logvar_q,mean_p,logvar_p,reduce=True,aslist=False):
	#computes the DKL(q(x)//p(x)) per gaussian of each sample in a batch. Returns (batch,DKL)
	alogvar_q=logvar_q
	amean_q=mean_q 
	var_p = torch.exp(logvar_p)
	DKL=0.0
	for mean_q,logvar_q in zip(amean_q,alogvar_q):
		var_q = torch.exp(logvar_q)
		DKL+=0.5 * (-1 + logvar_p - logvar_q + (var_q/var_p) + torch.pow(mean_p-mean_q,2)/var_p).sum()
	return DKL

def sampler(param,shape):
	#sample from a gaussian distribution of a given shape
	mean,logvar=param
	std = torch.exp(logvar*0.5)
	sampler=torch.zeros(shape).cuda()
	sample=Variable(sampler.normal_(0,1))
	sample=sample*std+mean
	return sample

def sampler_optimized(param,monte_carlo_sampling_list):
	mean,logvar=param
	samples_list=list()
	for index in range(len(monte_carlo_sampling_list)):
		m,logv=mean[index],logvar[index]
		std = torch.exp(logv*0.5)
		monte_carlo_sampling_list[index].normal_(0,1)
		samples_list.append(Variable(monte_carlo_sampling_list[index])*std+m)
			
	return samples_list
