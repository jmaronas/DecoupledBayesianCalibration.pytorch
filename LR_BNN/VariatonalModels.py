# -*- coding: utf-8 -*-
import torch
import torch
if torch.__version__ != '0.4.0':
	raise RuntimeError('PyTorch version must be 0.4.0')
import torch.nn as nn

import os
import numpy

from NNets_utils import return_activation, apply_linear
from bayesian_utils import DKL_gaussian_optimized

####################################
####################################
####################################
####################################

class variatonal_local_reparametrization(nn.Module): #local_reparametrization
	def __init__(self,topology,batch,learn_prior):
		super(variatonal_local_reparametrization, self).__init__()
		self.softmax=return_activation('softmax')
		self.relu = return_activation('relu')
		self.ce_cost=torch.nn.functional.cross_entropy
		self.batch=batch
		self.topology=topology
		self.learn_prior=learn_prior
		self.zero_list=[]

		self.torch_sampling=[]

		weight_list_mean=[]
		weight_list_variance=[]
		bias_list_mean=[]#we can implement the variance reduction with only sampling from a matrix made up from same parameters
		bias_list_var=[]#we can implement the variance reduction with only sampling from a matrix made up from same parameters


		if self.learn_prior:
                        prior_weight_list_mean=[]                                                     
                        prior_weight_list_variance=[]                                                 
                        prior_bias_list_mean=[]                                                       
                        prior_bias_list_var=[]

		for idx in range(0,len(topology)-1):
			w=nn.Parameter(torch.zeros((topology[idx],topology[idx+1])).cuda().normal_(0,1))
			weight_list_mean.append(w)
			w=nn.Parameter(torch.zeros((topology[idx],topology[idx+1])).cuda().normal_(0,1))
			weight_list_variance.append(w)

			self.torch_sampling.append(torch.zeros((batch,topology[idx+1])).cuda())#for allocate memory only once

			bias_for_optim=nn.Parameter(torch.zeros(topology[idx+1],).cuda())
			bias_list_mean.append(bias_for_optim)
			bias_for_optim=nn.Parameter(torch.zeros(topology[idx+1],).cuda())
			bias_list_var.append(bias_for_optim)

			if self.learn_prior:                                                          
                                #our starting point is the standard normal distribution
                                w=nn.Parameter(torch.zeros((topology[idx],topology[idx+1])).cuda())   
                                prior_weight_list_mean.append(w)
                                w=nn.Parameter(torch.zeros((topology[idx],topology[idx+1])).cuda())   
                                prior_weight_list_variance.append(w)                                  
                                bias_for_optim=nn.Parameter(torch.zeros(topology[idx+1],).cuda())
                                prior_bias_list_mean.append(bias_for_optim)
                                bias_for_optim=nn.Parameter(torch.zeros(topology[idx+1],).cuda())
                                prior_bias_list_var.append(bias_for_optim)

			else:
				self.zero_list.append(torch.tensor(0.0).cuda())

		self.variatonal_bias_list_mean=nn.ParameterList(bias_list_mean)
		self.variatonal_bias_list_logvar=nn.ParameterList(bias_list_var)
		self.variatonal_weight_list_mean=nn.ParameterList(weight_list_mean)
		self.variatonal_weight_list_logvar=nn.ParameterList(weight_list_variance)
	

		if self.learn_prior:
                        self.prior_bias_list_mean=nn.ParameterList(prior_bias_list_mean)
                        self.prior_bias_list_logvar=nn.ParameterList(prior_bias_list_var)
                        self.prior_weight_list_mean=nn.ParameterList(prior_weight_list_mean)
                        self.prior_weight_list_logvar=nn.ParameterList(prior_weight_list_variance)	


	def sample(self,mu,var,index):
		epsilon=self.torch_sampling[index].normal_(0,1).data#samples from the normal distribution
		x=epsilon*var.sqrt()+mu#local reparameterized
		return x


	def forward(self,x):
		pos_activation=x

		for index in range(len(self.variatonal_weight_list_mean)-1):
			w_mean,w_logvar,b_mean,b_logvar=self.variatonal_weight_list_mean[index],self.variatonal_weight_list_logvar[index],self.variatonal_bias_list_mean[index],self.variatonal_bias_list_logvar[index]
		

			mu=torch.mm(pos_activation,w_mean)

			w_var=torch.exp(w_logvar)
			pos_act=pos_activation**2
			var=torch.mm(pos_act,w_var)

			x_linear_projection=self.sample(mu,var,index)#this function returns a sample from the local reparameterized weights

			#now we sample the variational bias and add to preactivation
			b_var=torch.exp(b_logvar)#bvar is possitive

			pre_activation=self.sample(b_mean,b_var,index) + x_linear_projection
			
			#apply activation
			pos_activation=self.relu(pre_activation)			

		#last forward out of loop as it has linear activation
		w_mean,w_logvar,b_mean,b_logvar=self.variatonal_weight_list_mean[-1],self.variatonal_weight_list_logvar[-1],self.variatonal_bias_list_mean[-1],self.variatonal_bias_list_logvar[-1]

		w_var=torch.exp(w_logvar)
		b_var=torch.exp(b_logvar)
	
		mu=torch.mm(pos_activation,w_mean)

		pos_act=pos_activation**2
		var=torch.mm(pos_act,w_var)

		x_linear_projection=self.sample(mu,var,-1)#this function returns a sample from the local reparameterized weights

		#now we sample the variational bias and add to preactivation
		linear_activation=self.sample(b_mean,b_var,-1) + x_linear_projection

		return linear_activation


	def forward_test(self,x):
		if x.size(0)!=self.batch:
			aux_sampler=self.torch_sampling
			self.__realloc__(x.size(0))
		out=self.forward(x)
		if x.size(0)!=self.batch:
			self.torch_sampling=aux_sampler
		return out

	def __realloc__(self,batch):
		self.torch_sampling=[]
		for idx in range(0,len(self.topology)-1):
			self.torch_sampling.append(torch.zeros((batch,self.topology[idx+1])).cuda())#for allocate memory only once

	def cost(self,x,t,MC_samples,dkl_scale_factor):
		if t.size(0)!=self.batch:
			aux_sampler=self.torch_sampling
			self.__realloc__(t.size(0))
		#Evaluate the cost of the ELBO estimator: E_q [log p(t|x,w)] - DKL
	
		if not self.learn_prior:
                        #Kullback lieber divergence assuming gaussian priors
                        DKL_bias=DKL_gaussian_optimized(self.variatonal_bias_list_mean,self.variatonal_bias_list_logvar,self.zero_list,self.zero_list,reduce_batch_dim=True,reduce_sample_dim=True)
                        DKL_weight=DKL_gaussian_optimized(self.variatonal_weight_list_mean,self.variatonal_weight_list_logvar,self.zero_list,self.zero_list,reduce_batch_dim=True,reduce_sample_dim=True)

                else:
                        #Kullback lieber divergence learning priors
                        DKL_bias=DKL_gaussian_optimized(self.variatonal_bias_list_mean,self.variatonal_bias_list_logvar,self.prior_bias_list_mean,self.prior_bias_list_logvar,reduce_batch_dim=True,reduce_sample_dim=True)
                        DKL_weight=DKL_gaussian_optimized(self.variatonal_weight_list_mean,self.variatonal_weight_list_logvar,self.prior_bias_list_mean,self.prior_bias_list_logvar,reduce_batch_dim=True,reduce_sample_dim=True)
	
		DKL_cost=dkl_scale_factor*(DKL_bias+DKL_weight)

		
		LLH_mc=torch.zeros(t.shape).cuda()
		#likelihood 
		for mc in range(MC_samples):
			out=self.forward(x)	
			LLH_mc+=self.ce_cost(out,t,reduce=False)

		LLH_mc*=1/numpy.float32(MC_samples)

	
		ELBO=LLH_mc+DKL_cost
		
		if t.size(0)!=self.batch:
			self.torch_sampling=aux_sampler

		return ELBO,DKL_cost,LLH_mc

	def predictive_sampler(self):
		self.torch_sampling=[]
		
		for index in range(len(self.topology)-1):

			l1=self.topology[index]
			l2=self.topology[index+1]
			self.torch_sampling.extend([torch.zeros((l1,l2)).cuda(),torch.zeros((l2,)).cuda()])

	def predictive_sample(self):
		self.sampled_params=[]
		counter=0
		for index in range(len(self.variatonal_weight_list_mean)):
			w_mean,w_logvar,b_mean,b_logvar=self.variatonal_weight_list_mean[index],self.variatonal_weight_list_logvar[index],self.variatonal_bias_list_mean[index],self.variatonal_bias_list_logvar[index]
			w_var=torch.exp(w_logvar)
			b_var=torch.exp(b_logvar)

			w=self.sample(w_mean,w_var,counter)
			b=self.sample(b_mean,b_var,counter+1)

			counter+=2
			self.sampled_params.append((w,b))

	def predictive_forward(self,x):
		x_=x
		for (w,b) in self.sampled_params[0:-1]:
			x_=self.relu(x_.mm(w)+b)

		w,b=self.sampled_params[-1]
		
		a=x_.mm(w)+b
		return self.softmax(x_.mm(w)+b)

