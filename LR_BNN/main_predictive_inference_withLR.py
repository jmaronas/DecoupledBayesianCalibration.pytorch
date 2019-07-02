#torch
import torch
if torch.__version__ != '0.4.0':
	raise RuntimeError('PyTorch version must be 0.4.0')
import torch.nn as nn
torch.manual_seed(seed=1)
torch.cuda.manual_seed(seed=1)
from torch.nn.functional import softmax

#python
import os
import numpy
import argparse
import sys

#ours
from utils import load_checkpoint,move_gpu

from calibration_utils import compute_ECE,accuracy_per_bin,average_confidence_per_bin,accuracy,average_confidence
from data_utils import create_dataset
from VariatonalModels import variatonal_local_reparametrization

def parse_args():
	parser = argparse.ArgumentParser(description='Training Variational Distribution for Bayesian Neural Networks Enjoy!')
	'''String Variables'''
	parser.add_argument('--model_net', type=str,choices=['senetA','senetB','vgg','mobilenet','densenet-121','densenet-169','dpn-92','preactresnet-18','preactresnet-164','resnet-18','resnet-50','resnet-101','resnext-29_8x16','vgg-19','wide-resnet-28x10','wide-resnet-40x10','wide-resnet-16x8'],required=True,help='which model to train')
	parser.add_argument('--data_dir', type=str,required=True,help='where is the data')
	parser.add_argument('--dataset', type=str,choices=['cifar10','cifar100','svhn','gender','birds','cars','vggface2'],required=True,help='dataset to use')
	parser.add_argument('--n_gpu', type=int,default=0,help='which gpu to use')
	parser.add_argument('--model_dir',type=str,required=True,help='bnn model dir to be loaded')
	parser.add_argument('--valid_test',type=str,choices=['test','valid'],required=True,default='test')
	parser.add_argument('--layer_dim', type=int,required=True,help='layer dimension')
	parser.add_argument('--n_layers', type=int,required=True,help='number of layers')
	parser.add_argument('--prior_is_learnable', required=False,action='store_true',help='make the prior learnable')
	parser.add_argument('--MCsamples',type=int,required=True,help='predictive monte carlo samples')
	
	args=parser.parse_args()

	torch.cuda.set_device(args.n_gpu)
	move_gpu(args.n_gpu)

	return args

if __name__=='__main__':
	args=parse_args()
	data_dir,model_net,dataset_name,model_dir=args.data_dir,args.model_net,args.dataset,args.model_dir

	#load dataset into data loader	
	logit_predicted_valid=torch.from_numpy(numpy.load(data_dir+model_net+"_"+dataset_name+'/'+dataset_name+"_"+model_net+"_logit_prediction_valid.npy")).float().cuda()
	logit_predicted_test=torch.from_numpy(numpy.load(data_dir+model_net+"_"+dataset_name+'/'+dataset_name+"_"+model_net+"_logit_prediction_test.npy")).float().cuda()
	true_valid=torch.from_numpy(numpy.load(data_dir+model_net+"_"+dataset_name+'/'+dataset_name+"_"+model_net+"_true_valid.npy")).long().cuda()
	true_test=torch.from_numpy(numpy.load(data_dir+model_net+"_"+dataset_name+'/'+dataset_name+"_"+model_net+"_true_test.npy")).long().cuda()


	tr=[logit_predicted_valid,logit_predicted_test]
	te=[true_valid,true_test]
	dataset=[tr,te]
	trainLoader,validLoader=create_dataset(dataset,'FC',batch_size_tr=1000,batch_size_te=1000,tocuda=True)


	#networks
	if dataset_name in ['cifar10','svhn']:
		input_dim=10
	elif dataset_name=='cifar100':
		input_dim=100
	elif dataset_name in ['gender','vggface2']:
		input_dim=2
	elif dataset_name=='cars':
		input_dim=196
	elif dataset_name=='birds':
		input_dim=200
	else:
		raise NotImplemented

	layer_dim,n_layers=args.layer_dim,args.n_layers
	top=[input_dim]+[layer_dim]*n_layers+[input_dim]
	
	BNN=variatonal_local_reparametrization(top,1000,args.prior_is_learnable)

	params=load_checkpoint(args.model_dir,"")

	BNN.load_state_dict(params)

	BNN.cuda()


	n_bins=15

	if args.valid_test=='valid':
		'''
		Before calibration	
		'''
		avg_confidence_beforeBNN_valid=average_confidence(logit_predicted_valid)*100
		acc_beforeBNN_valid=accuracy(logit_predicted_valid,true_valid)*100

		accuracy_bin,prob,samples_per_bin=accuracy_per_bin(logit_predicted_valid,true_valid,n_bins)
		conf_bin,prob,samples_per_bin=average_confidence_per_bin(logit_predicted_valid,n_bins)
		ece_before_BNN_valid=compute_ECE(accuracy_bin,conf_bin,samples_per_bin)*100


		print "ECE for {} bins unCALIBRATED getting {} with average confidence {} and accuraccy {}. Difference getting {}".format	(n_bins,ece_before_BNN_valid,avg_confidence_beforeBNN_valid,acc_beforeBNN_valid,abs(acc_beforeBNN_valid.float()-avg_confidence_beforeBNN_valid.float()))


		BNN.predictive_sampler()

		predicted_afterBNN=0.0
		ECE_best=numpy.inf
		sample_ECE=0
		for i in range(1,args.MCsamples+1,1):
		

			BNN.predictive_sample()
			prediction=BNN.predictive_forward(logit_predicted_valid).data
			predicted_afterBNN+=prediction

			
			'''
			evaluate on this sample
			'''

			aux=(predicted_afterBNN/float(i))
			

			avg_confidence_BNN_valid=average_confidence(aux,apply_softmax=False)*100
			acc_BNN_valid=accuracy(aux,true_valid,apply_softmax=False)*100


			accuracy_bin,prob,samples_per_bin=accuracy_per_bin(aux,true_valid,n_bins,apply_softmax=False)
			conf_bin,prob,samples_per_bin=average_confidence_per_bin(aux,n_bins,apply_softmax=False)
			ece_after_BNN_valid=compute_ECE(accuracy_bin,conf_bin,samples_per_bin)*100
			
			if ece_after_BNN_valid < ECE_best:
				ECE_best=ece_after_BNN_valid
				sample_ECE = i

			print "ECE for {} bins and MC samples {} CALIBRATED getting {} with average confidence {} and accuraccy {}. Difference getting {}".format(n_bins,i,ece_after_BNN_valid,avg_confidence_BNN_valid,acc_BNN_valid,abs(acc_BNN_valid.float()-avg_confidence_BNN_valid.float()))


		print "Best value with  bins {}  ECE {} with sample {}".format(n_bins,ECE_best,sample_ECE)
		
	
	else:
		BNN.predictive_sampler()
		predicted_afterBNN=0.0
		for i in range(1,args.MCsamples+1,1):
		

			BNN.predictive_sample()
			prediction=BNN.predictive_forward(logit_predicted_test).data
			predicted_afterBNN+=prediction


		predicted_afterBNN*=1/float(args.MCsamples)

		avg_confidence_BNN_valid=average_confidence(predicted_afterBNN,apply_softmax=False)*100
		acc_BNN_valid=accuracy(predicted_afterBNN,true_test,apply_softmax=False)*100

		accuracy_bin,prob,samples_per_bin=accuracy_per_bin(predicted_afterBNN,true_test,n_bins,apply_softmax=False)
		conf_bin,prob,samples_per_bin=average_confidence_per_bin(predicted_afterBNN,n_bins,apply_softmax=False)
		ece_after_BNN_valid=compute_ECE(accuracy_bin,conf_bin,samples_per_bin)*100

		print "ECE for {} bins and MC samples {} CALIBRATED getting {} with average confidence {} and accuraccy {}. Difference getting {}".format(n_bins,i,ece_after_BNN_valid,avg_confidence_BNN_valid,acc_BNN_valid,abs(acc_BNN_valid.float()-avg_confidence_BNN_valid.float()))


