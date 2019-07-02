#torch
import torch
if torch.__version__ != '0.3.1':
	raise RuntimeError('PyTorch version must be 0.3.1')
import torch.nn as nn
from torch.autograd import Variable
torch.manual_seed(seed=1)
torch.cuda.manual_seed(seed=1)

#python
import os
import logging
import numpy
import argparse

#ours
from data_utils import create_dataset
from VariatonalModels import VAE_reconstruction, VAE_variational
from calibration_utils import average_confidence,accuracy,accuracy_per_bin,average_confidence_per_bin,compute_ECE
from bayesian_utils import sampler


def parse_args():
	parser = argparse.ArgumentParser(description='Predictive Distribution by  Enjoy!')
	parser.add_argument('--dataset', type=str,choices=['gender','cifar10','svhn'],required=True,help='dataset to use')
	parser.add_argument('--model_net', type=str,choices=['densenet-169','densenet-121','wide-resnet-40x10','wide-resnet-28x10','wide-resnet-16x8','resnet-18','resnet-50','resnet-101','resnext-29_8x16','preactresnet-164','preactresnet-18','vgg-19','dpn-92'],required=True,help='which model to train')
	parser.add_argument('--model_dir', type=str,required=True,help='Directory to the trained model')
	parser.add_argument('--data_dir', type=str,required=True,help='Directory to data')
	parser.add_argument('--valid_test', type=str,choices=['validation','test'],required=True,help='whether to test or validate model')
	parser.add_argument('--MC_samples', type=int,required=True,help='how many samples for inference in test')

	aux=parser.parse_args()
	arguments=list()
	arguments=[aux.MC_samples,aux.model_dir,aux.data_dir,aux.dataset,aux.model_net,aux.valid_test]
	return arguments

def anneal_lr(lr_init,epochs_N,e):
        lr_new=-(lr_init/epochs_N)*e+lr_init
        return lr_new

if __name__=='__main__':

	MC_samples,model_dir,data_dir,dataset_name,model_net,valid_test=parse_args()

	#load dataset into data loader
	logit_predicted_test=torch.from_numpy(numpy.load(data_dir+model_net+"_"+dataset_name+'/'+dataset_name+"_"+model_net+"_logit_prediction_test.npy")).cuda()
	logit_predicted_valid=torch.from_numpy(numpy.load(data_dir+model_net+"_"+dataset_name+'/'+dataset_name+"_"+model_net+"_logit_prediction_valid.npy")).cuda()
	true_test=torch.from_numpy(numpy.load(data_dir+model_net+"_"+dataset_name+'/'+dataset_name+"_"+model_net+"_true_test.npy")).long().cuda()
	true_valid=torch.from_numpy(numpy.load(data_dir+model_net+"_"+dataset_name+'/'+dataset_name+"_"+model_net+"_true_valid.npy")).long().cuda()
	tr=[logit_predicted_test,logit_predicted_valid]
	te=[true_test,true_valid]
	dataset=[tr,te]
	testLoader,validLoader=create_dataset(dataset,batch_size_tr=10000,batch_size_te=5000,tocuda=True)

	#networks
	variatonal = torch.load(model_dir+"models/"+"variatonal",map_location=lambda storage,loc:storage.cuda(0))
	reconstruction = torch.load(model_dir+"models/"+"reconstruction",map_location=lambda storage,loc:storage.cuda(0))
	q_m,q_logv=variatonal.forward()
 
	'''
	Measuring calibration
	'''
	n_bins=15
	if valid_test=='validation':
		#utilities
		CE=nn.functional.cross_entropy
		#before calibration
		avg_confidence_before_BNN_valid=average_confidence(logit_predicted_valid)*100
		acc_before_BNN_valid=accuracy(logit_predicted_valid,true_valid)*100

		accuracy_bin,prob,samples_per_bin=accuracy_per_bin(logit_predicted_valid,true_valid,n_bins)
		conf_bin,prob,samples_per_bin=average_confidence_per_bin(logit_predicted_valid,n_bins)
		ece_before_BNN_valid=compute_ECE(accuracy_bin,conf_bin,samples_per_bin)*100

		predicted_afterBNN=0.0
		print "ECE for {} bins without calibration getting {} with average confidence {} and accuraccy {}".format(n_bins,ece_before_BNN_valid,avg_confidence_before_BNN_valid,acc_before_BNN_valid)
		ECE_best=100000000
		diff_best=100000000
		sample_ECE = 0
		sample_diff = 0

		for i in range(1,MC_samples+1,1):

			llh_mc=0.0

			parameters=sampler([q_m,q_logv],q_m.shape)
			prediction=reconstruction.forward(Variable(logit_predicted_valid),parameters)
			prediction=torch.nn.functional.softmax(prediction)
			predicted_afterBNN+=prediction.data#in this case we average the probabilities given by the model

			
			aux=(predicted_afterBNN/float(i))

			avg_confidence_BNN_valid=average_confidence(aux,apply_softmax=False)*100
			acc_BNN_valid=accuracy(aux,true_valid,apply_softmax=False)*100

			accuracy_bin,prob,samples_per_bin=accuracy_per_bin(aux,true_valid,n_bins,apply_softmax=False)
			conf_bin,prob,samples_per_bin=average_confidence_per_bin(aux,n_bins,apply_softmax=False)
			ece_after_BNN_valid=compute_ECE(accuracy_bin,conf_bin,samples_per_bin)*100
		
			if ece_after_BNN_valid < ECE_best:
				ECE_best=ece_after_BNN_valid
				sample_ECE = i
			if  abs(acc_BNN_valid-avg_confidence_BNN_valid)<diff_best:
				diff_best = abs(acc_BNN_valid-avg_confidence_BNN_valid)
				sample_diff = i
			print "ECE for {} bins and MC samples {} CALIBRATED getting {} with average confidence {} and accuraccy {}. Difference getting {}".format(n_bins,i,ece_after_BNN_valid,avg_confidence_BNN_valid,acc_BNN_valid,abs(acc_BNN_valid-avg_confidence_BNN_valid))

		print "Best value with  bins {}  ECE {} with sample {}".format(n_bins,ECE_best,sample_ECE)
	
		
	
	else:
		#before calibration
		avg_confidence_before_BNN_test=average_confidence(logit_predicted_test)*100
		acc_before_BNN_test=accuracy(logit_predicted_test,true_test)*100

		accuracy_bin,prob,samples_per_bin=accuracy_per_bin(logit_predicted_test,true_test,n_bins)
		conf_bin,prob,samples_per_bin=average_confidence_per_bin(logit_predicted_test,n_bins)
		ece_before_BNN_test=compute_ECE(accuracy_bin,conf_bin,samples_per_bin)*100

		q_m,q_logv=variatonal.forward()

		predicted_afterBNN=0.0
		for i in range(1,MC_samples+1,1):
			parameters=sampler([q_m,q_logv],q_m.shape)
			prediction=reconstruction.forward(Variable(logit_predicted_test),parameters)
			prediction=torch.nn.functional.softmax(prediction)
			predicted_afterBNN+=prediction.data	
			
			
		predicted_afterBNN *= 1/float(MC_samples)
		predicted_afterBNN=predicted_afterBNN.cpu()
		true_test=true_test.cpu()
		'''
		After calibration
		'''
		avg_confidence_BNN_valid=average_confidence(predicted_afterBNN.numpy(),apply_softmax=False)*100
		acc_BNN_valid=accuracy(predicted_afterBNN.numpy(),true_test.numpy(),apply_softmax=False)*100
	
		accuracy_bin,prob,samples_per_bin=accuracy_per_bin(predicted_afterBNN.numpy(),true_test.numpy(),n_bins,apply_softmax=False)
		conf_bin,prob,samples_per_bin=average_confidence_per_bin(predicted_afterBNN.numpy(),n_bins,apply_softmax=False)

		ece_after_BNN_valid=compute_ECE(accuracy_bin,conf_bin,samples_per_bin)*100


		print "ECE for {} bins and MC samples {} CALIBRATED getting {} with average confidence {} and accuraccy {}. Difference getting {}".format(n_bins,i,ece_after_BNN_valid,avg_confidence_BNN_valid,acc_BNN_valid,abs(acc_BNN_valid-avg_confidence_BNN_valid))

	        

