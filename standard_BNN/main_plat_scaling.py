import  torch
import torch.nn as nn
from torch.autograd import Variable
import numpy

from data_utils import create_dataset
from calibration_utils import  average_confidence,accuracy,accuracy_per_bin,average_confidence_per_bin,compute_ECE

import argparse
def parse_args():
	parser = argparse.ArgumentParser(description='Plat Scaling by anonymus. Resarcher at anonymus anonymus@anonymus.es . Enjoy!')
	'''String Variables'''

	parser.add_argument('--model_net', type=str,choices=['wide-resnet-40x10','wide-resnet-28x10','wide-resnet-16x8','densenet-169','densenet-121','resnet-18','resnet-50','resnet-101','preactresnet-18','preactresnet-164','vgg-19','resnext-29_8x16','dpn-92','mobilenet','senetB','vgg'],required=True,help='which model to train.')
	parser.add_argument('--data_dir', type=str,required=True,help='where is the datasets')
	parser.add_argument('--dataset', type=str,choices=['gender','cifar10','cifar100','svhn','vggface2','cars','birds'],required=True,help='dataset to use')
	parser.add_argument('--T_factor', type=float,required=False,default=None,help='if provided perform test else run validation')
	parser.add_argument('--epochs', type=int,required=False,default=1,help='Number of epochs')
	aux=parser.parse_args()
        arguments=[aux.model_net,aux.data_dir,aux.dataset,aux.T_factor,aux.epochs]
	return arguments


model_net,data_dir,dataset_name,T_factor,epochs=parse_args()

logit_predicted_test=torch.from_numpy(numpy.load(data_dir+model_net+"_"+dataset_name+'/'+dataset_name+"_"+model_net+"_logit_prediction_test.npy")).cuda()
logit_predicted_valid=torch.from_numpy(numpy.load(data_dir+model_net+"_"+dataset_name+'/'+dataset_name+"_"+model_net+"_logit_prediction_valid.npy")).cuda()
true_test=torch.from_numpy(numpy.load(data_dir+model_net+"_"+dataset_name+'/'+dataset_name+"_"+model_net+"_true_test.npy")).long().cuda()
true_valid=torch.from_numpy(numpy.load(data_dir+model_net+"_"+dataset_name+'/'+dataset_name+"_"+model_net+"_true_valid.npy")).long().cuda()

#negative log likelihood
CE_loss=nn.functional.cross_entropy
softmax=nn.functional.softmax

#for training parameter T
T_plat=Variable(torch.ones(1,).cuda(),requires_grad=True)
n_bins=15

#create dataset 
dataset=[[logit_predicted_test,logit_predicted_valid],[true_test,true_valid]]
testLoader,validationLoader=create_dataset(dataset,batch_size_tr=100,batch_size_te=1000,tocuda=True)

#optimization
parameters= []
parameters=[T_plat]
optim = torch.optim.SGD(parameters, lr = 0.1,momentum=0.0)

#values for uncalibrated data
avg_confidence_beforePLAT_valid=average_confidence(logit_predicted_valid)*100
acc_beforePLAT_valid=accuracy(logit_predicted_valid,true_valid)*100

accuracy_bin,prob,samples_per_bin=accuracy_per_bin(logit_predicted_valid,true_valid,n_bins)
conf_bin,prob,samples_per_bin=average_confidence_per_bin(logit_predicted_valid,n_bins)
ece_before_algorithm_valid=compute_ECE(accuracy_bin,conf_bin,samples_per_bin)*100

if T_factor==None:

	for ep in range(epochs):
		MC_valid=0.
		acc_loss=0.
		for batch_idx,(x,t) in enumerate(validationLoader):
	
			x,t=Variable(x),Variable(t.cuda())
			z=x/T_plat
		
			loss=CE_loss(z,t)
			loss.backward()
			optim.step()
			optim.zero_grad()
			acc_loss += loss.data

		logit_aux=(logit_predicted_valid/T_plat.data)

		accuracy_bin,prob,samples_per_bin=accuracy_per_bin(logit_aux,true_valid,n_bins)
		conf_bin,prob,samples_per_bin=average_confidence_per_bin(logit_aux,n_bins)

		avg_confidence_afterPLAT=average_confidence(logit_aux)*100
		acc_afterPLAT=accuracy(logit_aux,true_valid)*100

		ece_loss=compute_ECE(accuracy_bin,conf_bin,samples_per_bin)*100
		acc_loss=acc_loss.cpu().numpy()

		print "On epoch {}: Ece uncalib [{}] calib [{}] cross entropy [{}] with T [{}]".format(ep,ece_before_algorithm_valid,ece_loss,acc_loss[0],T_plat.data.cpu().numpy()[0])
		print "Before Temp-scal===>Avg conf {} Acc {} || After Temp-scal====> Avg conf [{}] Acc [{}]".format(avg_confidence_beforePLAT_valid,acc_beforePLAT_valid,avg_confidence_afterPLAT,acc_afterPLAT)
		print "--------------------------"
	T_factor=T_plat.data

else:
	pass

######################################
######################################
#######################################
#############VALIDATION DATA###########
#######################################
'''Calibration Over Validation data'''

#validation without calibration
avg_confidence=average_confidence(logit_predicted_valid)*100
acc=accuracy(logit_predicted_valid,true_valid)*100
	
accuracy_bin,prob,samples_per_bin=accuracy_per_bin(logit_predicted_valid,true_valid,n_bins)
conf_bin,prob,samples_per_bin=average_confidence_per_bin(logit_predicted_valid,n_bins)
ece_before=compute_ECE(accuracy_bin,conf_bin,samples_per_bin)*100

print "====> Validation set Uncalibrated"
print  "Getting and ECE on valid set for " +model_net+ " for dataset "+dataset_name+" of {} for {} bins ".format(ece_before,n_bins)
print  "Getting accuracy of {}".format(acc)
print  "Getting average confidence of {}".format(avg_confidence)
print  "Difference {}".format(abs(avg_confidence-acc))

#validation with calibration
logit_aux=logit_predicted_valid/T_factor

accuracy_bin,prob,samples_per_bin=accuracy_per_bin(logit_aux,true_valid,n_bins)
conf_bin,prob,samples_per_bin=average_confidence_per_bin(logit_aux,n_bins)

avg_confidence_afterPLAT=average_confidence(logit_aux)*100
acc_afterPLAT=accuracy(logit_aux,true_valid)*100

ece_loss=compute_ECE(accuracy_bin,conf_bin,samples_per_bin)*100

print "====> Valid set Calibrated"

print  "Getting and ECE on valid set for " +model_net+ " for dataset "+dataset_name+" of {} for {} bins ".format(ece_loss,n_bins)
print  "Getting accuracy of {}".format(acc_afterPLAT)
print  "Getting average confidence of {}".format(avg_confidence_afterPLAT)
print  "Difference {}".format(abs(acc_afterPLAT-avg_confidence_afterPLAT))

#######################################
#######################################
#######################################
##########TEST DATA####################
######################################
######################################
######################################

'''Calibration Over Test data'''

################
#uncalibrated
avg_confidence=average_confidence(logit_predicted_test)*100
acc=accuracy(logit_predicted_test,true_test)*100
accuracy_bin,prob,samples_per_bin=accuracy_per_bin(logit_predicted_test,true_test,n_bins)
conf_bin,prob,samples_per_bin=average_confidence_per_bin(logit_predicted_test,n_bins)
ece_before=compute_ECE(accuracy_bin,conf_bin,samples_per_bin)*100

print "====> Test set Uncalibrated"
print  "Getting and ECE on test set for " +model_net+ " for dataset "+dataset_name+" of {} for {} bins ".format(ece_before,n_bins)
print  "Getting accuracy of {}".format(acc)
print  "Getting average confidence of {}".format(avg_confidence)
print  "Difference {}".format(abs(acc-avg_confidence))


################################
#calibrated

logit_aux=logit_predicted_test/T_factor
accuracy_bin,prob,samples_per_bin=accuracy_per_bin(logit_aux,true_test,n_bins)
conf_bin,prob,samples_per_bin=average_confidence_per_bin(logit_aux,n_bins)
avg_confidence_afterPLAT=average_confidence(logit_aux)*100
acc_afterPLAT=accuracy(logit_aux,true_test)*100
ece_loss=compute_ECE(accuracy_bin,conf_bin,samples_per_bin)*100

print "====> Test set Calibrated"
print  "Getting and ECE on test set for " +model_net+ " for dataset "+dataset_name+" of {} for {} bins ".format(ece_loss,n_bins)
print  "Getting accuracy of {}".format(acc_afterPLAT)
print  "Getting average confidence of {}".format(avg_confidence_afterPLAT)
print  "Difference {}".format(abs(acc_afterPLAT-avg_confidence_afterPLAT))











	
	
