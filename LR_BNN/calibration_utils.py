#torch
import torch
if torch.__version__ != '0.4.0':
        raise RuntimeError('PyTorch version must be 0.4.0')
from torch.nn.functional import softmax

#python
import numpy

def compute_ECE(acc_bin,conf_bin,samples_per_bin):
	assert len(acc_bin)==len(conf_bin)
	ece=0.0
	total_samples = float(samples_per_bin.sum())

	ece_list=[]
	for samples,acc,conf in zip(samples_per_bin,acc_bin,conf_bin): 
		ece_list.append(samples/total_samples*numpy.abs(acc-conf))
		ece+=samples/total_samples*numpy.abs(acc-conf)
	return ece

def accuracy_per_bin(predicted,real_tag,n_bins=10,apply_softmax=True):

	if type(predicted) is numpy.ndarray and predicted.dtype==numpy.float32:
		predicted=Variable(torch.from_numpy(predicted))
	elif type(predicted) is torch.Tensor and predicted.dtype is torch.float32:
		pass
	else:
		raise Exception( "Either torch.FloatTensor or numpy.ndarray type float32 expected")
		exit(-1)

	if type(real_tag) is numpy.ndarray and real_tag.dtype==numpy.int64 :
		real_tag=torch.from_numpy(real_tag)
	elif type(real_tag) is torch.Tensor and real_tag.dtype is torch.int64:
		pass
	else:
		raise Exception("Either torch.LongTensor or numpy.ndarray type int64 expected")
		exit(-1)

	if apply_softmax:
		predicted_prob=softmax(predicted,dim=1).data
	else:
		predicted_prob=predicted.data
	

	accuracy,index = torch.max(predicted_prob,1)
	selected_label=index.long()==real_tag

	prob=numpy.linspace(0,1,n_bins+1)
	acc=numpy.linspace(0,1,n_bins+1)
	total_data = len(accuracy)
	samples_per_bin=[]
	for p in range(len(prob)-1):
		#find elements with probability in between p and p+1
		min_=prob[p]
		max_=prob[p+1]
		boolean_upper = accuracy<=max_

		if p==0:#we include the first element in bin
			boolean_down=accuracy>=min_
		else:#after that we included in the previous bin
			boolean_down=accuracy>min_

		index_range=boolean_down & boolean_upper
		label_sel=selected_label[index_range]
		
		if len(label_sel)==0:
			acc[p]=0.0
		else:
			acc[p]=label_sel.sum().float()/float(len(label_sel))

		samples_per_bin.append(len(label_sel))

	samples_per_bin=numpy.array(samples_per_bin)
	acc=acc[0:-1]
	prob=prob[0:-1]
	return acc,prob,samples_per_bin

def average_confidence_per_bin(predicted,n_bins=10,apply_softmax=True):

	if type(predicted) is numpy.ndarray and predicted.dtype==numpy.float32:
		predicted=torch.from_numpy(predicted)
	elif type(predicted) is torch.Tensor and predicted.dtype is torch.float32:
		pass
	else:
		print "Either torch.FloatTensor or numpy.ndarray type float32 expected"
		exit(-1)

	if apply_softmax:
		predicted_prob=softmax(predicted,dim=1).data
	else:
		predicted_prob=predicted.data
	

	prob=numpy.linspace(0,1,n_bins+1)
	conf=numpy.linspace(0,1,n_bins+1)
	accuracy,index = torch.max(predicted_prob,1)

	samples_per_bin=[]

	for p in range(len(prob)-1):
		#find elements with probability in between p and p+1
		min_=prob[p]
		max_=prob[p+1]
		
		boolean_upper = accuracy<=max_

		if p==0:#we include the first element in bin
			boolean_down =accuracy>=min_
		else:#after that we included in the previous bin
			boolean_down =accuracy>min_

		index_range=boolean_down & boolean_upper
		prob_sel=accuracy[index_range]
		
		if len(prob_sel)==0:
			conf[p]=0.0
		else:
			conf[p]=prob_sel.sum().float()/float(len(prob_sel))

		samples_per_bin.append(len(prob_sel))

	samples_per_bin=numpy.array(samples_per_bin)
	conf=conf[0:-1]
	prob=prob[0:-1]

	return conf,prob,samples_per_bin


#compute average confidence
def average_confidence(predicted,apply_softmax=True):
	#migrated to version 0.4.0	
	if type(predicted) is numpy.ndarray and predicted.dtype==numpy.float32:
		predicted=torch.from_numpy(predicted)
	elif type(predicted) is torch.Tensor and predicted.dtype is torch.float32:
		pass
	else:
		raise Exception("Either torch.FloatTensor or numpy.ndarray type float32 expected, got {}".format(predicted.type()))

	if apply_softmax:
		predicted_prob=softmax(predicted,dim=1).data
	else:
		predicted_prob=predicted.data

	predicted_prob,index = torch.max(predicted_prob,1)
	
	return predicted_prob.sum().float()/float(predicted_prob.shape[0])


def accuracy(predicted,real_tag,apply_softmax=True):
	#migrated to version 0.4.0
	if type(predicted) is numpy.ndarray and predicted.dtype==numpy.float32:
		predicted=torch.from_numpy(predicted)
	elif type(predicted) is torch.Tensor  and predicted.dtype is torch.float32:
		pass
	else:
		raise Exception("Either torch.FloatTensor or numpy.ndarray type float32 expected")
		exit(-1)
	
	if type(real_tag) is numpy.ndarray and real_tag.dtype==numpy.int64:
		real_tag=torch.from_numpy(real_tag)
	elif type(real_tag) is torch.Tensor and real_tag.dtype is torch.int64:
		pass 
	else:
		raise Exception("Either torch.LongTensor or numpy.ndarray type int64 expected")
		exit(-1)
	


	if apply_softmax:
		predicted_prob=softmax(predicted,dim=1).data
	else:
		predicted_prob=predicted.data


	accuracy,index = torch.max(predicted_prob,1)
	selected_label=index==real_tag

	return selected_label.sum().float()/float(selected_label.shape[0])
