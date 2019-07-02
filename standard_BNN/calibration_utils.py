import torch

if torch.__version__ != '0.3.1':
	raise RuntimeError('PyTorch version must be 0.3.1')

from torch.nn.functional import softmax
from torch.autograd import Variable
import numpy


#compute average confidence
def average_confidence(predicted,apply_softmax=True):

	if type(predicted) is numpy.ndarray and predicted.dtype==numpy.float32:
		predicted=Variable(torch.from_numpy(predicted))
	elif type(predicted) is torch.FloatTensor or type(predicted) is torch.cuda.FloatTensor:
		predicted=Variable(predicted)
	else:
		print "Either torch.FloatTensor or numpy.ndarray type float32 expected"
		exit(-1)

	if apply_softmax:
		predicted_prob=softmax(predicted,dim=1).data
	else:
		predicted_prob=predicted.data

	predicted_prob,index = torch.max(predicted_prob,1)

	return predicted_prob.sum()/float(predicted_prob.shape[0])


def accuracy(predicted,real_tag,apply_softmax=True):
	
	if type(predicted) is numpy.ndarray and predicted.dtype==numpy.float32:
		predicted=Variable(torch.from_numpy(predicted))
	elif type(predicted) is torch.FloatTensor  or type(predicted) is torch.cuda.FloatTensor:
		predicted=Variable(predicted)
	else:
		print "Either torch.FloatTensor or numpy.ndarray type float32 expected"
		exit(-1)
	
	if type(real_tag) is numpy.ndarray and real_tag.dtype==numpy.int64:
		real_tag=torch.from_numpy(real_tag)
	elif type(real_tag) is torch.LongTensor or type(real_tag) is torch.cuda.LongTensor:
		pass 
	else:
		print "Either torch.LongTensor or numpy.ndarray type int64 expected"
		exit(-1)
	
	if apply_softmax:
		predicted_prob=softmax(predicted,dim=1).data
	else:
		predicted_prob=predicted.data

	accuracy,index = torch.max(predicted_prob,1)
	selected_label=index==real_tag
	return selected_label.sum()/float(selected_label.shape[0])


def accuracy_per_bin(predicted,real_tag,n_bins=10,apply_softmax=True):

	if type(predicted) is numpy.ndarray and predicted.dtype==numpy.float32:
		predicted=Variable(torch.from_numpy(predicted))
	elif type(predicted) is torch.FloatTensor or type(predicted) is torch.cuda.FloatTensor:
		predicted=Variable(predicted)
	else:
		print "Either torch.FloatTensor or numpy.ndarray type float32 expected"
		exit(-1)

	if type(real_tag) is numpy.ndarray and real_tag.dtype==numpy.int64 :
		real_tag=torch.from_numpy(real_tag)
	elif type(real_tag) is torch.LongTensor or type(real_tag) is torch.cuda.LongTensor:
		pass
	else:
		print "Either torch.LongTensor or numpy.ndarray type int64 expected"
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
			acc[p]=label_sel.sum()/float(len(label_sel))

		samples_per_bin.append(len(label_sel))

	samples_per_bin=numpy.array(samples_per_bin)
	acc=acc[0:-1]
	prob=prob[0:-1]
	return acc,prob,samples_per_bin


def average_confidence_per_bin(predicted,n_bins=10,apply_softmax=True):

	if type(predicted) is numpy.ndarray and predicted.dtype==numpy.float32:
		predicted=Variable(torch.from_numpy(predicted))
	elif type(predicted) is torch.FloatTensor or type(predicted) is torch.cuda.FloatTensor:
		predicted=Variable(predicted)
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
			conf[p]=prob_sel.sum()/float(len(prob_sel))

		samples_per_bin.append(len(prob_sel))

	samples_per_bin=numpy.array(samples_per_bin)
	conf=conf[0:-1]
	prob=prob[0:-1]

	return conf,prob,samples_per_bin

def compute_ECE(acc_bin,conf_bin,samples_per_bin):
	assert len(acc_bin)==len(conf_bin)
	ece=0.0
	total_samples = float(samples_per_bin.sum())

	for samples,acc,conf in zip(samples_per_bin,acc_bin,conf_bin):
		ece+=samples/total_samples*numpy.abs(acc-conf)

	return ece



def reliability_histogram(prob,acc,show=0,save=None,ece=None):
	assert len(prob)==len(acc)
	n_bins=len(prob)
	aux=numpy.linspace(0,1,n_bins+1)
	plt.bar(prob,acc,1/float(n_bins),align='edge',label='calibration',edgecolor=[0,0,0])
	plt.plot(aux,aux,'r',label='perfect calibration')
	plt.ylim((0,1))
	plt.xlim((0,1))
	plt.legend(fontsize=12)
	plt.xlabel('Confidence',fontsize=14,weight='bold')
	plt.ylabel('Accuracy',fontsize=14,weight='bold')

	props = dict(boxstyle='square', facecolor='lightblue', alpha=0.9)
	# place a text box in upper left in axes coords
	textstr = '$ECE=%.3f$'%(ece*100)
	plt.text(0.65, 0.05, textstr,weight='bold', fontsize=20,
        verticalalignment='bottom', bbox=props)
	plt.tick_params(axis='both',labelsize=12)
	if show:
		plt.show()
	else:
		if save !=None and type(save)==str:
			plt.savefig(save+'_'+str(n_bins)+'.png')
	plt.close()

def compute_MCE(acc_bin,conf_bin,samples_per_bin):
	assert len(acc_bin)==len(conf_bin)
	mce=0.0
	sample=0
	total_samples = float(samples_per_bin.sum())
	for i in range(len(samples_per_bin)):
		a=samples_per_bin[i]/total_samples*numpy.abs(acc_bin[i]-conf_bin[i])
		
		if a>mce:
			mce=a
			sample=i

	return mce,sample

	
