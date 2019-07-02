# -*- coding: utf-8 -*-
#torch
import torch
if torch.__version__ != '0.4.0':
	raise RuntimeError('PyTorch version must be 0.4.0')
import torch.nn as nn
torch.manual_seed(seed=1)
torch.cuda.manual_seed(seed=1)

#python
import os
import logging
import numpy
import argparse
import time

#ours

from utils import move_gpu,anneal_lr,select_optimizer,save_checkpoint, add_nan_file,add_experiment_notfinished,remove_experiment_notfinished
from data_utils import create_dataset
from VariatonalModels import variatonal_local_reparametrization


def parse_args():
	parser = argparse.ArgumentParser(description='Training Variational Distribution for Bayesian Neural Networks with Local Reparameterization. Enjoy!')
	'''String Variables'''
	parser.add_argument('--model_net', type=str,choices=['wide-resnet-28x10','wide-resnet-40x10','wide-resnet-16x8','densenet-121','densenet-169','preactresnet-18','preactresnet-164','dpn-92','resnet-18','resnet-50','resnet-101','resnext-29_8x16','vgg-19','vgg','senetB','mobilenet'],required=True,help='which model to train')
	parser.add_argument('--data_dir', type=str,required=True,help='where is the data')
	parser.add_argument('--dataset', type=str,choices=['cifar10','cifar100','svhn','gender','birds','cars','vggface2'],required=True,help='dataset to use')
	parser.add_argument('--MC_samples', type=int,required=True,help='Monte Carlo Samples to estimate ELBO')
	parser.add_argument('--layer_dim', type=int,required=True,help='layer dimension')
	parser.add_argument('--n_layers', type=int,required=True,help='number of layers')
	parser.add_argument('--save_after', type=int,default=10,required=False,help='save the model after save_after epochs')
	parser.add_argument('--save_model_every',default=None,type=int, required=False,help='save a model each 100 epochs')
	parser.add_argument('--epochs', type=int,nargs='+',required=True,help='number of epochs')
	parser.add_argument('--lr', type=float,nargs='+',required=True,help='learning rate')
	parser.add_argument('--batch', type=int,required=True,help='batch size')
	parser.add_argument('--anneal', type=str,choices=['Linear',None],required=True,help='use Linear anneal in last lr')
	parser.add_argument('--n_gpu', type=int,default=0,help='which gpu to use')
	parser.add_argument('--folder_name', type=str,default=None,required=False,help='name of the folder where model is saved. If not provided automatic name is given.')
	parser.add_argument('--dkl_scale_factor', type=float,default=1.0,required=False,help='scale dkl')
	parser.add_argument('--dkl_after_epochs', type=float,default=-1,required=False,help='warm up')
	parser.add_argument('--prior_is_learnable', required=False,action='store_true',help='make the prior learnable')
	args=parser.parse_args()
	torch.cuda.set_device(args.n_gpu)
	move_gpu(args.n_gpu)

	return args


if __name__=='__main__':

	args=parse_args()
	data_dir,model_net,dataset_name,batch=args.data_dir,args.model_net,args.dataset,args.batch
	#load dataset into data loader	
	logit_predicted_train=torch.from_numpy(numpy.load(data_dir+model_net+"_"+dataset_name+'/'+dataset_name+"_"+model_net+"_logit_prediction_train.npy")).cuda()
	logit_predicted_valid=torch.from_numpy(numpy.load(data_dir+model_net+"_"+dataset_name+'/'+dataset_name+"_"+model_net+"_logit_prediction_valid.npy")).cuda()
	logit_predicted_test=torch.from_numpy(numpy.load(data_dir+model_net+"_"+dataset_name+'/'+dataset_name+"_"+model_net+"_logit_prediction_test.npy")).cuda()
	true_train=torch.from_numpy(numpy.load(data_dir+model_net+"_"+dataset_name+'/'+dataset_name+"_"+model_net+"_true_train.npy")).long().cuda()
	true_valid=torch.from_numpy(numpy.load(data_dir+model_net+"_"+dataset_name+'/'+dataset_name+"_"+model_net+"_true_valid.npy")).long().cuda()
	true_test=torch.from_numpy(numpy.load(data_dir+model_net+"_"+dataset_name+'/'+dataset_name+"_"+model_net+"_true_test.npy")).long().cuda()

	tr=[logit_predicted_train,logit_predicted_valid]
	te=[true_train,true_valid]
	dataset=[tr,te]
	trainLoader,validLoader=create_dataset(dataset,'FC',batch_size_tr=batch,batch_size_te=batch,tocuda=True)
	tr=[logit_predicted_train,logit_predicted_test]
	te=[true_train,true_test]
	dataset=[tr,te]
	_,testLoader=create_dataset(dataset,'FC',batch_size_tr=batch,batch_size_te=batch,tocuda=True)

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

	
	BNN = variatonal_local_reparametrization(top,args.batch,args.prior_is_learnable).cuda()

	#annealing and algorithm	
	linear_anneal=False
	activate_anneal=False
	if args.anneal=='Linear':
		linear_anneal=True
	elif args.anneal==None:	
		pass
	else:
		raise NotImplemented

	#algorithm
	assert len(args.lr)==len(args.epochs),"Must provide lr_t and epochs_t with same length"
	torch.backends.cudnn.enabled=True
	train_parameters=[]
	train_parameters+=BNN.parameters() 

	#for computing the scale factor of the dkl
	dkl_scale=args.dkl_scale_factor
	dkl_after_epoch=args.dkl_after_epochs

	#useful variables
	acc=0
	total_epoch=0
	t_eps=0
	save_after_epoch=args.save_after

	#for saving the model and logging
	topology='_'.join(str(e) for e in top)
	if args.prior_is_learnable:
		directory="./pretrain_models_learnprior/"+dataset_name+"/"+model_net+"/"+topology+"/"+args.folder_name+"/"
	else:
		directory="./pretrain_models/"+dataset_name+"/"+model_net+"/"+topology+"/"+args.folder_name+"/"
	if os.path.exists(directory):
		print "Folder name {} already exists".format(directory)
		exit()
	log_dir =directory+'logs/'
	model_dir=directory+'/models/'
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)
	add_experiment_notfinished(directory)#create this file to monitor errors in computer platform
	logging.basicConfig(filename=log_dir+'train.log',level=logging.INFO)
	logging.info("Logger for model: {}".format(topology))
	logging.info("Training specificacitions: {}".format([args.epochs,args.lr,args.anneal]))
	logging.info("Stochastic Optimization specs: MC samples {} dkl_after_epoch {} scale factor {}".format(args.MC_samples,dkl_after_epoch,dkl_scale))

	#Start training
	for ind,(l,ep) in enumerate(zip(args.lr,args.epochs)):
		optimizer=select_optimizer(train_parameters,lr=l,optim='ADAM')
		'''
		Activate annealing
		'''
		if ind == len(args.epochs)-1 and linear_anneal:
                        activate_anneal=True
                        lr_init=l
                        epochs_N=ep

		for e in range(ep):
			#annealing
			if activate_anneal:
				lr_new=anneal_lr(lr_init,epochs_N,e)
				optimizer=select_optimizer(train_parameters,lr=lr_new,optim='ADAM')

			elbo_d,dkl_d,llh_d,MCtrain,total_train,total_batch,MCvalid,total_valid,MCtest,total_test=[0.0]*10

			current_time=time.time()
			for batch_idx,(x,t) in enumerate(trainLoader):
				x,t=x,t.cuda()
			
				#train with warm up
				dkl_scale_factor= dkl_scale if dkl_after_epoch < total_epoch else 0.0
				
				ELBO,DKL,LLH_mc=BNN.cost(x,t,args.MC_samples,dkl_scale_factor)
				
				#we save stuff for printing after
				dkl_d += DKL.data
				llh_d += LLH_mc.data.sum()
				elbo_d+=ELBO.data.sum()

				COST=ELBO.mean()#ELBO cost average
				
				#gradient descent
				optimizer.zero_grad()
				COST.backward()
				optimizer.step()

				#compute train error
				aux=BNN.torch_sampling
				prediction=BNN.forward_test(x)
				_,index=torch.max(prediction,1)
				MCtrain+=(index.data==t.data).sum()
				
				total_train+=index.size(0)
				total_batch+=1
				
			
			for x,t in validLoader: #this is validation accuracy. We estimate this accuracy only with one monte carlo from the predictive distribution
				x,t=x,t.cuda()
				prediction=BNN.forward_test(x)
				_,index=torch.max(prediction,1)
				MCvalid+=(index.data==t).sum()
				total_valid+=index.size(0)

			#do the same for test data

			for x,t in testLoader: #this is validation accuracy. We estimate this accuracy only with one monte carlo from the predictive distribution
				x,t=x,t.cuda()
				prediction=BNN.forward_test(x)
				_,index=torch.max(prediction,1)
				MCtest+=(index.data==t).sum()
				total_test+=index.size(0)

			total_epoch+=1
			
			dkl_d=dkl_d.cpu().numpy()
			if torch.isnan(elbo_d):
				save_checkpoint(BNN.state_dict(),False,model_dir,filename='BNN.pth.tar')
				add_nan_file(directory)
				exit(-1)
			logging.info("On epoch {} Monte Carlo ELBO train: {:.3f} train acc {:.3f} valid acc {:.3f} test acc {:.3f} LLH train {:.3f} DKL train {:.3f} of total train samples {} total valid samples {} total test samples {} took {:.3f} minutes".format(e,elbo_d/total_train,float(MCtrain)/total_train*100,float(MCvalid)/total_valid*100,float(MCtest)/total_test*100,llh_d/total_train,dkl_d/total_batch,total_train,total_valid,total_test,(time.time()-current_time)/60.))

			print("On epoch {} Monte Carlo ELBO train: {:.3f} train acc {:.3f} valid acc {:.3f} test acc {:.3f} LLH train {:.3f} DKL train {:.3f} of total train samples {} total valid samples {} total test samples {} took {:.3f} minutes".format(e,elbo_d/total_train,float(MCtrain)/total_train*100,float(MCvalid)/total_valid*100,float(MCtest)/total_test*100,llh_d/total_train,dkl_d/total_batch,total_train,total_valid,total_test,(time.time()-current_time)/60.))

			#save the models
			if args.save_model_every==None:
				if acc==save_after_epoch:
					save_checkpoint(BNN.state_dict(),False,model_dir,filename='BNN.pth.tar')
					acc=1
				else:
					acc+=1

			if args.save_model_every!=None:
				if acc==args.save_model_every:
					model_dir_n = directory+str(t_eps)+'/models/'
					os.makedirs(model_dir_n)
					save_checkpoint(BNN.state_dict(),False,model_dir_n,filename='BNN_epoch'+str(t_eps)+'.pth.tar')
					acc=1
				else:
					acc+=1
			t_eps+=1



	save_checkpoint(BNN.state_dict(),False,model_dir,filename='BNN.pth.tar')
	remove_experiment_notfinished(directory)#remove the file if everything finished correctly
	exit(0)
