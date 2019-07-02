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
import time

#ours
from data_utils import create_dataset
from bayesian_utils import DKL_gaussian, sampler
from VariatonalModels import VAE_reconstruction, VAE_variational
from utils import move_gpu, anneal_lr, select_optimizer

#utilities
#from utils import select_optimizer,DKL_gaussian,sampler,normalize,likelihood,move_gpu

def parse_args():
	parser = argparse.ArgumentParser(description='Training Variational Distribution for Bayesian Neural Networks in Pytorch Enjoy!')
	'''String Variables'''
	parser.add_argument('--model_net', type=str,choices=['densenet-169','densenet-121','wide-resnet-40x10','wide-resnet-16x8','wide-resnet-28x10','resnet-18','resnet-50','resnet-101','vgg-19','resnext-29_8x16','preactresnet-164','preactresnet-18','dpn-92','mobilenet','senetB','vgg'],required=True,help='which model to train')
	parser.add_argument('--data_dir', type=str,required=True,help='where is the data')
	parser.add_argument('--dataset', type=str,choices=['gender','cifar10','svhn','cifar100','cars','birds','vggface2'],required=True,help='dataset to use')
	parser.add_argument('--MC_samples', type=int,required=True,help='Monte Carlo Samples to estimate ELBO')
	parser.add_argument('--dkl_after_epoch', type=int,required=True,help='use dkl term in ELBO after dkl_after_epochs epochs. Known as warm up see sonderby et al https://arxiv.org/abs/1602.02282')
	parser.add_argument('--dkl_scale_factor', type=str,required=True,help='scale factor of dkl. $\Beta$ in our paper.')
	parser.add_argument('--save_after', type=int,default=10,required=False,help='save the model after save_after epochs')
	parser.add_argument('--layer_dim', type=int,required=True,help='layer dimension')
	parser.add_argument('--n_layers', type=int,required=True,help='number of layers')
	parser.add_argument('--epochs', type=int,nargs='+',required=True,help='number of epochs')
	parser.add_argument('--lr', type=float,nargs='+',required=True,help='learning rate')
	parser.add_argument('--batch', type=int,required=True,help='batch size')
	parser.add_argument('--anneal', type=str,choices=['Linear',None],required=True,help='use Linear anneal in last lr')
	parser.add_argument('--n_gpu', type=int,default=0,help='which gpu to use')
	parser.add_argument('--folder_name', type=str,default=None,required=False,help='name of the folder where model is saved. If not provided automatic name is given.')

	aux=parser.parse_args()
	arguments=list()

	torch.cuda.set_device(aux.n_gpu)
	move_gpu(aux.n_gpu)

	arguments=[aux.model_net,aux.MC_samples,aux.dkl_scale_factor,aux.dkl_after_epoch,aux.data_dir,aux.dataset,aux.save_after,aux.epochs,aux.lr,aux.batch,aux.anneal,aux.folder_name,aux.layer_dim,aux.n_layers]
	return arguments


if __name__=='__main__':

	model_net,MC_samples,dkl_scale_factor,dkl_after_epoch,data_dir,dataset_name,save_after_epoch,epochs_t,lr_t,batch,anneal,folder_name,layer_dim,n_layers=parse_args()

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
	trainLoader,testLoader=create_dataset(dataset,batch_size_tr=batch,batch_size_te=batch,tocuda=True)

	#networks

	if dataset_name in ['cifar10','svhn']:
		input_dim=10
	elif dataset_name=='cifar100':
		input_dim=100
	elif dataset_name in ['gender','vggface2']:
		input_dim=2
	elif dataset_name == 'cars':
		input_dim = 196
	elif dataset_name == 'birds':
		input_dim = 200
	else:
		raise NotImplemented

	assert n_layers==2, "This code was used in pilot study and only supports 2 hidden layers."
	top=[layer_dim]*n_layers
	parameters=2*input_dim*layer_dim+layer_dim+input_dim + layer_dim**2+layer_dim*(n_layers-1)#number of total parameters

	reconstruction = VAE_reconstruction().cuda()
	variatonal = VAE_variational(parameters).cuda()

	topology='_'.join(str(e) for e in top)
 	if dataset_name in ['cifar10','svhn']:
		topology='10_'+topology+'_10'
		reconstruction.net=[10]
		reconstruction.net.extend(top+[10])
		variatonal.net=[10]
		variatonal.net.extend(top+[10])
	elif dataset_name=='cifar100':
		topology='100_'+topology+'_100'
		reconstruction.net=[100]
		reconstruction.net.extend(top+[100])
		variatonal.net=[100]
		variatonal.net.extend(top+[100])
	elif dataset_name in ['gender','vggface2']:
		topology='2_'+topology+'_2'
		reconstruction.net=[2]
		reconstruction.net.extend(top+[2])
		variatonal.net=[2]
		variatonal.net.extend(top+[2])
	elif dataset_name == 'cars':
		topology='196_'+topology+'_196'
		reconstruction.net=[196]
		reconstruction.net.extend(top+[196])
		variatonal.net=[196]
		variatonal.net.extend(top+[196])
	elif dataset_name == 'birds':
		topology='200_'+topology+'_200'
		reconstruction.net=[200]
		reconstruction.net.extend(top+[200])
		variatonal.net=[200]
		variatonal.net.extend(top+[200])
	else:
		raise NotImplemented

	#annealing and algorithm
		
	linear_anneal=False
	activate_anneal=False
	if anneal=='Linear':
		linear_anneal=True
	elif anneal==None:	
		pass
	else:
		raise NotImplemented

	#algorithm
	assert len(lr_t)==len(epochs_t),"Must provide lr_t and epochs_t with same length"
	torch.backends.cudnn.enabled=True
	train_parameters=[]
	train_parameters+=variatonal.parameters() 

	#for saving the model and logging
	counter=0
	directory="./pretrain_models/"+dataset_name+"/"+model_net+"/"+topology+"/"+str(counter)+"/"
	if folder_name==None:
		while True:
			if os.path.isdir(directory):
				counter+=1
				directory="./pretrain_models/"+dataset_name+"/"+model_net+"/"+topology+"/"+str(counter)+"/"
			else:
				break
	else:
		directory="./pretrain_models/"+dataset_name+"/"+model_net+"/"+topology+"/"+folder_name+"/"
	if os.path.exists(directory):
		print "Folder name {} already exists".format(directory)
		exit()

	model_dir = directory+'models/'
	log_dir =directory+'logs/'

	if not os.path.exists(model_dir):
		os.makedirs(model_dir)
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	
	logging.basicConfig(filename=log_dir+'train.log',level=logging.INFO)
	logging.info("Logger for model: {}".format(topology))
	logging.info("Training specificacitions: {}".format([epochs_t,lr_t,anneal]))
	logging.info("Stochastic Optimization specs: MC samples {} dkl_after_epoch {} scale factor {}".format(MC_samples,dkl_after_epoch,dkl_scale_factor))

	#for computing the scale factor of the dkl
	dkl_scale_factor = float(dkl_scale_factor)
	dkl_factor=False

	#useful variables
	torch_zero=Variable(torch.zeros(1,).cuda())	
	acc=0
	total_epoch=0

	#Start training
	for ind,(l,ep) in enumerate(zip(lr_t,epochs_t)):
		optimizer=select_optimizer(train_parameters,lr=l,optim='ADAM')
		'''
		Activate annealing
		'''
		if ind == len(epochs_t)-1 and linear_anneal:
                        activate_anneal=True
                        lr_init=l
                        epochs_N=ep

		for e in range(ep):
			if activate_anneal:
				lr_new=anneal_lr(lr_init,epochs_N,e)
				optimizer=select_optimizer(train_parameters,lr=lr_new,optim='ADAM')
			elbo_d=0.0
			MCtrain=0.0
			dkl_d=0.0
			llh_d=0.0
			total_train=0.0
			total_batch=0.0
			current_time=time.time()
			for batch_idx,(x,t) in enumerate(trainLoader):

				x,t=Variable(x),Variable(t.cuda())
			
				#parameter of variational distribution
				q_m,q_logv=variatonal.forward()

				if dkl_after_epoch < total_epoch:#warm up
					_DKL_vae = 1*DKL_gaussian(q_m,q_logv,torch_zero,torch_zero) 
					_DKL_vae=dkl_scale_factor*_DKL_vae.sum()#we sum as there is no batch dependency				
				else:
					_DKL_vae=torch_zero
	
				llh_CE_mc=0.0#Monte Carlo cross entropy	
				for i in range(MC_samples):
					sampled_params=sampler([q_m,q_logv],q_m.shape)#sample params from the variational distribution
					#reconstruction/likelihood term
					prediction=reconstruction.forward(x,sampled_params)
					llh_CE_mc += reconstruction.cost(t,prediction)				
				llh_CE_mc*=1/numpy.float32(MC_samples)#to approximate the expected value

				L_vae=llh_CE_mc+_DKL_vae#ELBO cost

				#we save stuff for printing after
				dkl_d += _DKL_vae.data
				llh_d += llh_CE_mc.data.sum()
				elbo_d+=L_vae.data.sum()

				COST=L_vae.mean()#ELBO cost average

				#gradient descent
				optimizer.zero_grad()
				COST.backward()
				optimizer.step()
				optimizer.zero_grad()

				#compute train error
				prediction=reconstruction.forward(x,sampled_params)
				_,index=torch.max(prediction,1)
				MCtrain+=(index.data==t.data).sum()
				
				total_train+=index.size(0)
				total_batch+=1
				
			MCvalid=0
			total_valid=0.0
			
			for x,t in testLoader: #this is validation accuracy. We estimate this accuracy only with one monte carlo from the predictive distribution
				x,t=Variable(x),t.cuda()
				q_m,q_logv=variatonal.forward()
				parameters=sampler([q_m,q_logv],q_m.shape)
				prediction=reconstruction.forward(x,parameters)
				_,index=torch.max(prediction,1)
				MCvalid+=(index.data==t).sum()
				total_valid+=index.size(0)

			#do the same for test data
			MCtest=0
			total_test=0.0
			q_m,q_logv=variatonal.forward()
			parameters=sampler([q_m,q_logv],q_m.shape)
			prediction=reconstruction.forward(Variable(logit_predicted_test),parameters)
			_,index=torch.max(prediction,1)
			MCtest+=(index.data==true_test).sum()
			total_test+=index.size(0)

			total_epoch+=1
			dkl_d=dkl_d.cpu().numpy()[0]
			
			logging.info("On epoch {} Monte Carlo ELBO train: {:.3f} train acc {:.3f} valid acc {:.3f} test acc {:.3f} LLH train {:.3f} DKL train {:.3f} of total train samples {} total valid samples {} total test samples {} took {:.3f} minutes".format(e,elbo_d/total_train,MCtrain/total_train*100,MCvalid/total_valid*100,MCtest/total_test*100,llh_d/total_train,dkl_d/total_batch,total_train,total_valid,total_test,(time.time()-current_time)/60.))

			print("On epoch {} Monte Carlo ELBO train: {:.3f} train acc {:.3f} valid acc {:.3f} test acc {:.3f} LLH train {:.3f} DKL train {:.3f} of total train samples {} total valid samples {} total test samples {} took {:.3f} minutes".format(e,elbo_d/total_train,MCtrain/total_train*100,MCvalid/total_valid*100,MCtest/total_test*100,llh_d/total_train,dkl_d/total_batch,total_train,total_valid,total_test,(time.time()-current_time)/60.))

			#save the models
			if acc==save_after_epoch:
				torch.save(reconstruction,model_dir+'reconstruction')
				torch.save(variatonal,model_dir+'variatonal') 
				acc=0
			else:
				acc+=1

	torch.save(reconstruction,model_dir+'reconstruction')
	torch.save(variatonal,model_dir+'variatonal') 
	
