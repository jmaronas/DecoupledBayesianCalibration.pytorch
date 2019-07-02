import torch
import math
import os
import shutil
if torch.__version__ != '0.4.0':
        raise RuntimeError('PyTorch version must be 0.4.0')

epsilon=torch.ones(1,)*1e-11
pi=torch.ones(1,)*float(math.pi)


def select_optimizer(parameters,lr=0.1,mmu=0.9,optim='SGD'):
	if optim=='SGD':
		optimizer = torch.optim.SGD(parameters,lr=lr,momentum=mmu)
	elif optim=='ADAM':
		optimizer = torch.optim.Adam(parameters,lr=lr)
	return optimizer

def anneal_lr(lr_init,epochs_N,e):
        lr_new=-(lr_init/epochs_N)*e+lr_init
        return lr_new
def move_gpu(gpu_i):
        global epsilon
        global pi
        epsilon=epsilon.cuda(gpu_i)
        pi=pi.cuda(gpu_i)


def add_nan_file(directory):
	#this function adds a .nandetected file so we can easily monitorize if our experiment suffer from numerical inestability
	#it creates in argument given by directory
	if not os.path.isdir(directory):
		raise Exception("Path to {} no exists".format(directory))
	open(directory+'.nandetected','w').close()


def save_checkpoint(state, is_best, directory, filename='checkpoint.pth.tar'):
    torch.save(state, directory+filename)
    if is_best:
        shutil.copyfile(directory+filename, directory+'model_best.pth.tar')

def add_experiment_notfinished(directory):
	#this function adds a .expfinished file so we can easily monitorize if our model finished correctly
	#it creates in argument given by directory
	if not os.path.isdir(directory):
		raise Exception("Path to {} no exists".format(directory))
	open(directory+'.expnotfinished','w').close()

def remove_experiment_notfinished(directory):
	if not os.path.isdir(directory):
		raise Exception("Path to {} no exists".format(directory))
	if not os.path.isfile(directory+".expnotfinished"):
		raise Exception("No file .expnotfinished in {}".format(directory))
	os.remove(directory+".expnotfinished")

def load_checkpoint(directory,filename):
        if os.path.isfile(directory+filename):
                checkpoint=torch.load(directory+filename,map_location='cpu')
                return checkpoint
        else:
                print ("File not found at {}".format(directory+filename))
                exit(-1)

