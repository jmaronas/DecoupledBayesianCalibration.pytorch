import torch
import torch.utils.data as torchdata

if torch.__version__ != '0.3.1':
	raise RuntimeError('PyTorch version must be 0.3.1')

def create_dataset(dataset,batch_size_tr=100,batch_size_te=100,tocuda=False):
	tr,te=dataset
	tr_feat,te_feat=tr
	tr_lab,te_lab=te
	if tocuda:
		tr_feat=tr_feat.cuda()
		te_feat=te_feat.cuda()
		tr_lab=tr_lab.cuda()
		te_lab=te_lab.cuda()
	dataset_tr=torchdata.TensorDataset(tr_feat,tr_lab)		
	dataset_te=torchdata.TensorDataset(te_feat,te_lab)
	trLoader=torchdata.DataLoader(dataset_tr,batch_size=batch_size_tr,shuffle=True)
	teLoader=torchdata.DataLoader(dataset_te,batch_size=batch_size_te,shuffle=False)
	return trLoader,teLoader



