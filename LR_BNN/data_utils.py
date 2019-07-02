import torch.utils.data as torchdata

def create_dataset(dataset,tipe,batch_size_tr=100,batch_size_te=100,transforms=None,tag_transforms=None,tocuda=False,n_workers=0,shuffle_test=False):

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
	teLoader=torchdata.DataLoader(dataset_te,batch_size=batch_size_te,shuffle=shuffle_test)
	return trLoader,teLoader


