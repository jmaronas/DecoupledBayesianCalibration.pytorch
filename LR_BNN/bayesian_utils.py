import torch
if torch.__version__ != '0.4.0':
        raise RuntimeError('PyTorch version must be 0.4.0')


#same as above but usefull for bayesian neural networks
def DKL_gaussian_optimized(mean_q,logvar_q,mean_p,logvar_p,reduce_batch_dim=False,reduce_sample_dim=False):
        alogvar_q=logvar_q
        amean_q=mean_q

        alogvar_p=logvar_p
        amean_p=mean_p
	
        if reduce_batch_dim and reduce_sample_dim:
                DKL=0.0
                for mean_q,logvar_q,mean_p,logvar_p in zip(amean_q,alogvar_q,amean_p,alogvar_p):
                        var_q = torch.exp(logvar_q)
                        var_p = torch.exp(logvar_p)
                        DKL+=0.5 * (-1 + logvar_p - logvar_q + (var_q/var_p) + torch.pow(mean_p-mean_q,2)/var_p).sum()

        elif reduce_sample_dim:
                DKL=[]
                for mean_q,logvar_q,mean_p,logvar_p in zip(amean_q,alogvar_q,amean_p,alogvar_p):
                        var_q = torch.exp(logvar_q)
                        var_p = torch.exp(logvar_p)
                        DKL.append(0.5 * (-1 + logvar_p - logvar_q + (var_q/var_p) + torch.pow(mean_p-mean_q,2)/var_p)).sum(1)
        else:
                DKL=[]
                for mean_q,logvar_q in zip(amean_q,alogvar_q):
                        var_q = torch.exp(logvar_q)
                        var_p = torch.exp(logvar_p)
                        DKL.append(0.5 * (-1 + logvar_p - logvar_q + (var_q/var_p) + torch.pow(mean_p-mean_q,2)/var_p))
        return DKL

