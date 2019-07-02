import torch
if torch.__version__ != '0.4.0':
        raise RuntimeError('PyTorch version must be 0.4.0')
import torch.nn as nn


def apply_linear(inp,out,act,shape=None,std=0.0,drop=0.0,bn=True):
        bias = False if bn else True

        w=nn.Linear(inp,out,bias=bias)
        if bn:
                wBN=nn.BatchNorm1d(out)
        activation=return_activation(act)
        if drop!=0:
                dropl=nn.Dropout(drop)
        if std!=0.0:
                assert shape!=None
                nlayer=add_gaussian(shape,std)

        #define the sequential
        forward_list=[w]
        if bn:
                forward_list.append(wBN)

        if std!=0.0:
                forward_list.append(nlayer)

        forward_list.append(activation)

        if drop!=0:
                forward_list.append(dropl)

        return nn.Sequential(*forward_list)


#class for linear activation
class Linear_act(nn.Module):
        def __init__(self):
                super(Linear_act, self).__init__()
        def forward(self,x):
                return x

def linear(x):
        return x

def return_activation(act,dim=1):
        if act=='relu':
                return nn.ReLU()
        elif act=='linear':
                return Linear_act()
        elif act=='softmax':
                return nn.Softmax(dim)
        elif act=='sigmoid':
                return nn.Sigmoid()
        elif act=='tanh':
                return nn.Tanh()
        else:
                raise NotImplemented

