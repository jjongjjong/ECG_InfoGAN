import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import os
import matplotlib.pyplot as plt
import PIL
from torchvision.transforms import ToTensor
import io

class NormalNLLLoss:  # THIS IS THE WAY TO CUSTERMIZE LOSS FUNCTION!!!
    def __call__(self,x,mu,var):
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())
        return nll


def noise_sample(n_dis_c,dis_c_dim,n_con_c,n_z,batch_size,device):
    z = torch.randn(batch_size,n_z,1,device=device) #noise is generated by channel unit
    idx = np.zeros((n_dis_c,batch_size))

    if(n_dis_c !=0):
        dis_c = torch.zeros(batch_size,n_dis_c,dis_c_dim,device=device)

        for i in range(n_dis_c):
            idx[i] = np.random.randint(dis_c_dim,size=batch_size) # discrete values
            dis_c[torch.arange(0,batch_size),i,idx[i]] = 1.0 # discrete valuse to one-hot vector, This is for input data

        dis_c = dis_c.view(batch_size,-1,1)
        z = torch.cat([z,dis_c],dim=1)


    if(n_con_c !=0):
        con_c = torch.rand(batch_size,n_con_c,1,device=device)*2-1
        z = torch.cat([z,con_c],dim=1)
    return z, idx


def absoluteFilePaths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

def filepathList_gen(directory:str,sample_run:bool) -> list:
    filepathList = []
    pathgen = absoluteFilePaths(directory)

    if sample_run:
        for i,path in enumerate(pathgen):
            filepathList.append(path)
            if i>1025:
                break
    else :
        filepathList = [path for path in pathgen]

    return filepathList

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def gen_plot(data,idx):
    plt.figure()
    plt.plot(data)
    plt.title(idx)
    buf = io.BytesIO()
    plt.savefig(buf,format='jpeg')
    buf.seek(0)

    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.close()
    return image
