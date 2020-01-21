import torch
from torch.utils.data import Dataset
import biosppy
import numpy as np
import pandas as pd
import os
import pickle

class ECG_dataset (Dataset):
    def __init__(self,filepathList:list,freq:float,sample_step:float,length:int,norm:str) -> (torch.Tensor,torch.FloatTensor,torch.IntTensor) :
        self.filepathList = filepathList
        self.freq = freq
        self.sample_step = sample_step
        self.length = length
        self.norm = norm
        print('origin {}hz, sample step {}, output {}hz, data len {}, norm {}'.format(freq,sample_step,freq/sample_step,length,norm))

    def __len__(self):
        return len(self.filepathList)

    def __getitem__(self, idx):
        ecg_arr = pickle.load(open(self.filepathList[idx],'rb'))
        ecg_mean = 0
        ecg_std = 1
        if self.norm=='zero':
            ecg_mean = ecg_arr.mean(keepdims=True)
            ecg_std = ecg_arr.std(keepdims=True)
            ecg_arr = (ecg_arr - ecg_mean)/(ecg_std+2e-100)
        elif self.norm=='minmax':
            ecg_min = ecg_arr.min(keepdims=True)
            ecg_max = ecg_arr.max(keepdims=True)
            ecg_arr = (ecg_arr-ecg_min)/(ecg_max-ecg_min+2e-100)

        ecg_arr = ecg_arr[::self.sample_step]

        assert len(ecg_arr<=self.length), print('ecg_arr[{}] is shorten than the length [{}]'.format(len(ecg_arr,self.length)))
        ecg_arr = ecg_arr[:self.length].reshape(1,-1)

        return torch.from_numpy(ecg_arr).float(),torch.tensor(ecg_mean[0]),torch.tensor(ecg_std[0])








