import torch
from torch.utils.data import Dataset
import biosppy
import numpy as np
import pandas as pd
import os
import pickle

class ECG_dataset (Dataset):
    def __init__(self,filepathList:list,freq:float,sample_step:float,length:int,norm:bool) -> (torch.Tensor,torch.FloatTensor,torch.IntTensor) :
        self.filepathList = filepathList
        self.freq = freq
        self.sample_step = sample_step
        self.length = length
        self.norm = norm

    def __len__(self):
        return len(self.filepathList)

    def __getitem__(self, idx):
        ecg_arr = pickle.load(open(self.filepathList,'rb'))
        ecg_mean = 0
        ecg_std = 1
        if self.norm:
            ecg_mean = ecg_arr.mean(keepdims=True)
            ecg_std = ecg_arr.std(keepdims=True)
            ecg_arr = ecg_arr - ecg_mean/ecg_std

        return torch.from_numpy(ecg_arr),torch.tensor(ecg_mean[0]),torch.tensor(ecg_std[0])








