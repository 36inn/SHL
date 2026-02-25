import numpy as np
from torch.utils.data import Dataset
#import torch                    #可以注释
#import os                       #可以注释
#from torch.utils.data import DataLoader #可以注释
#from torch.nn import DataParallel       #可以注释
def permute_rows(x):
    '''
    x is a np array
    '''
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]


def uni_instance_gen(n_j, n_m, low, high,seed=None):
    if seed != None:
        np.random.seed(seed)

    time0 = np.random.randint(low=low, high=high, size=(n_j, n_m,n_m-1))
    time1=np.random.randint(low=1, high=high, size=(n_j, n_m,1))
    times=np.concatenate((time0,time1),-1)

    for i in range(n_j):
        times[i]= permute_rows(times[i])

    return times


class FJSPDataset(Dataset):
    def __init__(self,n_j, n_m, low, high,num_samples=1000000,seed=None,  offset=0, distribution=None):
        super(FJSPDataset, self).__init__()
        self.data_set = []

        if seed != None:
            np.random.seed(seed)
        time0 = np.random.uniform(low=low, high=high, size=(num_samples, n_j, n_m, n_m - 1))
        time1 = np.random.uniform(low=0, high=high, size=(num_samples, n_j, n_m, 1))
        times = np.concatenate((time0, time1), -1)
        for j in range(num_samples):
            for i in range(n_j):
                times[j][i] = permute_rows(times[j][i])

            # Sample points randomly in [0, 1] square
        self.data = np.array(times)
        self.size = len(self.data)
    def getdata(self):
        return self.data

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


def override(fn):
    """
    override decorator
    """
    return fn
