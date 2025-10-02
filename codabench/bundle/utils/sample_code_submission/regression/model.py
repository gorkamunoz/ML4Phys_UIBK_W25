
import torch
from os.path import isfile
import numpy as np

class model:
    def __init__(self):
        pass

    def _tamsd(self, trajs, t_lags):
        tamsd = np.zeros((len(t_lags), trajs.shape[0]), dtype= float)
        
        for idx, tlag in enumerate(t_lags):                  
            tamsd[idx, :] = ((trajs[:, tlag:, :]-trajs[:, :-tlag, :])**2).sum(-1).mean(1)

        return tamsd

    
    def pred(self, trajs):

        trajs = trajs.unsqueeze(-1)

        N_t_lags = max(4, int(trajs.shape[1]*0.1))
        t_lags = np.arange(1, N_t_lags)

        tamsd = self._tamsd(trajs, t_lags)

        return np.polyfit(np.log(t_lags), np.log(tamsd), deg = 1)[0]