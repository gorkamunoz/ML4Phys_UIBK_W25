import torch
from os.path import isfile

class model:
    def __init__(self):
        pass
    
    def pred(self, x):
        mag = x.flatten(1,2).sum(-1) 
        return  (mag.abs() > x.shape[-1]**2/2).to(float)
