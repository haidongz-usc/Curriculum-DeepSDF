import torch.nn as nn
import torch.functional as F
import torch

class soft_L1(nn.Module):
    def __init__(self):
        super(soft_L1, self).__init__()

    def forward(self, input, target, eps=0.0):
        ret = torch.abs(input - target) - eps
        ret = torch.clamp(ret, min=0.0, max=100.0)
        return ret
