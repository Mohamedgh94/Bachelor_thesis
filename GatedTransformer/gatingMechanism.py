import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingMechanism(nn.Module):
    def __init__(self, d_model):
        super(GatingMechanism, self).__init__()
        self.gate = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        z = torch.sigmoid(self.gate(x))
        return z * x
