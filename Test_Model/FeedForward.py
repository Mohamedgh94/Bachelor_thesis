import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x
