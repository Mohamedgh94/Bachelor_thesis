import torch
import torch.nn as nn
import torch.nn.functional as F

from feedForward import FeedForward
from gatingMechanism import GatingMechanism
from multiHeadAttention import MultiHeadAttention

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.gating = GatingMechanism(d_model)
        
    def forward(self, x):
        att_output = self.attention(x, x, x)
        x = x + self.gating(att_output)
        ff_output = self.feed_forward(x)
        x = x + self.gating(ff_output)
        return x
