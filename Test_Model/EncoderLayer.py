import torch
import torch.nn as nn
import torch.nn.functional as F
from FeedForward import FeedForward
from MultiHeadAttention import MultiHeadAttention

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Apply Multihead Attention
        att_output = self.attention(x)
        # Add & Norm
        x = self.layer_norm_1(x + att_output)
        # Apply Feed Forward
        ff_output = self.feed_forward(x)
        # Add & Norm
        x = self.layer_norm_2(x + ff_output)
        return x