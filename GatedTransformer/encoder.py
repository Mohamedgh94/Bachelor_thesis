""" import torch
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
 """
import torch
import torch.nn as nn

from feedForward import FeedForward
from gatingMechanism import GatingMechanism
from multiHeadAttention import MultiHeadAttention

""" class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.gating = GatingMechanism(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        att_output = self.attention(x, x, x)
        att_output = self.dropout(att_output)
        x = self.layer_norm1(x + self.gating(att_output))

        ff_output = self.feed_forward(x)
        ff_output = self.dropout(ff_output)
        x = self.layer_norm2(x + self.gating(ff_output))
        return x
 """
import torch
import torch.nn as nn

from feedForward import FeedForward
from gatingMechanism import GatingMechanism
from multiHeadAttention import MultiHeadAttention

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.gating = GatingMechanism(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Multi-head attention
        att_output = self.attention(x, x, x)
        att_output = self.dropout(att_output)  # Apply dropout to attention output
        x = self.layer_norm1(x + self.gating(att_output))  # Add & Norm with gating

        # Feed-forward
        ff_output = self.feed_forward(x)
        ff_output = self.dropout(ff_output)  # Apply dropout to feed-forward output
        x = self.layer_norm2(x + self.gating(ff_output))  # Add & Norm with gating
        return x