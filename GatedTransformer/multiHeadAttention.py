import torch
import torch.nn as nn
import torch.nn.functional as F




class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(d_model, num_heads)
        
    def forward(self, q, k, v):
        print(q.shape)
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)
        output, _ = self.attention(q, k, v)
        return output
