import torch
import torch.nn as nn
import torch.nn.functional as F




""" class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(d_model, num_heads)
        
    def forward(self, q, k, v):
        # Linear projections
        q = self.q_linear(q)  # Shape: [batch_size, seq_len, d_model]
        k = self.k_linear(k)
        v = self.v_linear(v)

        # Transpose for multi-head attention: from [batch_size, seq_len, d_model] to [seq_len, batch_size, d_model]
        q = q.transpose(0, 1)  # Now shape: [seq_len, batch_size, d_model]
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        # Apply the multi-head attention
        output, _ = self.attention(q, k, v)

        return output
 """
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(d_model, num_heads)
        
    def forward(self, q, k, v):
         # Transpose q, k, v to [batch_size, seq_len, d_model]
        #print(f"q shape: {q.shape}")
        q = q.transpose(0, 1)
        #print(f"k shape: {k.shape}")
        k = k.transpose(0, 1)
        # print(f"v shape: {v.shape}")
        v = v.transpose(0, 1)
         # Apply linear transformations
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)


        q = q.transpose(1, 0)
        k = k.transpose(1, 0)
        v = v.transpose(1, 0)

        output, _ = self.attention(q, k, v)
        return output 
