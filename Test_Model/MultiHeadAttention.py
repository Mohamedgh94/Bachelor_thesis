import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(d_model, num_heads)
        
    def forward(self, x):
        # Assuming x is of shape [seq_len, batch_size, d_model]
        q = k = v = x
        # Apply attention on the transposed matrix
        q = k = v = q.transpose(0, 1)  # Transpose for PyTorch's attention [batch_size, seq_len, d_model]
        output, _ = self.attention(q, k, v, need_weights=False)
        return output.transpose(0, 1)  # Transpose back to original shape
