import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding(x)
