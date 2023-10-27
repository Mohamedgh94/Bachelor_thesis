import torch
import torch.nn as nn
import torch.nn.functional as F

from embedding import EmbeddingLayer 
from encoder import EncoderLayer



class GatedTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers):
        super(GatedTransformer, self).__init__()
        self.embedding = EmbeddingLayer(vocab_size, d_model)
        self.encoders = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.classifier = nn.Linear(d_model, 5)  # 5 outputs for person, age, height, weight, and gender
        
    def forward(self, x):
        x = self.embedding(x)
        for encoder in self.encoders:
            x = encoder(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x
