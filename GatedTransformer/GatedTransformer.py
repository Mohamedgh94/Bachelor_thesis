import torch
import torch.nn as nn
import torch.nn.functional as F

from embedding import EmbeddingLayer 
from encoder import EncoderLayer


class GatedTransformer(nn.Module):
    def __init__(self, input_dim=45, d_model=512, num_heads=8, d_ff=2048, num_layers=6):
        super(GatedTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim , d_model )
        print(f'self.embedding.weight.shape',self.embedding.weight.shape)
        self.encoders = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.classifier = nn.Linear(d_model, 5)  # 5 outputs for person, age, height, weight, and gender

    def forward(self, x):  # Unindented to match with __init__
        print(f"Input shape: {x.shape}")  # Debugging line
        x = self.embedding(x)
        #print(f"After embedding shape: {x.shape}")  # Debugging line
        x = x.unsqueeze(0)  # Introduce a sequence length dimension of 1
        #print(f"After unsqueeze shape: {x.shape}")  # Debugging line
        for encoder in self.encoders:
            x = encoder(x)
            #print(f"After encoder shape: {x.shape}")  # Debugging line
        x = x.squeeze(0)  # Remove the sequence length dimension
        # print(f"After squeeze shape: {x.shape}")  # Debugging line
        x = self.classifier(x)
        #print(f"Output shape: {x.shape}")  # Debugging line
        return x

        """
    def forward(self, x):
        print(f"Input shape: {x.shape}")  # Debugging line
        
        x = self.embedding(x)
        print(f"After embedding shape: {x.shape}")  # Debugging line
        
        for encoder in self.encoders:
            x = x.transpose(0, 1)
            x = encoder(x)
            x = x.transpose(1, 0)
            print(f"After encoder shape: {x.shape}")  # Debugging line
        
        #x = x.mean(dim=1)
    
        #print(f"After mean shape: {x.shape}")  # Debugging line
        #x = x.view(x.shape[0], -1)
        #print(f"After view shape: {x.shape}")
        x = self.classifier(x)
        print(f"Output shape: {x.shape}")  # Debugging line
        
        return x
        """
