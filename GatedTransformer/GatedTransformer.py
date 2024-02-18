""" import torch
import torch.nn as nn
import torch.nn.functional as F

from embedding import EmbeddingLayer 
from encoder import EncoderLayer

class GatedTransformer(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, d_ff, num_layers,
                 num_person_ids, num_ages, num_heights, num_weights, num_genders):
        super(GatedTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        #self.encoders = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        # self.person_ids_classifier = nn.Linear(d_model, num_person_ids)
        self.age_classifier = nn.Linear(d_model, num_ages)
        self.height_classifier = nn.Linear(d_model, num_heights)
        self.weight_classifier = nn.Linear(d_model, num_weights)
        self.gender_classifier = nn.Linear(d_model, num_genders)

        self.encoders = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.gate = nn.Linear(num_genders+num_person_ids, d_model)
    def forward(self, x):
        print('shape before emb:', x.shape)
        x = self.embedding(x)
        x = x.unsqueeze(0)  # Introduce a sequence length dimension of 1

        for encoder in self.encoders:
            x = encoder(x)

        x = x.squeeze(0)  # Remove the sequence length dimension

        # Output raw logits for classification tasks
        # person_id_logits = self.person_ids_classifier(x)
        age_logits = self.age_classifier(x)
        height_logits = self.height_classifier(x)
        weight_logits = self.weight_classifier(x)
        gender_logits = self.gender_classifier(x)

        
        outputs = {
            'age': age_logits,
            'height': height_logits,
            'weight': weight_logits,
            'gender': gender_logits
        }
        return outputs
 """

""" 
 """
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import EncoderLayer
from feedForward import FeedForward
from gatingMechanism import GatingMechanism
from multiHeadAttention import MultiHeadAttention

class GatedTransformer(nn.Module):
    def __init__(self,input_dim, d_model, num_heads, d_ff, num_layers,config,num_classes,dropout_rate=0.1):
        super(GatedTransformer, self).__init__()
        self.config = config
        self.embedding = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoders = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)])

        
        self.person_ids_classifier = nn.Linear(d_model,num_classes)
        self.age_classifier = nn.Linear(d_model, 2)
        self.height_classifier = nn.Linear(d_model, 2)
        self.weight_classifier = nn.Linear(d_model, 2)
        self.gender_classifier = nn.Linear(d_model, 2)

        # # Gating mechanism (if applicable)
        #self.gate = nn.Linear(d_model)

    def forward(self, x):
        print (f'initializing data shape : {x.shape}')
        x = self.embedding(x)
        print (f'data shape after embedding : {x.shape}')
        #x = self.positional_encoding(x)
        print (f'data shape after pos_encod : {x.shape}')
        x = self.dropout(x)  # Apply dropout after embedding
        x = self.layer_norm(x)  # Apply layer normalization

        x = x.unsqueeze(0)  # Introduce a sequence length dimension of 1

        for encoder in self.encoders:
            x = encoder(x)

        x = x.squeeze(0)  # Remove the sequence length dimension

        if self.config['output_type'] == 'softmax':
            person_id_output = F.softmax(self.person_ids_classifier(x),dim=1) 
            return person_id_output
        elif self.config['output_type'] == 'attribute':
            age = torch.sigmoid(self.age_classifier(x))
            height = torch.sigmoid(self.height_classifier(x))
            weight = torch.sigmoid(self.weight_classifier(x))
            gender = torch.sigmoid(self.gender_classifier(x))
            return age,height,weight,gender
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x